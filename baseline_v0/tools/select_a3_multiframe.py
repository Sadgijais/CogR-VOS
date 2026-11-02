#!/usr/bin/env python3
"""
Task 6 candidate selection, arm A3 (Amendment 6 item 4): multi-frame selection without a VLM. Needs no ground truth. GPU (SAM 2) plus CLIP.

For every sentence of the cached Grounding-DINO folder with at least 2 boxes, and for each of the cached boxes (up to 5):
  1. start SAM 2.1 Hiera-tiny (same model, checkpoint, device and offload settings as V0, read from --config) with that box as the prompt at frame 0 and
     follow it through the first N frames (frame 0 plus N - 1 more, counted in the dataset's own annotated frame list); the candidate is tracked ALONE
     (its own fresh SAM 2 state, so candidates cannot suppress each other and SAM 2's rule "no new object after tracking started" is respected). The state
     is built on a temporary folder that links only the first N frames of the video (SAM 2 is causal: later frames cannot change these masks);
  2. on each of the N frames: a mask with fewer than 200 pixels scores 0; otherwise crop the image around the mask (8 px padding, grey background outside
     the mask, common.embed.masked_crop), take its CLIP ViT-B/32 image embedding and score = cosine with the CLIP text embedding of the sentence text exactly
     as stored in the grounding file (no prompt template);
  3. the candidate's score is the mean over its N frame scores; the highest wins; a tie goes to the higher detector score (the lower list position).
A video with fewer than N frames uses all of its frames (n_frames_used is logged).
SAM 2 is causal, so the first 5 frames of the 10-frame track are the 5-frame track: ONE pass gives both arms (--ns 5 10, the defaults); scores are
the mean over the first N frames of the same tracks. The N-frame tracks are thrown away: for each N a COPY of the grounding folder is written
(<out-prefix>_n5, <out-prefix>_n10) with the chosen box first and the others in their old order; V0 is then run again from frame 0 on that copy.
Also writes selection_a3.csv (scores, chosen positions, cost per sentence) and timing_a3.csv (per video) into the first output folder, and logs the
extra computation: SAM 2 frames and CLIP passes. The source folder is never modified; existing output folders are refused. The ground truth is never opened.

  python -I tools/select_a3_multiframe.py --config <run>/config.yaml --root ~/data/MeViS-subset --grounding <run>/grounding --out-prefix <run>/grounding_a3
  (smoke test on one video: add --video 16ae95660b3a and use an out-prefix in a scratch folder)
"""
import argparse, csv, json, os, shutil, sys, tempfile, time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

try:
    import torch
except ImportError:                      # the pure logic can be tested without torch
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MIN_PIXELS = 200


def sync():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def choose(scores):
    """Index of the highest score; an exact tie goes to the lower index (= higher detector score in the sorted cache)."""
    best = 0
    for i, s in enumerate(scores):
        if s > scores[best]:
            best = i
    return best


def track_candidate(predictor, frames_dir, box, n_frames, cfg):
    """Boolean masks of the first n_frames frames for ONE box prompted at frame 0, in its own fresh SAM 2 state (same calls and settings as stage 2)."""
    state = predictor.init_state(video_path=str(frames_dir), offload_video_to_cpu=cfg["offload_video_to_cpu"],
                                 offload_state_to_cpu=cfg["offload_state_to_cpu"])
    predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1, box=np.array(box[:4], dtype=np.float32))
    masks = {}
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=n_frames - 1):
        for obj_id, logits in zip(obj_ids, mask_logits):
            masks[frame_idx] = (logits > 0.0).squeeze().cpu().numpy()
        if frame_idx >= n_frames - 1:
            break
    predictor.reset_state(state)
    return masks


def link_first_frames(frame_paths, n, folder):
    """A folder holding only the first n frames of a video under their own names (links; a copy if linking is not possible)."""
    for p in frame_paths[:n]:
        dst = os.path.join(folder, p.name)
        try:
            os.symlink(str(p), dst)
        except OSError:
            shutil.copy(str(p), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the run's config.yaml: only the propagation block is used (same SAM 2 settings as V0)")
    ap.add_argument("--root", required=True, help="dataset root with JPEGImages/<tier>/")
    ap.add_argument("--grounding", required=True, help="source folder with <video>.json (never modified)")
    ap.add_argument("--out-prefix", required=True, help="output folders <prefix>_n<N> are created (must not exist yet)")
    ap.add_argument("--ns", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--tier", default="480p")
    ap.add_argument("--pad", type=int, default=8)
    ap.add_argument("--clip-device", default="cpu", help="cpu (default, keeps the 4 GB GPU for SAM 2) or cuda")
    ap.add_argument("--video", default=None, help="smoke test: only this video (the output folders then hold only this video)")
    a = ap.parse_args()

    from common.embed import ClipCropEmbedder, TextEmbedder, masked_crop

    cfg = yaml.safe_load(open(Path(a.config).expanduser()))["propagation"]
    root, src = Path(a.root).expanduser(), Path(a.grounding).expanduser()
    prefix = Path(a.out_prefix).expanduser()
    ns = sorted(set(a.ns))
    n_max = max(ns)
    outs = {n: Path(f"{prefix}_n{n}") for n in ns}
    for n, o in outs.items():
        if o.exists():
            sys.exit(f"refusing to write into an existing folder: {o}")
    for o in outs.values():
        o.mkdir(parents=True)

    from sam2.build_sam import build_sam2_video_predictor
    t_start = time.time()
    print(f"Loading SAM2 ({cfg['model_cfg']}) ...")
    predictor = build_sam2_video_predictor(cfg["model_cfg"], cfg["checkpoint"], device=cfg["device"])
    text_emb, img_emb = TextEmbedder(device=a.clip_device), ClipCropEmbedder(device=a.clip_device)

    sel_rows, tim_rows = [], []
    tot = {"sentences": 0, "scored_sentences": 0, "sam_frames": 0, "clip_passes": 0, "sam_s": 0.0, "clip_s": 0.0}
    changed = {n: 0 for n in ns}
    for gfile in sorted(src.glob("*.json")):
        video = gfile.stem
        if a.video and video != a.video:
            continue
        data = json.load(open(gfile))
        frame_paths = sorted((root / "JPEGImages" / a.tier / video).glob("*.jpg"))
        n_use = min(n_max, len(frame_paths))
        images = [Image.open(p).convert("RGB") for p in frame_paths[:n_use]]
        tmp = tempfile.TemporaryDirectory(prefix="a3_frames_")
        link_first_frames(frame_paths, n_use, tmp.name)
        v_sam, v_clip = 0.0, 0.0
        new = {n: json.loads(json.dumps(data)) for n in ns}
        for exp_id, entry in data.items():
            tot["sentences"] += 1
            boxes = entry["boxes"]
            row = {"video": video, "exp_id": exp_id, "n_candidates": len(boxes), "n_frames_used": n_use if len(boxes) > 1 else 0}
            for n in ns:
                row[f"chosen_pos_n{n}"] = 0
                row[f"mean_scores_n{n}"] = ""
            row.update({"frame_scores": "", "detector_scores": "", "sam_s": 0.0, "clip_s": 0.0, "sam_frames": 0, "clip_passes": 0})
            if len(boxes) > 1:
                tot["scored_sentences"] += 1
                t = text_emb([entry["expression"]])[0]
                per_cand, sam_s, clip_s, sam_frames, passes = [], 0.0, 0.0, 0, 0
                for b in boxes:
                    sync(); t0 = time.perf_counter()
                    masks = track_candidate(predictor, tmp.name, b, n_use, cfg)
                    sync(); sam_s += time.perf_counter() - t0
                    sam_frames += n_use - 1
                    t0 = time.perf_counter()
                    scores = [0.0] * n_use
                    crops, where = [], []
                    for k in range(n_use):
                        m = masks.get(k)
                        if m is not None and int(m.sum()) >= MIN_PIXELS:
                            crops.append(masked_crop(images[k], m, a.pad)); where.append(k)
                    if crops:
                        emb = img_emb.embed_images(crops)
                        for k, c in zip(where, emb @ t):
                            scores[k] = float(c)
                        passes += len(crops)
                    clip_s += time.perf_counter() - t0
                    per_cand.append(scores)
                for n in ns:
                    nn = min(n, n_use)
                    means = [float(np.mean(s[:nn])) for s in per_cand]
                    best = choose(means)
                    row[f"chosen_pos_n{n}"] = best
                    row[f"mean_scores_n{n}"] = ";".join(f"{x:.5f}" for x in means)
                    if best != 0:
                        new[n][exp_id]["boxes"] = [boxes[best]] + boxes[:best] + boxes[best + 1:]
                        changed[n] += 1
                row["frame_scores"] = "|".join(",".join(f"{x:.4f}" for x in s) for s in per_cand)
                row["detector_scores"] = ";".join(f"{float(b[4]):.3f}" for b in boxes)
                row.update({"sam_s": round(sam_s, 3), "clip_s": round(clip_s, 3), "sam_frames": sam_frames, "clip_passes": passes})
                v_sam += sam_s; v_clip += clip_s
                tot["sam_frames"] += sam_frames; tot["clip_passes"] += passes; tot["sam_s"] += sam_s; tot["clip_s"] += clip_s
            sel_rows.append(row)
        tmp.cleanup()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        for n in ns:
            json.dump(new[n], open(outs[n] / gfile.name, "w"), indent=2)
        tim_rows.append({"video": video, "n_frames_used": n_use, "sam_s": round(v_sam, 3), "clip_s": round(v_clip, 3)})
        print(f"{video}: {len(data)} sentences, SAM 2 {v_sam:.1f} s (includes the state set-up), CLIP {v_clip:.1f} s")

    first = outs[ns[0]]
    fields = list(sel_rows[0].keys())
    with open(first / "selection_a3.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(sel_rows)
    with open(first / "timing_a3.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tim_rows[0].keys()))
        w.writeheader()
        w.writerows(tim_rows)
    secs = time.time() - t_start
    sc = max(tot["scored_sentences"], 1)
    print(f"A3 multi-frame (SAM 2 on {cfg['device']}, CLIP on {a.clip_device}), N = {ns}: {tot['sentences']} sentences, {tot['scored_sentences']} with more than one box")
    print("box moved to the front: " + ", ".join(f"N={n}: {changed[n]}" for n in ns))
    print(f"extra computation: {tot['sam_frames']} SAM 2 frames, {tot['clip_passes']} CLIP passes; SAM 2 {tot['sam_s']:.1f} s ({tot['sam_s'] / sc:.2f} s per sentence), "
          f"CLIP {tot['clip_s']:.1f} s ({tot['clip_s'] / sc:.2f} s per sentence); {secs:.1f} s in total including model loading")
    print("written: " + ", ".join(str(o) for o in outs.values()))


if __name__ == "__main__":
    sys.exit(main())
