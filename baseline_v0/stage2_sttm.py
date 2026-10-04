#!/usr/bin/env python3
"""
Stage 2 (V1) — SAM2 propagation with Semantic-Temporal Target Memory.

Same prompts, same model, same settings as V0 (stage2_propagation.py). The only
difference: after every frame, each expression's TargetMemory decides whether to
store that frame, and SAM2's conditioning set is kept equal to
{frame 0 (anchor)} + {curated frames}  (read action A, see sttm_sam2.py).

  --no-sttm   run this exact loop with the memory switched off. Masks must be
              byte-identical to V0's: that proves the wrapper itself changes nothing.

Writes, next to the PNG predictions:
  <results>/timing_per_video.csv   wall time, ms/frame, peak VRAM, STTM overhead
  <results>/timing_summary.txt
  <results>/sttm_per_expression.csv   writes / merges / evictions / blocked_by-reason / events
  <results>/sttm_log/<video>.json     bank contents, event log, per-write decisions

Usage:
    python stage2_sttm.py --config config_v1.yaml
    python stage2_sttm.py --config config_v1.yaml --video blackswan --video india
    python stage2_sttm.py --config config_v1.yaml --no-sttm --pred-dir predictions_v1_check --results-dir results_v1_check
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common.dataset import DavisLayout
from common.io_utils import load_json, write_binary_mask
from sttm import STTMParams, TargetMemory
from sttm_sam2 import object_score, sync_bank

TIMING_FIELDS = ["video", "n_frames", "n_objects", "init_s", "propagate_s", "inference_s",
                 "sttm_overhead_s", "ms_per_frame", "peak_alloc_mb", "peak_reserved_mb"]
STTM_FIELDS = ["video", "exp_id", "expression", "frames", "writes", "merges", "evictions",
               "blocked_not_visible", "blocked_truncated", "blocked_low_similarity",
               "blocked_paced", "blocked_not_better", "n_absent_events", "n_reappeared_events",
               "final_bank_frames", "min_margin", "missing_frames"]


def box_to_prompt(box):
    x1, y1, x2, y2, _ = box
    return [x1, y1, x2, y2]


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _frame_hw(p: Path):
    with Image.open(p) as im:
        w, h = im.size
    return (h, w)


def mask_confidence(logits_np: np.ndarray, mask: np.ndarray) -> float:
    """Stand-in for SAM2's IoU prediction (not stored in the video state):
    mean sigmoid(logit) over the predicted-mask pixels = how certain the mask is."""
    if not mask.any():
        return 0.0
    return float((1.0 / (1.0 + np.exp(-logits_np[mask]))).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v1.yaml")
    ap.add_argument("--video", action="append", default=None, help="only these videos (repeatable)")
    ap.add_argument("--no-sttm", action="store_true", help="memory off (must reproduce V0 masks)")
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    v1 = cfg.get("v1", {})
    use_sttm = (not args.no_sttm) and bool(v1.get("use_sttm", True))
    params = STTMParams(**v1.get("params", {}))

    layout = DavisLayout(root=Path(cfg["dataset"]["root"]), require_480p=cfg["evaluation"]["require_480p"])
    grounding_dir = Path(cfg["paths"]["grounding_out"])
    emb_dir = Path(v1.get("embeddings_out", "embeddings/"))
    pred_dir = Path(args.pred_dir or cfg["paths"]["predictions_out"])
    results_dir = Path(args.results_dir or cfg["paths"]["results_out"])
    (results_dir / "sttm_log").mkdir(parents=True, exist_ok=True)

    files = sorted(grounding_dir.glob("*.json"))
    if args.video:
        files = [f for f in files if f.stem in set(args.video)]
    if not files:
        raise FileNotFoundError(f"no grounding cache in {grounding_dir}/ (or --video matched nothing)")

    import torch
    from sam2.build_sam import build_sam2_video_predictor

    print(f"STTM {'ON' if use_sttm else 'OFF'}; params={params}")
    print(f"Loading SAM2 ({cfg['propagation']['model_cfg']}) ...")
    predictor = build_sam2_video_predictor(
        cfg["propagation"]["model_cfg"], cfg["propagation"]["checkpoint"],
        device=cfg["propagation"]["device"])
    crop_emb = None
    if use_sttm:
        from common.embed import CropEmbedder
        crop_emb = CropEmbedder(device=cfg["propagation"]["device"])

    timing_rows, sttm_rows = [], []

    for gfile in files:
        video = gfile.stem
        vg = load_json(gfile)
        frame_paths = layout.frame_paths(video)
        n_frames = len(frame_paths)
        H, W = _frame_hw(frame_paths[0])
        print(f"Propagating {video}: {len(vg)} expression(s), {n_frames} frames")

        cache = np.load(emb_dir / f"{video}.npz") if use_sttm else None

        torch.cuda.reset_peak_memory_stats()
        _sync()
        t_start = time.perf_counter()
        state = predictor.init_state(
            video_path=str(layout.jpeg_dir / video),
            offload_video_to_cpu=cfg["propagation"]["offload_video_to_cpu"],
            offload_state_to_cpu=cfg["propagation"]["offload_state_to_cpu"])
        _sync()
        init_s = time.perf_counter() - t_start

        exp_ids = list(vg.keys())
        sam_obj_id_of = {}
        for i, exp_id in enumerate(exp_ids, start=1):
            entry = vg[exp_id]
            if not entry["boxes"]:
                print(f"  WARNING: {video}/{exp_id} has no grounding boxes — empty mask (V0 behaviour)")
                sam_obj_id_of[exp_id] = None
                continue
            sam_obj_id_of[exp_id] = i
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=i,
                box=np.array(box_to_prompt(entry["boxes"][0]), dtype=np.float32))

        # one memory per expression that has a SAM2 object; obj_idx = obj_id - 1
        mems: dict[int, TargetMemory] = {}
        logs: dict[int, list] = {}
        frame_logs: dict[int, list] = {}
        missing_counts: dict[int, int] = {}
        if use_sttm:
            for exp_id in exp_ids:
                oid = sam_obj_id_of[exp_id]
                if oid is None:
                    continue
                boxes_emb = cache[f"{exp_id}__boxes"]
                mems[oid - 1] = TargetMemory(
                    params, (H, W), cache[f"{exp_id}__text"],
                    boxes_emb[1:] if len(boxes_emb) > 1 else None)   # [0] is the target's own box
                logs[oid - 1] = []
                frame_logs[oid - 1] = []
                missing_counts[oid - 1] = 0

        _sync()
        t_prop = time.perf_counter()
        overhead = 0.0
        video_masks: dict[int, dict[int, np.ndarray]] = {}
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            img = None
            for k, (obj_id, logits) in enumerate(zip(obj_ids, mask_logits)):
                lg = logits.squeeze()
                mask = (lg > 0.0).cpu().numpy()
                video_masks.setdefault(obj_id, {})[frame_idx] = mask
                mem = mems.get(k)
                if mem is None:
                    continue
                _sync()
                t0 = time.perf_counter()
                if img is None:
                    img = Image.open(frame_paths[frame_idx]).convert("RGB")
                emb = crop_emb.embed_masked(img, mask) if mask.any() else None
                if frame_idx == 0:
                    if emb is None:
                        del mems[k]          # empty prompt mask: no memory, plain V0 behaviour
                    else:
                        mem.set_anchor(0, mask, emb)
                else:
                    od = state["output_dict_per_obj"][k]
                    out = od["non_cond_frame_outputs"].get(frame_idx) or od["cond_frame_outputs"].get(frame_idx)
                    osc = object_score(out)
                    conf = mask_confidence(lg.float().cpu().numpy(), mask)
                    dec = mem.update(frame_idx, mask, emb, osc, conf)
                    frame_logs[k].append({"f": frame_idx, "obj_score": round(osc, 2), "conf": round(conf, 3),
                                          "sim": None if dec.get("sim") is None else round(dec["sim"], 3),
                                          "result": dec.get("action") or dec.get("blocked_by")})
                    if dec.get("written"):
                        r = sync_bank(state, k, mem.read_frames())
                        missing_counts[k] += len(r["missing"])
                        logs[k].append({"frame": frame_idx, "action": dec["action"],
                                        "bank": mem.read_frames(), **{kk: r[kk] for kk in ("promoted", "demoted")}})
                _sync()
                overhead += time.perf_counter() - t0
        _sync()
        propagate_s = time.perf_counter() - t_prop
        inference_s = time.perf_counter() - t_start
        peak_a = torch.cuda.max_memory_allocated() / 2**20
        peak_r = torch.cuda.max_memory_reserved() / 2**20
        timing_rows.append({
            "video": video, "n_frames": n_frames,
            "n_objects": sum(v is not None for v in sam_obj_id_of.values()),
            "init_s": round(init_s, 3), "propagate_s": round(propagate_s, 3),
            "inference_s": round(inference_s, 3), "sttm_overhead_s": round(overhead, 3),
            "ms_per_frame": round(propagate_s / n_frames * 1000.0, 2),
            "peak_alloc_mb": round(peak_a, 1), "peak_reserved_mb": round(peak_r, 1)})
        print(f"  timing: {inference_s:.1f}s total ({overhead:.1f}s STTM overhead), "
              f"{propagate_s / n_frames * 1000.0:.0f} ms/frame, peak VRAM {peak_a:.0f} MB")

        # ---- write masks (not timed)
        for exp_id in exp_ids:
            oid = sam_obj_id_of[exp_id]
            for fi, fp in enumerate(frame_paths):
                if oid is None:
                    m = np.zeros((H, W), bool)
                else:
                    m = video_masks.get(oid, {}).get(fi, np.zeros((H, W), bool))
                write_binary_mask(pred_dir / video / exp_id / (fp.stem + ".png"), m)

        # ---- STTM logs
        vlog = {}
        for exp_id in exp_ids:
            oid = sam_obj_id_of[exp_id]
            if oid is None or (oid - 1) not in mems:
                continue
            k = oid - 1
            s = mems[k].summary()
            st = s["stats"]
            ev = s["events"]
            margin = s["context"].get("margin")
            vlog[exp_id] = {"expression": vg[exp_id]["expression"], "summary": {
                kk: (vv if kk != "velocity" else list(vv)) for kk, vv in s.items() if kk != "events"},
                "events": ev, "writes": logs[k], "frames": frame_logs[k]}
            sttm_rows.append({
                "video": video, "exp_id": exp_id, "expression": vg[exp_id]["expression"],
                "frames": st["frames"], "writes": st["writes"], "merges": st["merges"],
                "evictions": st["evictions"],
                **{f"blocked_{n}": st["blocked"][n] for n in
                   ("not_visible", "truncated", "low_similarity", "paced", "not_better")},
                "n_absent_events": sum(e["event"] == "absent" for e in ev),
                "n_reappeared_events": sum(e["event"] == "reappeared" for e in ev),
                "final_bank_frames": " ".join(map(str, s["bank_frames"])),
                "min_margin": "" if margin is None else round(margin, 3),
                "missing_frames": missing_counts[k]})
        if vlog:
            (results_dir / "sttm_log" / f"{video}.json").write_text(json.dumps(vlog, indent=2, default=float))

        predictor.reset_state(state)

    del predictor
    torch.cuda.empty_cache()

    with open(results_dir / "timing_per_video.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TIMING_FIELDS); w.writeheader(); w.writerows(timing_rows)
    if sttm_rows:
        with open(results_dir / "sttm_per_expression.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=STTM_FIELDS); w.writeheader(); w.writerows(sttm_rows)
    frames = sum(r["n_frames"] for r in timing_rows)
    prop = sum(r["propagate_s"] for r in timing_rows)
    lines = [f"sttm: {'on' if use_sttm else 'off'}", f"videos: {len(timing_rows)}", f"frames: {frames}",
             f"total_inference_s: {sum(r['inference_s'] for r in timing_rows):.1f}",
             f"total_sttm_overhead_s: {sum(r['sttm_overhead_s'] for r in timing_rows):.1f}",
             f"overall_ms_per_frame: {prop / frames * 1000.0:.1f}",
             f"median_video_ms_per_frame: {statistics.median(r['ms_per_frame'] for r in timing_rows):.1f}",
             f"max_peak_alloc_mb: {max(r['peak_alloc_mb'] for r in timing_rows):.0f}",
             f"max_peak_reserved_mb: {max(r['peak_reserved_mb'] for r in timing_rows):.0f}"]
    (results_dir / "timing_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Stage 2 (V1) done. Predictions under {pred_dir}/")


if __name__ == "__main__":
    main()
