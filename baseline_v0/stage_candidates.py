#!/usr/bin/env python3
"""
Stage candidates (V4, step 6): the disk cache of "who could the target be on frame t" for every frame of the
expressions that can trigger a re-ID search.

Why a cache: on a 4 GB card Grounding-DINO and the SAM 2 video tracker cannot be loaded together, and the V4 live
loop (stage_reid.py) needs the SAM 2 tracker. So everything the live loop needs from Grounding-DINO and from SAM 2's
image decoder is computed here, once, and read from disk later. Every V4 variant (off, oracle, memory, VLM) reuses it,
which also makes the variants comparable: they all choose among the same candidates.

What is stored, per (video, expression), for every frame f >= 1 (frame 0 is the prompt):
  the top-K Grounding-DINO boxes for the expression on that frame (same model and thresholds as stage 1; imported from
  stage1_grounding.py, not copied) and, for each box, a SAM 2 mask from a box prompt on that single frame
  (SAM2ImagePredictor with the SAME checkpoint as the tracker).

Which expressions: those that fire at least one V3 event (read from V3's events_per_expression.csv). On Ref-DAVIS17 val
that is 37 of 61 expressions (2115 of 3923 frames). The other 24 keep their V2 passive masks and need no candidates.

Two passes so the two models are never resident together:
  pass A  Grounding-DINO boxes for every frame      -> <out>/boxes/<video>.json
  pass B  SAM 2 masks for those boxes               -> <out>/<video>/<exp_id>.npz
Both passes resume: finished files are skipped (use --force to redo).

Usage (from baseline_v0/):
    python stage_candidates.py --config config_v2_final.yaml --events-csv results_v3_gemini_events_abstain/events_per_expression.csv --video india
    python stage_candidates.py --config config_v2_final.yaml --events-csv results_v3_gemini_events_abstain/events_per_expression.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

EVENT_COLS = ("reappear", "disappear", "coherence_drop", "distractor", "drift_suspect")


# ------------------------------------------------------------------ pure logic (unit-tested, no GPU)
def select_expressions(events_csv: Path, only_videos=None):
    """-> sorted [(video, exp_id, expression_text)] for expressions with >= 1 V3 trigger event."""
    out = []
    with open(events_csv) as f:
        for r in csv.DictReader(f):
            if only_videos and r["video"] not in only_videos:
                continue
            if sum(int(r[c]) for c in EVENT_COLS) > 0:
                out.append((r["video"], r["exp_id"], r["expression"]))
    return sorted(out)


def pack_masks(masks: np.ndarray) -> np.ndarray:
    """(N, K, H, W) bool -> (N, K, ceil(H*W/8)) uint8."""
    n, k, h, w = masks.shape
    return np.packbits(masks.reshape(n, k, h * w), axis=-1)


def unpack_masks(packed: np.ndarray, hw) -> np.ndarray:
    h, w = int(hw[0]), int(hw[1])
    n, k, _ = packed.shape
    return np.unpackbits(packed, axis=-1, count=h * w).reshape(n, k, h, w).astype(bool)


def save_store(path: Path, frames, boxes, packed, hw, K: int):
    """frames: list[int]; boxes: list over frames of list of [x1,y1,x2,y2,score] (len <= K);
    packed: list over frames of list of 1-D uint8 arrays (np.packbits of the flattened (H, W) mask),
    same length as the boxes of that frame (masks_pass returns exactly this)."""
    n, h, w = len(frames), int(hw[0]), int(hw[1])
    b = np.zeros((n, K, 4), np.float32)
    s = np.zeros((n, K), np.float32)
    cnt = np.zeros(n, np.int8)
    nbytes = (h * w + 7) // 8
    m = np.zeros((n, K, nbytes), np.uint8)
    for i in range(n):
        cnt[i] = len(boxes[i][:K])
        for j, bx in enumerate(boxes[i][:K]):
            b[i, j], s[i, j] = bx[:4], bx[4]
            m[i, j] = packed[i][j]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, frames=np.asarray(frames, np.int32), count=cnt, boxes=b, scores=s,
                        masks=m, hw=np.asarray([h, w], np.int32))
    tmp.replace(path)


class CandidateStore:
    """Read side, used by the V4 live loop. get(f) -> list of {"box": [x1,y1,x2,y2], "score": float, "mask": bool (H,W)}
    ordered by Grounding-DINO score (best first). has(f) is False for a frame that was never cached."""

    def __init__(self, path: Path):
        z = np.load(path)
        self.frames = [int(x) for x in z["frames"]]
        self._row = {f: i for i, f in enumerate(self.frames)}
        self._count, self._boxes, self._scores = z["count"], z["boxes"], z["scores"]
        self._packed, self.hw = z["masks"], tuple(int(x) for x in z["hw"])

    def has(self, f: int) -> bool:
        return int(f) in self._row

    def get(self, f: int):
        i = self._row[int(f)]
        k = int(self._count[i])
        masks = unpack_masks(self._packed[i:i + 1, :k], self.hw)[0] if k else []
        return [{"box": [float(x) for x in self._boxes[i, j]], "score": float(self._scores[i, j]), "mask": masks[j]}
                for j in range(k)]


def boxes_pass(frame_paths, exprs, frames, ground_fn):
    """exprs: {exp_id: text}. -> {exp_id: {str(f): [[x1,y1,x2,y2,score], ...]}}"""
    return {eid: {str(f): [list(map(float, b)) for b in ground_fn(frame_paths[f], text)] for f in frames}
            for eid, text in exprs.items()}


def masks_pass(frame_paths, boxes, frames, segmenter):
    """boxes: {exp_id: {str(f): [[x1,y1,x2,y2,score], ...]}}. Calls segmenter.set_frame(path) once per frame
    (shared by all expressions of the video), then segmenter.mask(box) per box.
    -> {exp_id: [per frame in `frames` order: list of packed masks (np.packbits of the flattened bool mask)]}
    Masks are packed at once (1/8 of the memory), so a long video with several expressions stays small."""
    out = {eid: [] for eid in boxes}
    for f in frames:
        segmenter.set_frame(frame_paths[f])
        for eid in boxes:
            out[eid].append([np.packbits(np.asarray(segmenter.mask(b[:4]), bool).reshape(-1)) for b in boxes[eid][str(f)]])
    return out


# ------------------------------------------------------------------ GPU glue (not unit-tested; run on the real data)
def _frame_hw(p: Path):
    from PIL import Image
    with Image.open(p) as im:
        return im.size[1], im.size[0]


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v2_final.yaml")
    ap.add_argument("--events-csv", required=True)
    ap.add_argument("--out-dir", default="candidates")
    ap.add_argument("--video", action="append", default=None)
    ap.add_argument("--stride", type=int, default=1, help="cache every N-th frame (1 = every frame; keep 1 for real runs)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from common.dataset import DavisLayout
    from common.io_utils import dump_json, load_json

    cfg = yaml.safe_load(open(args.config))
    K = int(cfg["grounding"]["top_k_boxes"])
    layout = DavisLayout(root=Path(cfg["dataset"]["root"]), require_480p=cfg["evaluation"]["require_480p"])
    out = Path(args.out_dir)
    sel = select_expressions(Path(args.events_csv), set(args.video) if args.video else None)
    if not sel:
        raise SystemExit("no expressions with a trigger event matched (check --events-csv and --video)")
    by_video: dict = {}
    for v, e, t in sel:
        by_video.setdefault(v, {})[e] = t
    print(f"candidates for {len(sel)} expressions in {len(by_video)} videos; K={K}, stride={args.stride}")

    def frames_of(video):
        n = len(layout.frame_paths(video))
        return list(range(1, n, args.stride))

    # ---------------- pass A: Grounding-DINO boxes
    todo_a = [v for v in by_video if args.force or not (out / "boxes" / f"{v}.json").exists()]
    if todo_a:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        import stage1_grounding as S1
        mid = cfg["grounding"]["model_id"]
        print(f"pass A: loading {mid} ...")
        S1.run_grounding_dino._processor = AutoProcessor.from_pretrained(mid)
        S1.run_grounding_dino._model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(
            cfg["grounding"]["device"]).eval()
        ground = lambda p, text: S1.run_grounding_dino(p, text, cfg)
        for v in todo_a:
            t0 = time.perf_counter()
            fps = layout.frame_paths(v)
            res = boxes_pass(fps, by_video[v], frames_of(v), ground)
            dump_json(out / "boxes" / f"{v}.json", res)
            empty = sum(1 for e in res.values() for b in e.values() if not b)
            print(f"  {v}: {len(by_video[v])} expr x {len(frames_of(v))} frames, frames with no box: {empty}, "
                  f"{time.perf_counter() - t0:.0f}s")
        del S1.run_grounding_dino._model
        torch.cuda.empty_cache()

    # ---------------- pass B: SAM 2 masks from those boxes
    todo_b = [v for v in by_video if args.force or any(not (out / v / f"{e}.npz").exists() for e in by_video[v])]
    if todo_b:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from PIL import Image
        print(f"pass B: loading SAM 2 ({cfg['propagation']['model_cfg']}) ...")
        model = build_sam2(cfg["propagation"]["model_cfg"], cfg["propagation"]["checkpoint"],
                           device=cfg["propagation"]["device"])
        pred = SAM2ImagePredictor(model)

        class Seg:
            def set_frame(self, path):
                pred.set_image(np.array(Image.open(path).convert("RGB")))

            def mask(self, box):
                m, _, _ = pred.predict(box=np.asarray(box, np.float32), multimask_output=False)
                return m[0] > 0

        seg = Seg()
        for v in todo_b:
            t0 = time.perf_counter()
            fps, frames = layout.frame_paths(v), frames_of(v)
            boxes = load_json(out / "boxes" / f"{v}.json")
            boxes = {e: boxes[e] for e in by_video[v]}
            with torch.inference_mode(), torch.autocast(cfg["propagation"]["device"], dtype=torch.bfloat16):
                masks = masks_pass(fps, boxes, frames, seg)
            hw = _frame_hw(fps[0])
            for e in by_video[v]:
                save_store(out / v / f"{e}.npz", frames, [boxes[e][str(f)] for f in frames], masks[e], hw, K)
            print(f"  {v}: masks done, {time.perf_counter() - t0:.0f}s")
        del pred, model
        torch.cuda.empty_cache()
    print(f"done. {sum(len(x) for x in by_video.values())} expression files under {out}/")


if __name__ == "__main__":
    main()
