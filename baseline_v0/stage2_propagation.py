#!/usr/bin/env python3
"""
Stage 2 — propagation.

Reads the grounding cache from stage 1, takes the TOP-1 box per expression
(V0 does not use the other 4 — they are cached for V4's re-identification
step later), prompts SAM2's video predictor on frame 0, and propagates the
mask to the end of the video. Writes one palette-style PNG per frame under
predictions/<video>/<exp_id>/.

Timing (added after the V0 freeze, does not change any mask):
results/timing_per_video.csv and results/timing_summary.txt.

Usage:
    python stage2_propagation.py --config config.yaml
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from common.dataset import DavisLayout
from common.io_utils import load_json, write_binary_mask


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def box_to_prompt(box):
    """[x1,y1,x2,y2,score] -> the (x1,y1,x2,y2) box SAM2 expects."""
    x1, y1, x2, y2, _score = box
    return [x1, y1, x2, y2]


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


TIMING_FIELDS = ["video", "n_frames", "n_objects", "init_s", "propagate_s",
                 "inference_s", "ms_per_frame", "peak_alloc_mb", "peak_reserved_mb"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    layout = DavisLayout(root=Path(cfg["dataset"]["root"]), require_480p=cfg["evaluation"]["require_480p"])
    grounding_dir = Path(cfg["paths"]["grounding_out"])
    pred_dir = Path(cfg["paths"]["predictions_out"])
    results_dir = Path(cfg["paths"]["results_out"])
    results_dir.mkdir(parents=True, exist_ok=True)

    grounding_files = sorted(grounding_dir.glob("*.json"))
    if not grounding_files:
        raise FileNotFoundError(
            f"No grounding cache found in {grounding_dir}/. Run "
            "stage1_grounding.py first — stage 2 only reads its output, it "
            "never calls Grounding-DINO itself."
        )

    import torch
    from sam2.build_sam import build_sam2_video_predictor

    print(f"Loading SAM2 ({cfg['propagation']['model_cfg']}) ...")
    _sync()
    t_load = time.perf_counter()
    predictor = build_sam2_video_predictor(
        cfg["propagation"]["model_cfg"],
        cfg["propagation"]["checkpoint"],
        device=cfg["propagation"]["device"],
    )
    _sync()
    model_load_s = time.perf_counter() - t_load
    timing_rows = []

    for gfile in grounding_files:
        video = gfile.stem
        video_grounding = load_json(gfile)
        frames_dir = layout.jpeg_dir / video
        frame_paths = layout.frame_paths(video)
        n_frames = len(frame_paths)

        print(f"Propagating {video}: {len(video_grounding)} expression(s), {n_frames} frames")

        # offload flags: required to fit SAM2's memory bank on a 4 GB card.
        torch.cuda.reset_peak_memory_stats()
        _sync()
        t_start = time.perf_counter()
        state = predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=cfg["propagation"]["offload_video_to_cpu"],
            offload_state_to_cpu=cfg["propagation"]["offload_state_to_cpu"],
        )
        _sync()
        init_s = time.perf_counter() - t_start

        # batch_objects: all expressions as distinct object ids in ONE state.
        exp_ids = list(video_grounding.keys())
        sam_obj_id_of = {}
        for i, exp_id in enumerate(exp_ids, start=1):
            entry = video_grounding[exp_id]
            if not entry["boxes"]:
                print(f"  WARNING: {video}/{exp_id} has no grounding boxes at all — "
                      f"Grounding-DINO found nothing above threshold for "
                      f"'{entry['expression']}'. Emitting an empty mask for "
                      f"every frame (this IS the correct V0 behaviour for "
                      f"'no candidate' — see config.yaml grounding thresholds "
                      f"if this looks wrong for too many expressions).")
                sam_obj_id_of[exp_id] = None
                continue
            top1 = box_to_prompt(entry["boxes"][0])
            sam_obj_id = i
            sam_obj_id_of[exp_id] = sam_obj_id
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=sam_obj_id,
                box=np.array(top1, dtype=np.float32),
            )
            if not cfg["propagation"]["batch_objects"]:
                # One state per expression instead — simpler, slower.
                # (Left here for clarity; default path is batched.)
                pass

        # Propagate and write out masks.
        _sync()
        t_prop = time.perf_counter()
        video_masks: dict[int, dict[int, np.ndarray]] = {}
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            for obj_id, logits in zip(obj_ids, mask_logits):
                mask = (logits > 0.0).squeeze().cpu().numpy()
                video_masks.setdefault(obj_id, {})[frame_idx] = mask
        _sync()
        propagate_s = time.perf_counter() - t_prop
        inference_s = time.perf_counter() - t_start
        peak_alloc_mb = torch.cuda.max_memory_allocated() / 2**20
        peak_reserved_mb = torch.cuda.max_memory_reserved() / 2**20
        timing_rows.append({
            "video": video, "n_frames": n_frames,
            "n_objects": sum(v is not None for v in sam_obj_id_of.values()),
            "init_s": round(init_s, 3), "propagate_s": round(propagate_s, 3),
            "inference_s": round(inference_s, 3),
            "ms_per_frame": round(propagate_s / n_frames * 1000.0, 2),
            "peak_alloc_mb": round(peak_alloc_mb, 1),
            "peak_reserved_mb": round(peak_reserved_mb, 1),
        })
        print(f"  timing: {inference_s:.1f}s total, "
              f"{propagate_s / n_frames * 1000.0:.0f} ms/frame, "
              f"peak VRAM {peak_alloc_mb:.0f} MB")

        for exp_id in exp_ids:
            sam_obj_id = sam_obj_id_of[exp_id]
            out_dir = pred_dir / video / exp_id
            for frame_idx, frame_path in enumerate(frame_paths):
                if sam_obj_id is None:
                    mask = np.zeros(_frame_hw(frame_path), dtype=bool)
                else:
                    mask = video_masks.get(sam_obj_id, {}).get(
                        frame_idx, np.zeros(_frame_hw(frame_path), dtype=bool)
                    )
                out_name = frame_path.stem + ".png"
                write_binary_mask(out_dir / out_name, mask)

        predictor.reset_state(state)

    del predictor
    torch.cuda.empty_cache()
    _write_timing(results_dir, timing_rows, model_load_s)
    print(f"Stage 2 done. Wrote predictions under {pred_dir}/")


def _write_timing(results_dir: Path, rows: list, model_load_s: float) -> None:
    with open(results_dir / "timing_per_video.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TIMING_FIELDS)
        w.writeheader()
        w.writerows(rows)
    frames = sum(r["n_frames"] for r in rows)
    prop = sum(r["propagate_s"] for r in rows)
    lines = [
        f"videos: {len(rows)}",
        f"frames: {frames}",
        f"model_load_s: {model_load_s:.1f}",
        f"total_inference_s: {sum(r['inference_s'] for r in rows):.1f}",
        f"overall_ms_per_frame: {prop / frames * 1000.0:.1f}",
        f"median_video_ms_per_frame: {statistics.median(r['ms_per_frame'] for r in rows):.1f}",
        f"max_peak_alloc_mb: {max(r['peak_alloc_mb'] for r in rows):.0f}",
        f"max_peak_reserved_mb: {max(r['peak_reserved_mb'] for r in rows):.0f}",
        "note: first video includes CUDA warm-up; stage 1 (grounding) excluded",
    ]
    (results_dir / "timing_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _frame_hw(frame_path: Path):
    from PIL import Image

    with Image.open(frame_path) as img:
        w, h = img.size
    return (h, w)


if __name__ == "__main__":
    main()
