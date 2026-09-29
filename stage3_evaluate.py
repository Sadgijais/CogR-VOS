#!/usr/bin/env python3
"""
Stage 3 — evaluation.

Compares every prediction under predictions/<video>/<exp_id>/ against the
ground-truth palette PNGs, computes J and F per (video, expression), and
writes:
  results/per_video_results.csv   — one row per (video, expression, obj_id)
  results/summary.csv             — aggregated mean J / F / J&F

This stage has NO dependency on GPU, Grounding-DINO, or SAM2 — it is pure
numpy/PIL and can be exercised end-to-end on any machine, which is exactly
what tools/test_fixture.py does (and what was actually run and verified in
the environment that wrote this code — see README.md "What has and hasn't
been tested here").

Usage:
    python stage3_evaluate.py --config config.yaml
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from common.dataset import DavisLayout, load_expressions
from common.io_utils import read_palette_mask, read_binary_mask
from common.metrics import sequence_scores


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_one(
    layout: DavisLayout,
    video: str,
    exp_id: str,
    obj_id: int,
    pred_dir: Path,
    exclude_first_last: bool,
    bound_th: int,
) -> dict:
    anno_paths = layout.anno_paths(video)
    pred_paths_dir = pred_dir / video / exp_id
    if not anno_paths:
        raise FileNotFoundError(f"no annotation frames for {video}")

    gt_masks = [read_palette_mask(p, obj_id) for p in anno_paths]
    pred_masks = []
    for anno_p in anno_paths:
        pred_p = pred_paths_dir / (anno_p.stem + ".png")
        if pred_p.exists():
            pred_masks.append(read_binary_mask(pred_p))
        else:
            # Missing prediction file is treated as "predicted empty" —
            # NOT skipped. A silently-skipped frame would inflate the
            # score by only counting the easy frames.
            pred_masks.append(np.zeros_like(gt_masks[-1]))

    scores = sequence_scores(pred_masks, gt_masks, exclude_first_last, bound_th)
    scores["video"] = video
    scores["exp_id"] = exp_id
    scores["obj_id"] = obj_id
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    layout = DavisLayout(root=Path(cfg["dataset"]["root"]), require_480p=cfg["evaluation"]["require_480p"])
    pred_dir = Path(cfg["paths"]["predictions_out"])
    results_dir = Path(cfg["paths"]["results_out"])
    results_dir.mkdir(parents=True, exist_ok=True)

    annotator_set = cfg["dataset"].get("annotator_set")
    expressions = load_expressions(layout, annotator_set)

    rows = []
    for e in expressions:
        pred_exp_dir = pred_dir / e.video / e.exp_id
        if not pred_exp_dir.exists():
            print(f"skip {e.video}/{e.exp_id}: no predictions found (run stage 2 first)")
            continue
        try:
            scores = evaluate_one(
                layout,
                e.video,
                e.exp_id,
                e.obj_id,
                pred_dir,
                cfg["evaluation"]["exclude_first_last"],
                cfg["evaluation"]["bound_th_px"],
            )
        except Exception as exc:
            print(f"ERROR evaluating {e.video}/{e.exp_id}: {exc}")
            continue
        rows.append(
            {
                "video": e.video,
                "exp_id": e.exp_id,
                "obj_id": e.obj_id,
                "expression": e.text,
                "n_frames_scored": scores["n_frames_scored"],
                "J": round(scores["J"], 2),
                "F": round(scores["F"], 2),
                "JF": round(scores["JF"], 2),
            }
        )

    per_video_path = results_dir / "per_video_results.csv"
    with open(per_video_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video", "exp_id", "obj_id", "expression", "n_frames_scored", "J", "F", "JF"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {per_video_path}")

    if rows:
        mean_j = float(np.mean([r["J"] for r in rows]))
        mean_f = float(np.mean([r["F"] for r in rows]))
        mean_jf = float(np.mean([r["JF"] for r in rows]))
    else:
        mean_j = mean_f = mean_jf = float("nan")

    target = cfg["target"]["ref_davis17_jf"]
    summary_path = results_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "n_expressions_scored", "value": len(rows)})
        writer.writerow({"metric": "n_videos_scored", "value": len({r["video"] for r in rows})})
        writer.writerow({"metric": "mean_J", "value": round(mean_j, 2)})
        writer.writerow({"metric": "mean_F", "value": round(mean_f, 2)})
        writer.writerow({"metric": "mean_JF", "value": round(mean_jf, 2)})
        writer.writerow({"metric": "target_JF_ref_davis17", "value": target})
        writer.writerow({"metric": "delta_vs_target", "value": round(mean_jf - target, 2)})

    print(f"Wrote {summary_path}")
    print(f"\nmean J&F = {mean_jf:.2f}  (target: {target})")
    if mean_jf > 72:
        print(
            "NOTE: this is notably ABOVE the 66.2 target. Per "
            "V0_Baseline_Spec.md §6.3, an unexpectedly high V0 number is a "
            "reason for suspicion, not celebration — V0 has no mechanism "
            "that could beat AL-Ref-SAM2 (74.2). Check for frame-0 leakage, "
            "wrong annotator set, or exclude_first_last being silently off."
        )
    elif mean_jf < 20:
        print(
            "NOTE: this is very low. Per V0_Baseline_Spec.md §6.3, check "
            "mask orientation, resolution mismatch, and whether "
            "predictions are all empty before assuming the models are just "
            "bad at this."
        )


if __name__ == "__main__":
    main()
