#!/usr/bin/env python3
"""
Stage 4 — failure analysis.

For every (video, expression), works out not just HOW WRONG the prediction
is (stage 3 already did that) but HOW it went wrong: did grounding pick the
wrong object from frame 0, did the mask drift onto a distractor mid-video,
did tracking just collapse with nothing to blame, or is the object right
but the mask edges sloppy? This split is the entire reason a baseline is
worth running — see common/failure_taxonomy.py's module docstring for the
exact definitions, reproduced from CogR-VOS_V0_Baseline_Spec.md.

Also computes, per expression: number of ID-drift events (H2 evidence),
recovery rate and mean delay after disappearance (H4 evidence — expect this
to be mostly empty on Ref-DAVIS17, see README), and false-absence frame
counts.

No GPU dependency — pure numpy/PIL, same as stage 3. This is the stage
whose logic was actually exercised end-to-end in this environment via
tools/test_fixture.py.

Usage:
    python stage4_failure_analysis.py --config config.yaml
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
from common.io_utils import read_palette_mask, read_binary_mask, unique_palette_ids
from common.metrics import iou
from common.failure_taxonomy import (
    find_drift_events,
    evaluate_recovery,
    count_false_absence_frames,
    classify_failure_mode,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def analyze_one(
    layout: DavisLayout,
    video: str,
    exp_id: str,
    obj_id: int,
    pred_dir: Path,
    fa_cfg: dict,
) -> dict:
    anno_paths = layout.anno_paths(video)
    pred_exp_dir = pred_dir / video / exp_id

    gt_target_masks = [read_palette_mask(p, obj_id) for p in anno_paths]
    pred_masks = []
    for anno_p in anno_paths:
        pred_p = pred_exp_dir / (anno_p.stem + ".png")
        pred_masks.append(
            read_binary_mask(pred_p) if pred_p.exists() else np.zeros_like(gt_target_masks[-1])
        )

    # Every OTHER object id that appears anywhere in this video's
    # annotations — the candidate pool a drift event could have landed on.
    all_ids: set[int] = set()
    for p in anno_paths:
        all_ids |= unique_palette_ids(p)
    other_ids = sorted(all_ids - {obj_id})
    other_gt_masks = {
        str(oid): [read_palette_mask(p, oid) for p in anno_paths] for oid in other_ids
    }

    drift_events = find_drift_events(
        pred_masks, gt_target_masks, other_gt_masks,
        drift_low=fa_cfg["drift_low"], drift_high=fa_cfg["drift_high"],
    )
    recovery = evaluate_recovery(
        pred_masks, gt_target_masks,
        recovery_iou=fa_cfg["recovery_iou"], recovery_window=fa_cfg["recovery_window"],
    )
    n_false_absence = count_false_absence_frames(
        pred_masks, gt_target_masks, recovery.recovery_windows
    )

    # mean_j / mean_f here are computed WITHOUT exclude_first_last — this
    # is a diagnostic view over every frame, deliberately not the same
    # number as stage 3's headline J&F. See README for why.
    from common.metrics import f_measure

    j_per_frame = np.array([iou(p, g) for p, g in zip(pred_masks, gt_target_masks)])
    f_per_frame = np.array([f_measure(p, g) for p, g in zip(pred_masks, gt_target_masks)])
    mean_j = float(j_per_frame.mean()) * 100
    mean_f = float(f_per_frame.mean()) * 100

    first_visible_idx = next(
        (i for i, m in enumerate(gt_target_masks) if m.any()), 0
    )
    first_visible_iou = float(j_per_frame[first_visible_idx])

    failure_mode = classify_failure_mode(
        mean_j, mean_f, first_visible_iou, len(drift_events),
        grounding_failure_iou=fa_cfg["grounding_failure_iou"],
        track_lost_mean_iou=fa_cfg["track_lost_mean_iou"],
        poor_mask_gap=fa_cfg["poor_mask_gap"],
    )

    return {
        "video": video,
        "exp_id": exp_id,
        "obj_id": obj_id,
        "mean_J": round(mean_j, 2),
        "mean_F": round(mean_f, 2),
        "failure_mode": failure_mode,
        "n_drift_events": len(drift_events),
        "drift_onsets": ";".join(str(ev["onset"]) for ev in drift_events),
        "drift_onto_obj": ";".join(str(ev["onto"]) for ev in drift_events),
        "n_reappearance_events": recovery.n_events,
        "recovery_rate": (
            round(recovery.recovery_rate, 3) if recovery.n_events else ""
        ),
        "mean_recovery_delay": (
            round(recovery.mean_delay, 2) if recovery.delays else ""
        ),
        "n_false_absence_frames": n_false_absence,
    }


FIELDNAMES = [
    "video", "exp_id", "obj_id", "mean_J", "mean_F", "failure_mode",
    "n_drift_events", "drift_onsets", "drift_onto_obj",
    "n_reappearance_events", "recovery_rate", "mean_recovery_delay",
    "n_false_absence_frames",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    layout = DavisLayout(root=Path(cfg["dataset"]["root"]), require_480p=cfg["evaluation"]["require_480p"])
    pred_dir = Path(cfg["paths"]["predictions_out"])
    out_path = Path(cfg["paths"]["failure_analysis_out"])

    annotator_set = cfg["dataset"].get("annotator_set")
    expressions = load_expressions(layout, annotator_set)
    fa_cfg = cfg["failure_analysis"]

    rows = []
    mode_counts: dict[str, int] = {}
    for e in expressions:
        if not (pred_dir / e.video / e.exp_id).exists():
            continue
        try:
            row = analyze_one(layout, e.video, e.exp_id, e.obj_id, pred_dir, fa_cfg)
        except Exception as exc:
            print(f"ERROR analyzing {e.video}/{e.exp_id}: {exc}")
            continue
        rows.append(row)
        mode_counts[row["failure_mode"]] = mode_counts.get(row["failure_mode"], 0) + 1

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}\n")
    print("Failure mode breakdown:")
    for mode, count in sorted(mode_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {mode:26s} {count}")

    n_recoverable = sum(1 for r in rows if r["n_reappearance_events"])
    print(
        f"\n{n_recoverable}/{len(rows)} expressions had at least one "
        "disappearance-then-reappearance event. If this is near zero on "
        "Ref-DAVIS17, that matches CogR-VOS_Dataset_Audit_Findings.md "
        "(4/61 objects vanish, 2/30 videos) — H4 evidence lives on "
        "Long-RVOS, not here. Do not read a blank recovery_rate column as "
        "a bug."
    )


if __name__ == "__main__":
    main()
