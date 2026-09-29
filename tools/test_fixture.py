#!/usr/bin/env python3
"""
Synthetic-data regression test for stages 3 and 4 — the ONLY part of this
harness that can be (and has been) actually run and verified without a GPU
or the real dataset.

It builds a tiny, fully-controlled fake "dataset" on disk (three videos
with known, hand-designed ground truth and predictions), runs the exact
same evaluation and failure-analysis code the real pipeline uses, and
asserts the numbers come out to hand-computed values. If this script
prints "ALL CHECKS PASSED", the J/F math and the drift/recovery/
false-absence taxonomy are implemented correctly — independent of whether
Grounding-DINO or SAM2 ever produce a good mask.

What each fixture video is built to prove:

  clean_video  — prediction == ground truth every frame.
                 Expect J = 100.0 exactly (a perfect track scores perfectly).

  drift_video  — target (object 1) sits in one place for 40 frames; a
                 distractor (object 2) sits in a different, non-overlapping
                 place for the same 40 frames. The prediction matches
                 object 1 exactly for frames 0-19, then matches object 2
                 exactly for frames 20-39 (a textbook identity switch).
                 With frame 0 and frame 39 excluded (DAVIS convention),
                 19 scored frames are perfect and 19 are a total miss:
                 Expect J = 50.0 exactly, and exactly ONE drift event,
                 onset at frame 20, onto object "2".

  gap_video    — target visible frames 0-9, genuinely absent (empty GT)
                 frames 10-14, visible again frames 15-29. The prediction
                 has two unrelated one-off blank frames at 5-6 (while the
                 target IS visible — a false-absence error), correctly
                 predicts empty during the real 10-14 absence, then lags
                 two frames behind the frame-15 reappearance before
                 catching up.
                 Expect: 1 reappearance event, recovered, delay = 2 frames,
                 recovery_rate = 1.0, and exactly 2 false-absence frames
                 (the two blips — the two lag frames are attributed to
                 recovery delay instead, not double-counted).

Usage:
    python tools/test_fixture.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.dataset import DavisLayout
from common.io_utils import write_binary_mask
from common.metrics import sequence_scores
from common.failure_taxonomy import (
    find_drift_events,
    evaluate_recovery,
    count_false_absence_frames,
)

H, W = 50, 50
CHECKS = []


def check(name: str, actual, expected, tol: float = 1e-6):
    ok = abs(actual - expected) <= tol if isinstance(expected, (int, float)) else actual == expected
    CHECKS.append((name, ok, actual, expected))
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: got {actual!r}, expected {expected!r}")


def square_mask(x0, y0, size=10, shape=(H, W)):
    m = np.zeros(shape, dtype=bool)
    m[y0:y0 + size, x0:x0 + size] = True
    return m


def write_palette_frame(path: Path, obj_masks: dict[int, np.ndarray]):
    """obj_masks: {obj_id: bool mask}. Writes one multi-object palette PNG."""
    arr = np.zeros((H, W), dtype=np.uint8)
    for obj_id, mask in obj_masks.items():
        arr[mask] = obj_id
    img = Image.fromarray(arr, mode="P")
    # A minimal palette is enough — read_palette_mask only reads raw indices.
    img.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0] + [0] * (256 * 3 - 9))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def build_video(root: Path, name: str, n_frames: int, obj1_frames, obj2_frames=None):
    """
    obj1_frames / obj2_frames: dict[frame_idx] -> mask (or omit = empty)
    Writes Annotations/480p/<name>/%05d.png for every frame in [0, n_frames).
    """
    obj2_frames = obj2_frames or {}
    for i in range(n_frames):
        masks = {}
        if i in obj1_frames:
            masks[1] = obj1_frames[i]
        if i in obj2_frames:
            masks[2] = obj2_frames[i]
        write_palette_frame(root / "Annotations" / "480p" / name / f"{i:05d}.png", masks)


def build_predictions(pred_root: Path, video: str, exp_id: str, n_frames: int, frame_masks: dict):
    for i in range(n_frames):
        mask = frame_masks.get(i, np.zeros((H, W), dtype=bool))
        write_binary_mask(pred_root / video / exp_id / f"{i:05d}.png", mask)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="cogrvos_fixture_"))
    print(f"Building synthetic fixture under {tmp}")
    try:
        layout = DavisLayout(root=tmp, require_480p=False)
        pred_root = tmp / "predictions"

        obj1_sq = square_mask(0, 0)
        obj2_sq = square_mask(20, 20)
        empty = np.zeros((H, W), dtype=bool)

        # ---- clean_video: 20 frames, perfect prediction throughout ----
        n = 20
        build_video(tmp, "clean_video", n, {i: obj1_sq for i in range(n)})
        build_predictions(pred_root, "clean_video", "0", n, {i: obj1_sq for i in range(n)})

        # ---- drift_video: 40 frames, switches to object 2 at frame 20 ----
        n = 40
        build_video(
            tmp, "drift_video", n,
            {i: obj1_sq for i in range(n)},
            {i: obj2_sq for i in range(n)},
        )
        drift_pred = {i: (obj1_sq if i < 20 else obj2_sq) for i in range(n)}
        build_predictions(pred_root, "drift_video", "0", n, drift_pred)

        # ---- gap_video: 30 frames, disappearance 10-14, two false-absence
        #      blips at 5-6, two-frame recovery lag after reappearance ----
        n = 30
        obj1_gap_frames = {i: obj1_sq for i in range(0, 10)}
        obj1_gap_frames.update({i: obj1_sq for i in range(15, 30)})
        # frames 10-14 deliberately absent from the dict -> empty GT
        build_video(tmp, "gap_video", n, obj1_gap_frames)

        gap_pred = {i: obj1_sq for i in range(0, 5)}       # 0-4 correct
        # 5,6 blank (false absence)
        gap_pred.update({i: obj1_sq for i in range(7, 10)})  # 7-9 correct
        # 10-14 blank (correctly matches genuine absence)
        # 15,16 blank (recovery lag)
        gap_pred.update({i: obj1_sq for i in range(17, 30)})  # 17-29 correct
        build_predictions(pred_root, "gap_video", "0", n, gap_pred)

        # ================= stage 3 style checks =================
        print("\nStage 3 (J/F) checks:")

        def gt_masks_for(video, obj_id, n_frames):
            return [
                np.array(Image.open(tmp / "Annotations" / "480p" / video / f"{i:05d}.png")) == obj_id
                for i in range(n_frames)
            ]

        def pred_masks_for(video, exp_id, n_frames):
            from common.io_utils import read_binary_mask
            return [
                read_binary_mask(pred_root / video / exp_id / f"{i:05d}.png")
                for i in range(n_frames)
            ]

        clean_scores = sequence_scores(
            pred_masks_for("clean_video", "0", 20), gt_masks_for("clean_video", 1, 20),
            exclude_first_last=True,
        )
        check("clean_video J", round(clean_scores["J"], 4), 100.0)
        check("clean_video F", round(clean_scores["F"], 4), 100.0)

        drift_scores = sequence_scores(
            pred_masks_for("drift_video", "0", 40), gt_masks_for("drift_video", 1, 40),
            exclude_first_last=True,
        )
        check("drift_video J", round(drift_scores["J"], 4), 50.0)

        # ================= stage 4 style checks =================
        print("\nStage 4 (failure taxonomy) checks:")

        drift_gt1 = gt_masks_for("drift_video", 1, 40)
        drift_gt2 = gt_masks_for("drift_video", 2, 40)
        drift_pred_masks = pred_masks_for("drift_video", "0", 40)

        events = find_drift_events(drift_pred_masks, drift_gt1, {"2": drift_gt2})
        check("drift_video n_drift_events", len(events), 1)
        check("drift_video drift onset", events[0]["onset"] if events else None, 20)
        check("drift_video drift onto", events[0]["onto"] if events else None, "2")

        gap_gt1 = gt_masks_for("gap_video", 1, 30)
        gap_pred_masks = pred_masks_for("gap_video", "0", 30)

        recovery = evaluate_recovery(gap_pred_masks, gap_gt1)
        check("gap_video n_reappearance_events", recovery.n_events, 1)
        check("gap_video recovery_rate", round(recovery.recovery_rate, 4), 1.0)
        check("gap_video mean_delay", round(recovery.mean_delay, 4), 2.0)

        n_false_absence = count_false_absence_frames(gap_pred_masks, gap_gt1, recovery.recovery_windows)
        check("gap_video n_false_absence_frames", n_false_absence, 2)

        no_drift = find_drift_events(gap_pred_masks, gap_gt1, {})
        check("gap_video n_drift_events (none expected)", len(no_drift), 0)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{sum(1 for _, ok, *_ in CHECKS if ok)}/{len(CHECKS)} checks passed.")
    if all(ok for _, ok, *_ in CHECKS):
        print("ALL CHECKS PASSED — stage 3/4 metric and failure-taxonomy code is correct")
        print("on this synthetic data. This does NOT verify stage 1 (Grounding-DINO) or")
        print("stage 2 (SAM2) — those need a real GPU and the real dataset to exercise.")
        return 0
    else:
        print("SOME CHECKS FAILED — do not trust stage 3/4 output until these pass.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
