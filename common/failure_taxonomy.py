"""
The drift / recovery / false-absence taxonomy behind failure_analysis.csv.

These definitions are written down in CogR-VOS_V0_Baseline_Spec.md and the
deck (Deck v2, Slide 19) and are reproduced here verbatim so V0 and every
later CogR-VOS variant (V1..V5) emit directly comparable numbers:

  ID drift event
    IoU on the target < `drift_low` (default 0.10) while the target is
    genuinely visible in the ground truth, AND IoU on some OTHER annotated
    object in the same video > `drift_high` (default 0.50), for a run of
    contiguous frames. Contiguous frames count as ONE event, not many.
    Both conditions are required — low IoU alone is just a lost track;
    high IoU on another object alone could be incidental overlap.

  Recovery
    After a genuine disappearance-then-reappearance in the ground truth
    (a run of frames where the target's GT mask is empty, bounded on both
    sides by frames where it is visible), the target is "recovered" if the
    prediction's IoU with the target reaches >= `recovery_iou` (default
    0.50) within `recovery_window` (default 10) frames of the return frame.

  False-absence frame (a.k.a. NRE, not-reported error)
    A frame where the ground truth shows the target as visible (non-empty)
    but the prediction is empty — i.e. the model wrongly declared the
    target gone. Frames that fall inside an active "waiting to recover"
    window right after a genuine reappearance are counted under recovery
    delay instead, not double-counted here.

Failure-mode classification (per video, per object/expression) is our own
diagnostic layer on top of these primitives — see `classify_failure_mode`
below for the exact rule and its documented limits.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import iou

DEFAULT_DRIFT_LOW = 0.10
DEFAULT_DRIFT_HIGH = 0.50
DEFAULT_RECOVERY_IOU = 0.50
DEFAULT_RECOVERY_WINDOW = 10
DEFAULT_GROUNDING_FAILURE_IOU = 0.10
DEFAULT_TRACK_LOST_MEAN_IOU = 0.30
DEFAULT_POOR_MASK_GAP = 0.15


def _is_empty(mask: np.ndarray) -> bool:
    return not mask.any()


def find_drift_events(
    pred_masks: list[np.ndarray],
    gt_target_masks: list[np.ndarray],
    other_gt_masks: dict[str, list[np.ndarray]],
    drift_low: float = DEFAULT_DRIFT_LOW,
    drift_high: float = DEFAULT_DRIFT_HIGH,
) -> list[dict]:
    """
    Returns a list of drift events: [{"onset": frame_idx, "onto": obj_id,
    "length": n_frames}, ...]. `other_gt_masks` maps other object ids in
    the same video to their per-frame GT mask lists (same length/alignment
    as gt_target_masks).
    """
    n = len(gt_target_masks)
    flagged = [False] * n
    onto_for_frame: dict[int, str] = {}

    for t in range(n):
        if _is_empty(gt_target_masks[t]):
            continue  # target genuinely not there — not a drift question
        target_iou = iou(pred_masks[t], gt_target_masks[t])
        if target_iou >= drift_low:
            continue
        best_obj, best_iou = None, 0.0
        for obj_id, masks in other_gt_masks.items():
            if t >= len(masks):
                continue
            other_iou = iou(pred_masks[t], masks[t])
            if other_iou > best_iou:
                best_obj, best_iou = obj_id, other_iou
        if best_obj is not None and best_iou > drift_high:
            flagged[t] = True
            onto_for_frame[t] = best_obj

    events = []
    t = 0
    while t < n:
        if flagged[t]:
            onset = t
            onto = onto_for_frame[t]
            length = 0
            while t < n and flagged[t]:
                length += 1
                t += 1
            events.append({"onset": onset, "onto": onto, "length": length})
        else:
            t += 1
    return events


def find_reappearance_events(gt_target_masks: list[np.ndarray]) -> list[dict]:
    """
    A reappearance event is a maximal run of GT-empty frames that is
    preceded AND followed by a GT-visible frame (so the video both starts
    and ends with the target visible relative to that run — a target that
    is absent from frame 0, or stays absent to the last frame, is not a
    "disappearance-then-reappearance" by this definition).

    Returns [{"absence_start": t0, "absence_end": t1, "return_frame": t2}]
    where absence runs from t0..t1 inclusive and t2 = t1 + 1 is the first
    frame the target is visible again.
    """
    n = len(gt_target_masks)
    visible = [not _is_empty(m) for m in gt_target_masks]
    events = []
    t = 0
    while t < n:
        if not visible[t]:
            start = t
            while t < n and not visible[t]:
                t += 1
            end = t - 1
            has_before = start > 0 and visible[start - 1]
            has_after = t < n and visible[t]
            if has_before and has_after:
                events.append({"absence_start": start, "absence_end": end, "return_frame": t})
        else:
            t += 1
    return events


@dataclass
class RecoveryStats:
    n_events: int = 0
    n_recovered: int = 0
    delays: list = field(default_factory=list)
    recovery_windows: list = field(default_factory=list)  # [(start, end_exclusive), ...]

    @property
    def recovery_rate(self) -> float:
        return (self.n_recovered / self.n_events) if self.n_events else float("nan")

    @property
    def mean_delay(self) -> float:
        return float(np.mean(self.delays)) if self.delays else float("nan")


def evaluate_recovery(
    pred_masks: list[np.ndarray],
    gt_target_masks: list[np.ndarray],
    recovery_iou: float = DEFAULT_RECOVERY_IOU,
    recovery_window: int = DEFAULT_RECOVERY_WINDOW,
) -> RecoveryStats:
    events = find_reappearance_events(gt_target_masks)
    stats = RecoveryStats(n_events=len(events))
    n = len(gt_target_masks)
    for ev in events:
        ret = ev["return_frame"]
        window_end = min(ret + recovery_window, n)
        recovered_at = None
        for t in range(ret, window_end):
            if iou(pred_masks[t], gt_target_masks[t]) >= recovery_iou:
                recovered_at = t
                break
        if recovered_at is not None:
            stats.n_recovered += 1
            stats.delays.append(recovered_at - ret)
            stats.recovery_windows.append((ret, recovered_at))
        else:
            # Not recovered within the window — the whole window still
            # counts as "waiting", so it is excluded from false-absence.
            stats.recovery_windows.append((ret, window_end))
    return stats


def count_false_absence_frames(
    pred_masks: list[np.ndarray],
    gt_target_masks: list[np.ndarray],
    recovery_windows: list[tuple[int, int]],
) -> int:
    """
    Frames where GT is visible but the prediction is empty, excluding
    frames already accounted for by an active recovery window (those are
    reported via recovery delay, not double-counted here).
    """
    in_window = np.zeros(len(gt_target_masks), dtype=bool)
    for start, end in recovery_windows:
        in_window[start:end] = True

    count = 0
    for t, (pred, gt) in enumerate(zip(pred_masks, gt_target_masks)):
        if in_window[t]:
            continue
        if not _is_empty(gt) and _is_empty(pred):
            count += 1
    return count


def classify_failure_mode(
    mean_j: float,
    mean_f: float,
    first_visible_iou: float,
    n_drift_events: int,
    grounding_failure_iou: float = DEFAULT_GROUNDING_FAILURE_IOU,
    track_lost_mean_iou: float = DEFAULT_TRACK_LOST_MEAN_IOU,
    poor_mask_gap: float = DEFAULT_POOR_MASK_GAP,
) -> str:
    """
    One label per (video, object). This is a diagnostic heuristic, not a
    precise measurement — read failure_analysis.csv's raw numbers before
    trusting the label on any single borderline case.

      grounding_failure       — the very first visible frame was already
                                 wrong (mean_j on that frame < threshold).
                                 Upstream of everything CogR-VOS adds.
      identity_drift          — >=1 drift event was detected. This is the
                                 failure mode CogR-VOS is designed to fix.
      track_lost_no_distractor— mean quality is low but no drift event
                                 fired (nothing to blame it on) — a
                                 candidate for the coherence estimator to
                                 catch even without a named distractor.
      poor_mask_quality       — right object, region score reasonable, but
                                 boundary score lags noticeably behind it
                                 (sloppy edges) — a SAM2 ceiling issue, not
                                 something memory/reasoning fixes.
      ok                      — none of the above fired.
    """
    j = mean_j / 100.0
    f = mean_f / 100.0

    if first_visible_iou < grounding_failure_iou:
        return "grounding_failure"
    if n_drift_events > 0:
        return "identity_drift"
    if j < track_lost_mean_iou:
        return "track_lost_no_distractor"
    if (j - f) > poor_mask_gap:
        return "poor_mask_quality"
    return "ok"
