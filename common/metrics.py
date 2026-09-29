"""
Region (J) and boundary (F) similarity between two binary masks.

Conventions (decided in CogR-VOS_V0_Baseline_Spec.md, kept identical to the
official DAVIS semi-supervised evaluator where it matters for comparability,
and documented where we deliberately simplify):

- J (region similarity) = IoU. Reported as a percentage 0-100, matching every
  published Ref-DAVIS17 number (e.g. the 66.2 V0 target).
- F (boundary accuracy) = a boundary F-measure, tolerant within `bound_th`
  pixels, falling off monotonically beyond that (NOT a hard cutoff), so a
  mask that is *almost* aligned still scores something rather than zero.
- Both-empty rule: if both the prediction and the ground truth are empty for
  a frame (correctly predicting "target not here"), J = F = 1.0 for that
  frame. This matters a lot for Long-RVOS-style disappearance frames, where
  punishing a correct "nothing here" call would be wrong. It is exactly the
  behaviour called out in the working primer for this harness's F.
- One-empty rule: if exactly one of the two is empty, J = F = 0.0.

This is NOT a byte-for-byte reimplementation of the official DAVIS toolkit's
contour-matching algorithm (that uses bipartite matching over both contours).
It is a simpler, well-defined boundary measure that is monotonic and cheap,
documented as our own choice — say so if a reviewer asks how F was computed.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

DEFAULT_BOUND_TH = 4  # pixels


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return mask.astype(bool)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-union of two binary masks, with the both-empty rule."""
    pred = _as_bool(pred)
    gt = _as_bool(gt)
    if not pred.any() and not gt.any():
        return 1.0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Interior boundary pixels of a binary mask (mask minus its erosion)."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def f_measure(pred: np.ndarray, gt: np.ndarray, bound_th: int = DEFAULT_BOUND_TH) -> float:
    """Boundary F-measure with a monotonically-decaying tolerance band."""
    pred = _as_bool(pred)
    gt = _as_bool(gt)

    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0

    pred_b = _boundary(pred)
    gt_b = _boundary(gt)

    if not pred_b.any() or not gt_b.any():
        # A mask with no interior boundary (e.g. a single-pixel blob) —
        # fall back to IoU rather than dividing by zero.
        return iou(pred, gt)

    # Distance transform of "not boundary" gives, at every pixel, the
    # distance to the nearest boundary pixel of that mask.
    gt_dist = distance_transform_edt(np.logical_not(gt_b))
    pred_dist = distance_transform_edt(np.logical_not(pred_b))

    # Precision: how close is each *predicted* boundary pixel to the nearest
    # *ground-truth* boundary pixel, scored 1.0 at distance 0 falling
    # linearly to 0.0 at distance >= bound_th.
    precision_scores = np.clip(1.0 - gt_dist[pred_b] / bound_th, 0.0, 1.0)
    # Recall: the symmetric question for ground-truth boundary pixels.
    recall_scores = np.clip(1.0 - pred_dist[gt_b] / bound_th, 0.0, 1.0)

    p = float(precision_scores.mean())
    r = float(recall_scores.mean())
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def sequence_scores(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    exclude_first_last: bool = True,
    bound_th: int = DEFAULT_BOUND_TH,
) -> dict:
    """
    J and F for one (video, object) track, averaged over frames.

    `pred_masks` and `gt_masks` must be the same length and frame-aligned.
    When `exclude_first_last` is True (the DAVIS convention — see
    CogR-VOS_V0_Baseline_Spec.md §5.2), the first and last frame of the
    track are dropped before averaging: frame 0 is where the method was
    *given* the answer as a prompt, so scoring it is free marks.

    Returns per-frame arrays too, so stage 4 (failure analysis) does not
    need to recompute IoU from scratch.
    """
    n = len(gt_masks)
    if n != len(pred_masks):
        raise ValueError(f"pred/gt length mismatch: {len(pred_masks)} vs {n}")

    j_per_frame = np.array([iou(pred_masks[i], gt_masks[i]) for i in range(n)])
    f_per_frame = np.array(
        [f_measure(pred_masks[i], gt_masks[i], bound_th) for i in range(n)]
    )

    if exclude_first_last and n > 2:
        idx = np.arange(1, n - 1)
    else:
        idx = np.arange(n)

    mean_j = float(j_per_frame[idx].mean()) if len(idx) else float("nan")
    mean_f = float(f_per_frame[idx].mean()) if len(idx) else float("nan")

    return {
        "J": mean_j * 100.0,
        "F": mean_f * 100.0,
        "JF": (mean_j + mean_f) / 2.0 * 100.0,
        "n_frames_scored": int(len(idx)),
        "j_per_frame": j_per_frame,   # full-length, 0..n-1, NOT trimmed
        "f_per_frame": f_per_frame,   # full-length, 0..n-1, NOT trimmed
        "scored_index": idx,
    }
