"""
Tracklet Coherence Score (TCS) — V2.

Pure numpy, no GPU, no SAM2 import: testable on fake data (tools/test_tcs.py).
Design: V2_TUNING_PLAN.md.

One number c_t in [0,1] per frame per target, answering "is this track still
trustworthy?", built from four cheap terms. Nothing here calls a VLM.

  mask        SAM 2's own certainty (object score, mean mask probability) AND how
              much the mask shape agrees with the previous visible mask
  motion      how far the observed centroid is from where the memory's velocity
              predicted it (size-normalised)
  appearance  DINOv2 cosine of the masked crop to the memory's identity prototype
  semantic    CLIP image-text cosine of the masked crop to the expression,
              relative to the same number at frame 0

Bands:  HIGH (c >= tau_high) stable | MEDIUM | LOW (c < tau_low) uncertain.
An absent target (empty / invisible mask) is NOT scored as drift: c = 0, band LOW,
but visible=False and state="absent", so V3/V4 can tell absence from drift.

Call order inside the frame loop:   tcs.score(...)  BEFORE  mem.update(...)
because TCS reads the memory's state from the PREVIOUS frame (predicted next
centroid, last area, prototype). score() never modifies the memory.
V2 action: optionally block memory writes on frames that are not HIGH
(mem.update(..., allow_write=False)). Nothing else changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sttm import mask_geometry

TERMS = ("mask", "motion", "appearance", "semantic")


@dataclass
class TCSParams:
    # Term weights. Fixed equal in V2 (not tuned: see V2_TUNING_PLAN.md).
    w_mask: float = 1.0
    w_motion: float = 1.0
    w_appearance: float = 1.0
    w_semantic: float = 1.0
    # Band thresholds. 0.50 / 0.70 are PLACEHOLDERS until tools/tcs_analysis.py
    # suggests values from the ablation tier; the tuned ones live in config_v2_final.yaml.
    tau_low: float = 0.50
    tau_high: float = 0.70
    # Motion residual (in object diameters) at which the motion term falls to 1/e.
    motion_scale: float = 0.5
    # Appearance cosine at/below this maps to 0 (0.0 = use the raw cosine).
    app_floor: float = 0.0
    # How the four terms are combined.
    #   "geometric"  (default) weighted geometric mean: ONE collapsed term pulls c down.
    #                Drift, identity swaps and teleports each break one signal while the
    #                others stay high, and an average hides exactly that (tools/test_tcs.py).
    #   "arithmetic" weighted mean, kept for the ablation.
    agg: str = "geometric"
    eps: float = 0.02          # per-term floor so log(0) cannot happen


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate a boolean mask by (dx, dy) pixels; pixels pushed off the edge are dropped (no wrap)."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    if abs(dx) >= w or abs(dy) >= h:
        return out
    ys_src = slice(max(0, -dy), min(h, h - dy))
    xs_src = slice(max(0, -dx), min(w, w - dx))
    ys_dst = slice(max(0, dy), min(h, h + dy))
    xs_dst = slice(max(0, dx), min(w, w + dx))
    out[ys_dst, xs_dst] = mask[ys_src, xs_src]
    return out


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def band_of(c: float, p: TCSParams) -> str:
    if c >= p.tau_high:
        return "HIGH"
    if c < p.tau_low:
        return "LOW"
    return "MEDIUM"


class TrackletCoherence:
    """One instance per tracked target (one per referring expression)."""

    def __init__(self, params: TCSParams):
        self.p = params
        self.prev_mask: np.ndarray | None = None   # last VISIBLE predicted mask
        self.sem0: float | None = None             # CLIP image-text cosine at frame 0
        self.counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "absent": 0}

    # ------------------------------------------------------------------ anchor
    def set_anchor(self, mask: np.ndarray, sem_score: float | None = None) -> None:
        self.prev_mask = mask.copy()
        self.sem0 = sem_score

    # ------------------------------------------------------------------- score
    def score(self, frame_idx: int, mem, mask: np.ndarray, obj_score: float, mask_conf: float,
              appearance: np.ndarray | None, sem_score: float | None = None) -> dict:
        """Coherence of this frame's prediction. `mem` is the TargetMemory BEFORE its update()."""
        p = self.p
        g = mask_geometry(mask)
        visible = (g is not None and obj_score > mem.p.tau_obj and g[1] >= mem.p.a_min)
        if not visible:
            self.counts["absent"] += 1
            self.counts["LOW"] += 1
            return {"frame": frame_idx, "visible": False, "state": "absent", "c": 0.0,
                    "band": "LOW", "terms": {t: None for t in TERMS}, "sim": None, "margin": None}

        box, area, cen = g
        state = "visible" if mem.visible_prev else "reappeared"
        terms: dict[str, float | None] = {t: None for t in TERMS}

        # ---- mask term: SAM 2 certainty (+ shape agreement with the previous visible mask)
        parts = [_sigmoid(obj_score), _clip01(mask_conf)]
        if state == "visible" and self.prev_mask is not None and mem.last_centroid is not None:
            dx = int(round(cen[0] - mem.last_centroid[0]))
            dy = int(round(cen[1] - mem.last_centroid[1]))
            parts.append(mask_iou(shift_mask(self.prev_mask, dx, dy), mask))
        terms["mask"] = float(np.mean(parts))

        # ---- motion term: centroid vs the memory's predicted next centroid
        if state == "visible" and mem.predicted_next is not None and mem.last_area:
            r = float(np.hypot(cen[0] - mem.predicted_next[0], cen[1] - mem.predicted_next[1]))
            r /= max(float(np.sqrt(mem.last_area)), 1.0)
            terms["motion"] = float(np.exp(-r / max(p.motion_scale, 1e-6)))

        # ---- appearance term (+ distractor margin, logged for V3 but not part of c_t)
        sim = margin = None
        if appearance is not None:
            sim = _cos(appearance, mem.prototype())
            fl = min(max(p.app_floor, 0.0), 0.99)
            terms["appearance"] = _clip01((sim - fl) / (1.0 - fl))
            if mem.distractor_embs.size:
                d = float(max(_cos(appearance, e) for e in mem.distractor_embs))
                margin = sim - d

        # ---- semantic term: CLIP crop-vs-expression, relative to frame 0
        if sem_score is not None and self.sem0 is not None and self.sem0 > 1e-6:
            terms["semantic"] = _clip01(sem_score / self.sem0)

        # ---- combine over the terms that exist this frame (weights renormalised)
        w = {"mask": p.w_mask, "motion": p.w_motion, "appearance": p.w_appearance, "semantic": p.w_semantic}
        have = [(w[t], v) for t, v in terms.items() if v is not None]
        den = sum(wt for wt, _ in have)
        if den <= 0:
            c = 0.0
        elif p.agg == "arithmetic":
            c = float(sum(wt * v for wt, v in have) / den)
        else:
            c = float(np.exp(sum(wt * np.log(max(v, p.eps)) for wt, v in have) / den))

        band = band_of(c, p)
        self.counts[band] += 1
        self.prev_mask = mask.copy()
        return {"frame": frame_idx, "visible": True, "state": state, "c": c, "band": band,
                "terms": terms, "sim": sim, "margin": margin}

    def summary(self) -> dict:
        n = sum(v for k, v in self.counts.items() if k != "absent")
        return {"counts": dict(self.counts), "frames": n}
