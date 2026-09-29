"""
Semantic-Temporal Target Memory (STTM) — V1.

Pure numpy, no GPU, no SAM2 import: the memory logic is testable on fake data.
Design: CogR-VOS_V1_STTM_Design.md.

Six fields
  semantic     CLIP text embedding of the expression (cached, never changes)
  appearance   DINOv2 embedding of the masked target crop + running prototype
  motion       centroid, velocity (EMA), predicted next centroid
  spatial      box, area, border-touch flag
  context      similarity margin vs the cached distractor candidates +
               a visible / absent / reappeared event log
  reliability  r in [0,1] per stored entry

Bank = ONE permanent anchor (frame 0) + at most K-1 curated entries.
V1 uses reliability only to GATE WRITES. Nothing here decides "the track is
wrong" (that is V2+). Nothing here does re-ID (V4).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class STTMParams:
    K: int = 4                    # anchor + (K-1) curated entries
    tau_obj: float = 0.0          # SAM2 object-score logit must be > this to count as visible
    a_min: int = 200              # minimum mask area in pixels to count as visible
    tau_sim: float = 0.5          # min cosine to the identity prototype to be written
    delta: int = 5                # refractory: min frames between writes
    merge_thr: float = 0.95       # cosine >= this = near-duplicate -> merge, do not add
    half_life: float = 30.0       # freshness = 0.5 ** (age / half_life)
    anchor_weight: float = 0.5    # weight of the anchor in the identity prototype
    vel_ema: float = 0.7          # EMA factor for velocity
    border_px: int = 2            # mask within this many px of the frame edge = border-touching
    shrink_ratio: float = 0.8     # truncated = touches border AND area < ratio * last visible area


@dataclass
class Entry:
    frame_idx: int
    box: tuple                    # (x1, y1, x2, y2)
    area: int
    centroid: tuple               # (cx, cy)
    appearance: np.ndarray        # L2-normalised
    reliability: float
    t_written: int                # frame index at which it was written
    is_anchor: bool = False


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def mask_geometry(mask: np.ndarray):
    """(box, area, centroid) of a boolean mask, or None if empty."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return box, int(len(ys)), (float(xs.mean()), float(ys.mean()))


class TargetMemory:
    def __init__(self, params: STTMParams, frame_hw: tuple, text_emb: np.ndarray,
                 distractor_embs: np.ndarray | None = None):
        self.p = params
        self.h, self.w = frame_hw
        # ---- semantic field
        self.semantic = text_emb
        # ---- context field (cached candidates EXCLUDING the target's own box)
        self.distractor_embs = (distractor_embs if distractor_embs is not None
                                else np.zeros((0, 0), dtype=np.float32))
        self.events: list[dict] = []          # visible / absent / reappeared log
        self.context: dict = {"max_distractor_sim": None, "margin": None}
        # ---- memory bank
        self.anchor: Entry | None = None
        self.curated: list[Entry] = []
        # ---- motion / spatial live state
        self.velocity = (0.0, 0.0)
        self.last_centroid = None
        self.last_box = None
        self.last_area = None
        self.border_touch = False
        self.predicted_next = None
        # ---- tracking state
        self.visible_prev = True
        self.last_write_frame = -10**9
        self.absent_since = None
        # ---- counters (reported per video)
        self.stats = {"writes": 0, "merges": 0, "evictions": 0, "frames": 0,
                      "blocked": {"not_visible": 0, "truncated": 0, "low_similarity": 0,
                                  "paced": 0, "not_better": 0}}

    # ------------------------------------------------------------------ anchor
    def set_anchor(self, frame_idx: int, mask: np.ndarray, appearance: np.ndarray,
                   obj_score: float = 10.0, pred_iou: float = 1.0) -> None:
        g = mask_geometry(mask)
        if g is None:
            raise ValueError("anchor mask is empty — cannot start memory")
        box, area, cen = g
        self.anchor = Entry(frame_idx, box, area, cen, appearance, 1.0, frame_idx, True)
        self.last_write_frame = frame_idx
        self.last_centroid, self.last_box, self.last_area = cen, box, area
        self.border_touch = self._touches_border(box)
        self.predicted_next = cen
        self.events.append({"frame": frame_idx, "event": "visible"})

    # --------------------------------------------------------------- prototype
    def prototype(self) -> np.ndarray:
        a = self.anchor.appearance
        if not self.curated:
            return a
        c = np.mean([e.appearance for e in self.curated], axis=0)
        v = self.p.anchor_weight * a + (1 - self.p.anchor_weight) * c
        n = np.linalg.norm(v)
        return v / n if n > 0 else a

    # ------------------------------------------------------------------ helpers
    def _touches_border(self, box) -> bool:
        x1, y1, x2, y2 = box
        b = self.p.border_px
        return x1 <= b or y1 <= b or x2 >= self.w - b or y2 >= self.h - b

    def _freshness(self, e: Entry, now: int) -> float:
        return 0.5 ** (max(0, now - e.t_written) / self.p.half_life)

    def entries(self) -> list[Entry]:
        return ([self.anchor] if self.anchor else []) + list(self.curated)

    def read_frames(self) -> list[int]:
        """Frame indices the READ ACTION should condition on: anchor + curated."""
        return [e.frame_idx for e in self.entries()]

    # ------------------------------------------------------------------- update
    def update(self, frame_idx: int, mask: np.ndarray, appearance: np.ndarray | None,
               obj_score: float, pred_iou: float) -> dict:
        """Observe one predicted frame. Returns a decision dict for logging.
        `appearance` may be None when the mask is empty (nothing to embed)."""
        assert self.anchor is not None, "call set_anchor first"
        self.stats["frames"] += 1
        g = mask_geometry(mask)
        visible = (g is not None and obj_score > self.p.tau_obj and g[1] >= self.p.a_min)

        # ---- context: visible/absent/reappeared events
        if visible and not self.visible_prev:
            self.events.append({"frame": frame_idx, "event": "reappeared",
                                "absent_frames": frame_idx - (self.absent_since or frame_idx)})
        elif not visible and self.visible_prev:
            self.events.append({"frame": frame_idx, "event": "absent"})
            self.absent_since = frame_idx
        self.visible_prev = visible

        # ---- retire rule 4: FREEZE while absent (no writes, no evictions)
        if not visible:
            self.stats["blocked"]["not_visible"] += 1
            return {"written": False, "blocked_by": "not_visible", "visible": False}

        box, area, cen = g
        # ---- motion + spatial fields
        if self.last_centroid is not None:
            vx, vy = cen[0] - self.last_centroid[0], cen[1] - self.last_centroid[1]
            a = self.p.vel_ema
            self.velocity = (a * self.velocity[0] + (1 - a) * vx, a * self.velocity[1] + (1 - a) * vy)
        prev_area = self.last_area
        touches = self._touches_border(box)
        self.predicted_next = (cen[0] + self.velocity[0], cen[1] + self.velocity[1])
        self.last_centroid, self.last_box, self.last_area, self.border_touch = cen, box, area, touches

        # ---- context: margin vs distractors
        sim = _cos(appearance, self.prototype())
        if self.distractor_embs.size:
            d = float(max(_cos(appearance, e) for e in self.distractor_embs))
            self.context = {"max_distractor_sim": d, "margin": sim - d}

        # ---- write gate
        truncated = touches and prev_area is not None and area < self.p.shrink_ratio * prev_area
        if truncated:
            return self._block("truncated", frame_idx, sim)
        if sim < self.p.tau_sim:
            return self._block("low_similarity", frame_idx, sim)
        if frame_idx - self.last_write_frame < self.p.delta:
            return self._block("paced", frame_idx, sim)

        r = float(np.mean([_sigmoid(obj_score), float(np.clip(pred_iou, 0, 1)),
                           float(np.clip(sim, 0, 1))]))
        new = Entry(frame_idx, box, area, cen, appearance, r, frame_idx)
        return self._write(new, frame_idx)

    def _block(self, reason: str, frame_idx: int, sim: float) -> dict:
        self.stats["blocked"][reason] += 1
        return {"written": False, "blocked_by": reason, "visible": True, "sim": sim}

    # ---------------------------------------------------------- write + retire
    def _write(self, new: Entry, now: int) -> dict:
        cap = self.p.K - 1
        # retire rule 2: merge near-duplicates (never touches the anchor)
        for i, e in enumerate(self.curated):
            if _cos(new.appearance, e.appearance) >= self.p.merge_thr:
                keep_r = max(e.reliability, new.reliability)
                new.reliability = keep_r
                self.curated[i] = new
                self.stats["merges"] += 1
                self.last_write_frame = now
                return {"written": True, "action": "merged", "visible": True}
        if len(self.curated) < cap:
            self.curated.append(new)
            self.stats["writes"] += 1
            self.last_write_frame = now
            return {"written": True, "action": "added", "visible": True}
        # retire rule 3: evict lowest reliability x freshness (anchor is not in `curated`)
        scores = [e.reliability * self._freshness(e, now) for e in self.curated]
        j = int(np.argmin(scores))
        if new.reliability <= scores[j]:
            self.stats["blocked"]["not_better"] += 1
            return {"written": False, "blocked_by": "not_better", "visible": True}
        self.curated[j] = new
        self.stats["writes"] += 1
        self.stats["evictions"] += 1
        self.last_write_frame = now
        return {"written": True, "action": "evicted", "visible": True}

    # ----------------------------------------------------------------- summary
    def summary(self) -> dict:
        return {
            "bank_frames": self.read_frames(),
            "reliabilities": [round(e.reliability, 3) for e in self.entries()],
            "n_events": len(self.events),
            "events": self.events,
            "context": self.context,
            "velocity": self.velocity,
            "stats": self.stats,
        }
