"""
Memory-grounded re-identification (V4), pure logic: no GPU, no SAM 2, no VLM, no files. Unit-tested on CPU.

The live loop (stage_reid.py, next step) does the model work (SAM 2, DINOv2, the VLM). This module decides:
  1. which candidates are on the table at a search frame          build_candidates()
  2. how well each one fits what the target memory knows          rank()
  3. who picks one of them                                        choose_memory() / choose_oracle() (the VLM chooser is in vlm.py)
  4. what that pick means for the tracker                         decide()
  5. how an attempt is graded afterwards                          label_attempt()

All numbers below are the ones fixed in V4_PLAN.md before any V4 run. None of them is tuned.

Candidate: {"id": "current" | "d0" | "d1" ..., "mask": bool (H, W), "source": "current" | "dino", "dino_score": float | None}
"current" is the tracker's own mask on the search frame (only if non-empty). It is always kept, and a Grounding-DINO
candidate that overlaps it with IoU above `dup_iou` is dropped as a duplicate of it.

Memory state (a plain dict, built from sttm.TargetMemory by memory_state()):
  {"proto": L2-normalised DINOv2 prototype, "centroid": (x, y) of the last visible mask, "area": its area in pixels,
   "hw": (H, W)}
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CURRENT = "current"
NONE = "none"
LABELS = ("correct_restart", "wrong_restart", "weak_restart", "missed", "correct_none", "candidate_miss")


@dataclass
class ReIDParams:
    shortlist: int = 3          # candidates shown to the chooser
    dup_iou: float = 0.7        # DINO candidate overlapping another by more than this is a duplicate
    min_area: int = 200         # same as STTM a_min: smaller masks are not candidates
    score_min: float = 0.6      # memory-only chooser: best score must reach this, else "none"
    oracle_iou: float = 0.5     # oracle chooser and "good candidate" in the grading
    wrong_iou: float = 0.10     # restart on a candidate below this = wrong restart
    switch_margin: float = 0.10  # POST-HOC *_margin variants only: restart only if the pick beats the tracker's current mask by this much (memory score)


# ------------------------------------------------------------------ geometry
def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def _geom(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return (float(xs.mean()), float(ys.mean())), int(len(ys))


# ------------------------------------------------------------------ 1. candidates
def build_candidates(current_mask, dino_cands, p: ReIDParams):
    """current_mask: the tracker's mask on this frame (bool array) or None/empty. dino_cands: list of
    {"mask", "score", ...} from CandidateStore.get(f), best DINO score first.
    -> list of candidates, "current" first (if usable), then DINO candidates in score order, duplicates removed."""
    out = []
    if current_mask is not None and current_mask.any() and int(current_mask.sum()) >= p.min_area:
        out.append({"id": CURRENT, "mask": current_mask.astype(bool), "source": "current", "dino_score": None})
    for i, c in enumerate(dino_cands):
        m = np.asarray(c["mask"], bool)
        if int(m.sum()) < p.min_area:
            continue
        if any(mask_iou(m, o["mask"]) > p.dup_iou for o in out):
            continue
        out.append({"id": f"d{i}", "mask": m, "source": "dino", "dino_score": float(c["score"])})
    return out


# ------------------------------------------------------------------ 2. ranking with memory
def appearance_term(cos: float) -> float:
    """DINOv2 cosine in [-1, 1] -> [0, 1]."""
    return float(min(1.0, max(0.0, (cos + 1.0) / 2.0)))


def position_term(centroid, last_centroid, hw) -> float:
    """1 at the last known position, 0 one image diagonal away."""
    diag = float(np.hypot(hw[0], hw[1]))
    d = float(np.hypot(centroid[0] - last_centroid[0], centroid[1] - last_centroid[1]))
    return float(min(1.0, max(0.0, 1.0 - d / diag)))


def size_term(area: int, last_area: int) -> float:
    """Smaller area over larger area."""
    if area <= 0 or last_area <= 0:
        return 0.0
    return float(min(area, last_area) / max(area, last_area))


def memory_score(app: float, pos: float, size: float) -> float:
    """Geometric mean of the three terms, equal weights (same style as the V2 coherence score)."""
    return float((max(app, 0.0) * max(pos, 0.0) * max(size, 0.0)) ** (1.0 / 3.0))


def rank(cands, embeds: dict, state: dict, p: ReIDParams):
    """cands from build_candidates(); embeds: {candidate id: L2-normalised DINOv2 vector or None}.
    -> the candidates sorted by memory score, best first, each with "app", "pos", "size", "score" added.
    A candidate without an embedding is scored with an appearance term of 0 (it cannot rank well). Ties keep input order."""
    scored = []
    for order, c in enumerate(cands):
        g = _geom(c["mask"])
        e = embeds.get(c["id"])
        cos = float(np.dot(e, state["proto"])) if e is not None else -1.0
        app = appearance_term(cos)
        pos = position_term(g[0], state["centroid"], state["hw"]) if g else 0.0
        siz = size_term(g[1], state["area"]) if g else 0.0
        scored.append(({**c, "app": app, "pos": pos, "size": siz, "score": memory_score(app, pos, siz)}, order))
    scored.sort(key=lambda t: (-t[0]["score"], t[1]))
    return [c for c, _ in scored]


def shortlist(ranked, p: ReIDParams):
    return ranked[:p.shortlist]


def memory_state(mem, hw):
    """Build the plain state dict from an sttm.TargetMemory (uses its prototype and last visible geometry)."""
    return {"proto": mem.prototype(), "centroid": mem.last_centroid, "area": mem.last_area, "hw": tuple(hw)}


# ------------------------------------------------------------------ 3. choosers
def choose_memory(ranked, p: ReIDParams):
    """V4-memory: the top-ranked candidate if its score reaches score_min, else "none". -> (choice id, its score or None)"""
    if ranked and ranked[0]["score"] >= p.score_min:
        return ranked[0]["id"], ranked[0]["score"]
    return NONE, (ranked[0]["score"] if ranked else None)


def choose_oracle(cands, gt_mask, p: ReIDParams):
    """V4-oracle (a CEILING that uses ground truth, never a result): the candidate with the highest IoU to the true
    target if that IoU reaches oracle_iou, else "none". -> (choice id, its IoU)."""
    best, best_iou = NONE, 0.0
    if gt_mask is not None and gt_mask.any():
        for c in cands:
            v = mask_iou(c["mask"], gt_mask)
            if v > best_iou:
                best, best_iou = c["id"], v
    return (best, best_iou) if best_iou >= p.oracle_iou else (NONE, best_iou)


# ------------------------------------------------------------------ 4. what a pick means
def decide(choice: str, cands):
    """-> ("restart", candidate) when the pick is a candidate other than "current"; ("keep", None) for "current", "none"
    or an id that is not on the table. V4 never blanks a mask."""
    if choice in (NONE, CURRENT):
        return "keep", None
    for c in cands:
        if c["id"] == choice:
            return "restart", c
    return "keep", None


# ------------------------------------------------------------------ 5. grading one search attempt
def label_attempt(cands, choice: str, gt_mask, p: ReIDParams):
    """Exactly one of LABELS per search attempt, graded against the true target mask on the search frame.
      correct_restart  restarted on a candidate with IoU >= oracle_iou
      wrong_restart    restarted on a candidate with IoU < wrong_iou (including when the target is not visible)
      weak_restart     restarted on a candidate in between
      correct_none     nothing changed and nothing needed changing: the target is not visible, or the tracker's current
                       mask is already good (IoU >= oracle_iou)
      missed           nothing changed, the current mask is not good, but another candidate was good
      candidate_miss   nothing changed, the target is visible and no candidate was good
    -> (label, IoU of the chosen candidate or None)"""
    visible = gt_mask is not None and bool(gt_mask.any())
    action, cand = decide(choice, cands)
    if action == "restart":
        v = mask_iou(cand["mask"], gt_mask) if visible else 0.0
        if v >= p.oracle_iou:
            return "correct_restart", v
        return ("wrong_restart" if v < p.wrong_iou else "weak_restart"), v
    if not visible:
        return "correct_none", None
    ious = {c["id"]: mask_iou(c["mask"], gt_mask) for c in cands}
    if ious.get(CURRENT, 0.0) >= p.oracle_iou:
        return "correct_none", None
    if any(v >= p.oracle_iou for v in ious.values()):
        return "missed", None
    return "candidate_miss", None


# ------------------------------------------------------------------ 6. POST-HOC margin gate (V4 memory_margin / vlm_margin)
def margin_gate(choice, ranked, margin):
    """Keep the tracker unless the pick clearly beats its current mask. No current candidate (empty or tiny mask) -> restart is allowed."""
    if choice in (NONE, CURRENT):
        return choice
    cur = next((c for c in ranked if c["id"] == CURRENT), None)
    if cur is None:
        return choice
    pick = next((c for c in ranked if c["id"] == choice), None)
    if pick is None:
        return NONE
    return choice if pick["score"] - cur["score"] >= margin else CURRENT
