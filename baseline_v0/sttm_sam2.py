"""
Glue between STTM (sttm.py) and SAM2's video predictor state.  READ ACTION A.

SAM2 keeps, per object, two dicts of stored frame outputs:
    state["output_dict_per_obj"][obj_idx]["cond_frame_outputs"]      always attended
    state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"]  only the last 6 frames
`propagate_in_video` is a generator that re-reads these dicts every frame, so
between two yielded frames we can PROMOTE a curated frame (non_cond -> cond) so
SAM2 keeps attending to it, and DEMOTE an evicted one (cond -> non_cond).
No SAM2 source is patched. Frame 0 (the prompt) is never demoted.

Pure dict operations: testable without SAM2 or a GPU.
"""
from __future__ import annotations


def sync_bank(state: dict, obj_idx: int, wanted_frames: list, anchor_frame: int = 0) -> dict:
    """Make SAM2's conditioning set for `obj_idx` equal to `wanted_frames`
    (memory anchor + curated entries). Returns what changed."""
    od = state["output_dict_per_obj"][obj_idx]
    cond, non = od["cond_frame_outputs"], od["non_cond_frame_outputs"]
    want = set(wanted_frames) | {anchor_frame}
    promoted, demoted, missing = [], [], []
    for t in list(cond):
        if t not in want:
            non[t] = cond.pop(t)
            demoted.append(t)
    for t in sorted(want):
        if t in cond:
            continue
        if t in non:
            cond[t] = non.pop(t)
            promoted.append(t)
        else:
            missing.append(t)      # entry refers to a frame SAM2 has no stored output for
    return {"promoted": promoted, "demoted": demoted, "missing": missing}


def object_score(out: dict) -> float:
    """SAM2's object-score logit for a stored frame output (>0 means 'object present')."""
    v = out["object_score_logits"]
    try:
        return float(v.reshape(-1)[0])
    except AttributeError:
        return float(v)
