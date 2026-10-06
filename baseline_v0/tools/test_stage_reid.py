#!/usr/bin/env python3
"""CPU tests for stage_reid.py: when to search, who decides, and the restart driver (a fake SAM 2 stands in for the real one;
the real one is exercised by the safety test and the smoke run on your GPU)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reid as R
import stage_reid as S
from events import EventParams

H, W = 40, 80


def box_mask(x1, y1, x2, y2):
    m = np.zeros((H, W), bool)
    m[y1:y2, x1:x2] = True
    return m


# ------------------------------------------------------------------ SearchPlanner
def planner(guard=False, **kw):
    return S.SearchPlanner(EventParams(tau_low=0.5, tau_high=0.7, **kw), guard=guard)


def run_planner(pl, frames):
    """frames: list of (c, visible, state, mask_empty) for f = 1, 2, ... -> {frame: trigger}"""
    return {f: t for f, (c, v, st, me) in enumerate(frames, start=1)
            if (t := pl.step(f, c, v, st, None, me)) is not None}


def test_quiet_frames_never_search():
    assert run_planner(planner(), [(0.9, True, "visible", False)] * 30) == {}


def test_a_coherence_drop_searches_once_then_the_pause_holds():
    frames = [(0.9, True, "visible", False)] * 5 + [(0.2, True, "visible", False)] * 6 + [(0.9, True, "visible", False)] * 5
    out = run_planner(planner(), frames)
    assert out == {6: "coherence_drop"}                        # entering LOW fires once; the pause (10) hides what follows


def test_empty_mask_polls_every_ten_frames():
    frames = [(0.9, True, "visible", False)] * 3 + [(0.0, False, "absent", True)] * 35
    out = run_planner(planner(), frames)
    assert out[4] == "disappear"                                # the V3 event fires first
    polls = [f for f, t in out.items() if t == "absent_poll"]
    assert polls == [14, 24, 34]                                # then one every 10 frames while the mask stays empty


def test_poll_starts_when_there_was_no_event_at_all():
    out = run_planner(planner(), [(0.9, True, "visible", True)] * 25)           # empty mask, but the TCS never flagged an event
    assert out == {1: "absent_poll", 11: "absent_poll", 21: "absent_poll"}


def test_guards_cap_the_vlm_variant_only():
    frames = [(0.0, False, "absent", True)] * 200
    free = run_planner(planner(guard=False), frames)
    assert len(free) > 10
    pl = planner(guard=True)
    got = run_planner(pl, frames)
    assert len(got) == 5 and pl.skipped_guard_episode > 0                       # 5 per absent episode


def test_per_expression_guard_and_new_episode_resets_the_episode_counter():
    pl = planner(guard=True)
    frames = []
    for _ in range(4):                                                           # four absences, each long enough for 3 searches
        frames += [(0.9, True, "visible", False)] * 2 + [(0.0, False, "absent", True)] * 25
    got = run_planner(pl, frames)
    assert len(got) == S.GUARD_PER_EXPRESSION and pl.skipped_guard_expression > 0


# ------------------------------------------------------------------ decide_search
def cands_and_rank():
    cur = {"id": "current", "mask": box_mask(0, 0, 10, 10), "source": "current", "dino_score": None}
    a = {"id": "d0", "mask": box_mask(40, 10, 60, 30), "source": "dino", "dino_score": 0.9}
    b = {"id": "d1", "mask": box_mask(10, 20, 30, 40), "source": "dino", "dino_score": 0.8}
    cands = [cur, a, b]
    ranked = [{**a, "score": 0.8}, {**cur, "score": 0.5}, {**b, "score": 0.4}]
    return cands, ranked


def test_off_never_picks_and_oracle_memory_vlm_pick_as_specified():
    P = R.ReIDParams()
    cands, ranked = cands_and_rank()
    gt = box_mask(10, 20, 30, 40)                                                # the target is d1
    assert S.decide_search("off", cands, ranked, gt, P)["choice"] == "none"
    assert S.decide_search("oracle", cands, ranked, gt, P)["choice"] == "d1"
    assert S.decide_search("memory", cands, ranked, gt, P)["choice"] == "d0"      # best memory score, >= 0.6
    low = [{**c, "score": 0.3} for c in ranked]
    assert S.decide_search("memory", cands, low, gt, P)["choice"] == "none"
    seen = {}

    def pick(sl):
        seen["ids"] = [c["id"] for c in sl]
        return {"index": 1, "confidence": 0.9}

    d = S.decide_search("vlm", cands, ranked, gt, P, pick)
    assert seen["ids"] == ["d0", "current", "d1"] and d["choice"] == "current" and d["value"] == 0.9
    assert S.decide_search("vlm", cands, ranked, gt, P, lambda sl: {"index": None, "confidence": 0.2})["choice"] == "none"


def test_no_candidates_or_unknown_variant():
    P = R.ReIDParams()
    assert S.decide_search("vlm", [], [], None, P, lambda sl: 1 / 0)["choice"] == "none"      # never even asks
    try:
        S.decide_search("magic", [{"id": "d0"}], [{"id": "d0", "score": 1}], None, P)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ------------------------------------------------------------------ the restart driver with a fake SAM 2
class FakePredictor:
    """Mimics what V4 relies on in SAM 2: propagate_in_video replays outputs it already stored, computes the rest, and
    add_new_mask overwrites the stored output of that object at that frame. A computed frame moves each object's previous
    mask one pixel to the right."""

    def __init__(self, n_frames, n_obj=2):
        self.n, self.n_obj = n_frames, n_obj
        self.stored = {0: {i: box_mask(5 + 10 * i, 5, 15 + 10 * i, 15) for i in range(1, n_obj + 1)}}
        self.calls, self.added = [], []

    def propagate_in_video(self, state, start_frame_idx=None):
        self.calls.append(start_frame_idx)
        for f in range(0 if start_frame_idx is None else start_frame_idx, self.n):
            if f not in self.stored:
                prev = self.stored[f - 1]
                self.stored[f] = {i: np.roll(prev[i], 1, axis=1) for i in prev}
            yield f, list(range(1, self.n_obj + 1)), [self.stored[f][i] for i in range(1, self.n_obj + 1)]

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        self.added.append((frame_idx, obj_id))
        self.stored[frame_idx][obj_id] = np.asarray(mask, bool)


def test_driver_without_restarts_visits_every_frame_once():
    fp, seen = FakePredictor(12), []
    n = S.drive(fp, {}, lambda f, ids, lg: seen.append(f) or None)
    assert n == 0 and seen == list(range(12)) and fp.calls == [None] and fp.added == []


def test_restart_replaces_the_object_from_that_frame_on_and_never_touches_the_past():
    fp = FakePredictor(14)
    new = box_mask(50, 20, 60, 30)
    log = {}                                                                     # (frame, obj) -> mask the tracker reported

    def on_frame(f, ids, lg):
        for i, m in zip(ids, lg):
            log[(f, i)] = m.copy()
        return {1: new} if f == 6 and (6, 1) not in log_restarted else None

    log_restarted = set()
    orig = on_frame

    def on_frame2(f, ids, lg):
        r = orig(f, ids, lg)
        if r:
            log_restarted.add((f, 1))
        return r

    n = S.drive(fp, {}, on_frame2)
    assert n == 1 and fp.calls == [None, 6] and fp.added == [(6, 1)]
    frames_seen = sorted({f for f, _ in log})
    assert frames_seen == list(range(14))
    # before the restart: obj 1 followed its original track, one pixel per frame
    for f in range(0, 7):
        assert (log[(f, 1)] == np.roll(box_mask(15, 5, 25, 15), f, axis=1)).all(), f
    # after it: the new mask, moving on from frame 6 (frame 7 = new shifted by 1)
    for f in range(7, 14):
        assert (log[(f, 1)] == np.roll(new, f - 6, axis=1)).all(), f
    # the other object never noticed
    for f in range(14):
        assert (log[(f, 2)] == np.roll(box_mask(25, 5, 35, 15), f, axis=1)).all(), f


def test_two_restarts_and_restarting_every_frame_terminates():
    fp = FakePredictor(10)
    n = S.drive(fp, {}, lambda f, ids, lg: {1: box_mask(0, 0, 8, 8)} if f >= 1 else None)
    assert n == 9 and fp.calls == [None] + list(range(1, 10))                    # one restart per frame 1..9, each from its own frame
    fp2, seen = FakePredictor(10), []

    def on_frame(f, ids, lg):
        seen.append(f)
        return {2: box_mask(1, 1, 9, 9)} if f in (3, 7) else None

    assert S.drive(fp2, {}, on_frame) == 2 and seen == list(range(10)) and fp2.calls == [None, 3, 7]


def test_two_objects_restarted_on_the_same_frame_count_two_and_resume_once():
    fp = FakePredictor(8)
    n = S.drive(fp, {}, lambda f, ids, lg: {1: box_mask(0, 0, 5, 5), 2: box_mask(9, 9, 14, 14)} if f == 4 else None)
    assert n == 2 and fp.calls == [None, 4] and fp.added == [(4, 1), (4, 2)]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok   {t.__name__}")
    print(f"all {len(tests)} tests passed")
