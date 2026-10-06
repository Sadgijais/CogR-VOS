#!/usr/bin/env python3
"""CPU tests for reid.py (pure logic: no GPU, no SAM 2, no VLM, no files)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reid as R

P = R.ReIDParams()
H, W = 100, 200


def box_mask(x1, y1, x2, y2):
    m = np.zeros((H, W), bool)
    m[y1:y2, x1:x2] = True
    return m


def vec(*v):
    a = np.asarray(v, np.float32)
    return a / np.linalg.norm(a)


STATE = {"proto": vec(1, 0, 0), "centroid": (50.0, 50.0), "area": 20 * 20, "hw": (H, W)}
TARGET = box_mask(40, 40, 60, 60)             # centroid (49.5, 49.5), area 400


def test_current_first_and_tiny_masks_dropped():
    cur = box_mask(40, 40, 60, 60)
    dino = [{"mask": box_mask(100, 10, 120, 30), "score": 0.9}, {"mask": box_mask(0, 0, 5, 5), "score": 0.8}]   # 25 px: too small
    c = R.build_candidates(cur, dino, P)
    assert [x["id"] for x in c] == ["current", "d0"]


def test_duplicate_of_current_is_dropped_and_ids_keep_the_dino_index():
    cur = box_mask(40, 40, 60, 60)
    dino = [{"mask": box_mask(41, 40, 61, 60), "score": 0.9},          # IoU with current = 19/21 = 0.90 > 0.7 -> duplicate
            {"mask": box_mask(100, 10, 120, 30), "score": 0.8}]
    c = R.build_candidates(cur, dino, P)
    assert [x["id"] for x in c] == ["current", "d1"]


def test_two_overlapping_dino_candidates_keep_the_higher_scored_one():
    dino = [{"mask": box_mask(100, 10, 120, 30), "score": 0.9}, {"mask": box_mask(101, 10, 121, 30), "score": 0.7}]
    c = R.build_candidates(None, dino, P)
    assert [x["id"] for x in c] == ["d0"]


def test_empty_current_mask_is_not_a_candidate():
    c = R.build_candidates(np.zeros((H, W), bool), [{"mask": box_mask(10, 10, 40, 40), "score": 0.5}], P)
    assert [x["id"] for x in c] == ["d0"]


def test_term_functions():
    assert R.appearance_term(1.0) == 1.0 and R.appearance_term(-1.0) == 0.0 and abs(R.appearance_term(0.0) - 0.5) < 1e-9
    diag = float(np.hypot(H, W))
    assert R.position_term((50, 50), (50, 50), (H, W)) == 1.0
    assert abs(R.position_term((50 + diag / 2, 50), (50, 50), (H, W)) - 0.5) < 1e-6
    assert R.position_term((10_000, 50), (50, 50), (H, W)) == 0.0
    assert R.size_term(100, 400) == 0.25 and R.size_term(400, 100) == 0.25 and R.size_term(0, 400) == 0.0
    assert abs(R.memory_score(1, 1, 1) - 1.0) < 1e-9
    assert R.memory_score(1, 0, 1) == 0.0                       # one zero term kills the score (geometric mean)


def test_rank_prefers_the_candidate_that_matches_appearance_position_and_size():
    good = box_mask(42, 42, 62, 62)                              # near, same size
    far = box_mask(150, 10, 170, 30)                             # same size, far away
    small = box_mask(45, 45, 60, 60)                             # near, but 225 px vs 400
    cands = [{"id": "d0", "mask": far}, {"id": "d1", "mask": small}, {"id": "d2", "mask": good}]
    emb = {c["id"]: vec(1, 0, 0) for c in cands}                 # identical appearance: geometry decides
    r = R.rank(cands, emb, STATE, P)
    assert [c["id"] for c in r][0] == "d2"
    assert r[0]["score"] > r[1]["score"]
    # now appearance decides: the near one looks like someone else
    emb2 = {"d0": vec(1, 0, 0), "d1": vec(1, 0, 0), "d2": vec(-1, 0.1, 0)}
    assert R.rank(cands, emb2, STATE, P)[0]["id"] != "d2"


def test_rank_without_embedding_scores_appearance_zero_and_ties_keep_input_order():
    a = {"id": "d0", "mask": box_mask(42, 42, 62, 62)}
    b = {"id": "d1", "mask": box_mask(42, 42, 62, 62)}
    r = R.rank([a, b], {"d0": None, "d1": None}, STATE, P)
    assert [c["id"] for c in r] == ["d0", "d1"] and r[0]["app"] == 0.0 and r[0]["score"] == 0.0


def test_shortlist_is_at_most_three():
    ranked = [{"id": f"d{i}", "score": 1 - i * 0.1} for i in range(5)]
    assert [c["id"] for c in R.shortlist(ranked, P)] == ["d0", "d1", "d2"]


def test_memory_chooser_needs_the_minimum_score():
    assert R.choose_memory([{"id": "d0", "score": 0.7}, {"id": "d1", "score": 0.5}], P) == ("d0", 0.7)
    assert R.choose_memory([{"id": "d0", "score": 0.59}], P) == (R.NONE, 0.59)
    assert R.choose_memory([], P) == (R.NONE, None)


def test_oracle_picks_the_best_overlap_and_needs_half():
    c = [{"id": "current", "mask": box_mask(0, 0, 30, 30)}, {"id": "d0", "mask": box_mask(40, 40, 60, 60)},
         {"id": "d1", "mask": box_mask(45, 40, 65, 60)}]
    assert R.choose_oracle(c, TARGET, P)[0] == "d0"
    assert R.choose_oracle(c, box_mask(120, 60, 140, 80), P)[0] == R.NONE        # nothing overlaps
    assert R.choose_oracle(c, np.zeros((H, W), bool), P) == (R.NONE, 0.0)       # target not visible


def test_decide_never_blanks_and_keeps_on_current_none_or_unknown():
    c = [{"id": "current", "mask": TARGET}, {"id": "d0", "mask": box_mask(100, 10, 120, 30)}]
    assert R.decide("d0", c)[0] == "restart" and R.decide("d0", c)[1]["id"] == "d0"
    assert R.decide("current", c) == ("keep", None)
    assert R.decide("none", c) == ("keep", None)
    assert R.decide("zzz", c) == ("keep", None)


def test_every_label_is_reachable_and_exactly_one_is_returned():
    cur_bad = {"id": "current", "mask": box_mask(120, 60, 140, 80)}
    good = {"id": "d0", "mask": box_mask(40, 40, 60, 60)}
    weak = {"id": "d1", "mask": box_mask(50, 40, 70, 60)}                       # IoU with TARGET = 1/3
    off = {"id": "d2", "mask": box_mask(150, 10, 170, 30)}
    absent = np.zeros((H, W), bool)
    assert R.label_attempt([cur_bad, good], "d0", TARGET, P)[0] == "correct_restart"
    assert R.label_attempt([cur_bad, off], "d2", TARGET, P)[0] == "wrong_restart"
    assert R.label_attempt([cur_bad, weak], "d1", TARGET, P)[0] == "weak_restart"
    assert R.label_attempt([cur_bad, off], "d2", absent, P)[0] == "wrong_restart"           # restarting when it is gone
    assert R.label_attempt([cur_bad, good], "none", TARGET, P)[0] == "missed"
    assert R.label_attempt([cur_bad, good], "current", TARGET, P)[0] == "missed"
    assert R.label_attempt([cur_bad, off], "none", TARGET, P)[0] == "candidate_miss"
    assert R.label_attempt([cur_bad, off], "none", absent, P)[0] == "correct_none"
    cur_good = {"id": "current", "mask": TARGET}
    assert R.label_attempt([cur_good, off], "none", TARGET, P)[0] == "correct_none"
    assert R.label_attempt([cur_good, off], "current", TARGET, P)[0] == "correct_none"
    for lab in ("correct_restart", "wrong_restart", "weak_restart", "missed", "correct_none", "candidate_miss"):
        assert lab in R.LABELS


def test_memory_state_reads_prototype_and_last_visible_geometry():
    class Mem:
        last_centroid, last_area = (10.0, 20.0), 300

        def prototype(self):
            return vec(0, 1, 0)

    s = R.memory_state(Mem(), (H, W))
    assert s["centroid"] == (10.0, 20.0) and s["area"] == 300 and s["hw"] == (H, W) and np.allclose(s["proto"], [0, 1, 0])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok   {t.__name__}")
    print(f"all {len(tests)} tests passed")
