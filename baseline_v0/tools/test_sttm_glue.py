"""Tests for sttm_sam2.sync_bank on fake SAM2 state. Run: python tools/test_sttm_glue.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sttm_sam2 import object_score, sync_bank


def fake_state(n_frames=20, n_obj=2):
    st = {"output_dict_per_obj": {}}
    for o in range(n_obj):
        st["output_dict_per_obj"][o] = {
            "cond_frame_outputs": {0: {"tag": (o, 0)}},
            "non_cond_frame_outputs": {t: {"tag": (o, t)} for t in range(1, n_frames)},
        }
    return st


def test_promote_then_demote():
    st = fake_state()
    r = sync_bank(st, 0, [0, 7, 12])
    assert r["promoted"] == [7, 12] and r["demoted"] == []
    od = st["output_dict_per_obj"][0]
    assert sorted(od["cond_frame_outputs"]) == [0, 7, 12]
    assert 7 not in od["non_cond_frame_outputs"]       # never attended twice
    r = sync_bank(st, 0, [0, 12, 15])
    assert r["demoted"] == [7] and r["promoted"] == [15]
    assert 7 in od["non_cond_frame_outputs"] and 7 not in od["cond_frame_outputs"]


def test_anchor_never_demoted():
    st = fake_state()
    sync_bank(st, 0, [0, 5])
    sync_bank(st, 0, [])          # even an empty request keeps the prompt frame
    assert 0 in st["output_dict_per_obj"][0]["cond_frame_outputs"]


def test_objects_are_independent():
    st = fake_state()
    sync_bank(st, 0, [0, 9])
    assert sorted(st["output_dict_per_obj"][1]["cond_frame_outputs"]) == [0]


def test_missing_frame_reported_not_crashing():
    st = fake_state(n_frames=5)
    r = sync_bank(st, 0, [0, 99])
    assert r["missing"] == [99]


def test_outputs_are_the_same_objects_not_copies():
    st = fake_state()
    orig = st["output_dict_per_obj"][0]["non_cond_frame_outputs"][7]
    sync_bank(st, 0, [0, 7])
    assert st["output_dict_per_obj"][0]["cond_frame_outputs"][7] is orig


def test_object_score_reads_arrays_and_floats():
    assert object_score({"object_score_logits": np.array([[3.5]])}) == 3.5
    assert object_score({"object_score_logits": -1.0}) == -1.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\nall {len(tests)} tests passed")
