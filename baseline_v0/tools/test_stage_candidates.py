#!/usr/bin/env python3
"""CPU tests for stage_candidates.py (fake Grounding-DINO and fake SAM 2; no GPU, no dataset, no network)."""
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stage_candidates as C

H, W = 7, 11          # deliberately not multiples of 8, so the bit packing has padding to get wrong


def fake_ground(path, text):
    """2 boxes on every frame, a different one per frame number; frame 3 has none."""
    f = int(Path(path).stem)
    if f == 3:
        return []
    return [[1 + f, 1, 6, 5, 0.9], [0, 0, 3, 3, 0.4]]


class FakeSeg:
    def __init__(self):
        self.frames_set = []

    def set_frame(self, path):
        self.frames_set.append(Path(path).stem)

    def mask(self, box):
        m = np.zeros((H, W), bool)
        x1, y1, x2, y2 = [int(v) for v in box]
        m[y1:y2, x1:x2] = True
        return m


def paths(n):
    return [Path(f"{i:05d}.jpg") for i in range(n)]


def test_pack_roundtrip_with_odd_sizes():
    rng = np.random.default_rng(0)
    m = rng.random((3, 5, H, W)) > 0.5
    assert (C.unpack_masks(C.pack_masks(m), (H, W)) == m).all()


def write_events(path, rows):
    cols = ["video", "exp_id", "expression", "frames", "candidates", "calls", "suppressed", *C.EVENT_COLS,
            "keepalive", "periodic", "blanked_frames", "false_absence_flags"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for v, e, t, ev in rows:
            r = {c: 0 for c in cols}
            r.update(video=v, exp_id=e, expression=t, frames=10)
            r.update(ev)
            w.writerow(r)


def test_select_expressions_keeps_only_those_with_a_trigger_event():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "e.csv"
        write_events(p, [("a", "0_1", "red car", {"disappear": 1}), ("a", "0_2", "man", {}),
                         ("b", "0_1", "dog", {"keepalive": 3, "periodic": 2}),     # not triggers
                         ("b", "0_2", "cat", {"coherence_drop": 2, "reappear": 1})])
        assert C.select_expressions(p) == [("a", "0_1", "red car"), ("b", "0_2", "cat")]
        assert C.select_expressions(p, {"a"}) == [("a", "0_1", "red car")]


def test_end_to_end_store_roundtrip_and_lookup():
    frames = [1, 2, 3, 4]
    fps = paths(5)
    exprs = {"0_1": "a man", "0_2": "a dog"}
    boxes = C.boxes_pass(fps, exprs, frames, fake_ground)
    assert boxes["0_1"]["3"] == [] and len(boxes["0_1"]["2"]) == 2
    seg = FakeSeg()
    masks = C.masks_pass(fps, boxes, frames, seg)
    assert seg.frames_set == ["00001", "00002", "00003", "00004"]      # one set_frame per frame, shared by expressions
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "v" / "0_1.npz"
        C.save_store(p, frames, [boxes["0_1"][str(f)] for f in frames], masks["0_1"], (H, W), K=5)
        s = C.CandidateStore(p)
        assert s.has(2) and not s.has(0) and not s.has(9)
        assert s.get(3) == []                                           # frame with no box
        c = s.get(2)
        assert len(c) == 2 and c[0]["score"] > c[1]["score"]            # best first
        assert c[0]["mask"].shape == (H, W) and c[0]["mask"].dtype == bool
        want = FakeSeg().mask(c[0]["box"])
        assert (c[0]["mask"] == want).all()
        assert np.allclose(c[0]["box"], [3, 1, 6, 5])
        assert not list(p.parent.glob("*.tmp*"))                        # temp file was renamed away


def test_store_pads_when_fewer_than_k_boxes_and_truncates_when_more():
    fps = paths(3)
    seg = FakeSeg()
    many = lambda p, t: [[0, 0, 2, 2, 0.5 - 0.01 * i] for i in range(8)]
    boxes = C.boxes_pass(fps, {"e": "x"}, [1, 2], many)
    masks = C.masks_pass(fps, boxes, [1, 2], seg)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "e.npz"
        C.save_store(p, [1, 2], [boxes["e"]["1"], boxes["e"]["2"]], masks["e"], (H, W), K=5)
        assert len(C.CandidateStore(p).get(1)) == 5


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok   {t.__name__}")
    print(f"all {len(tests)} tests passed")
