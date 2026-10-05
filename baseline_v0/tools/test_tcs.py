"""Unit tests for tcs.py (+ the allow_write switch in sttm.py) on synthetic data.
No GPU, no dataset, no SAM 2.   Run:  python tools/test_tcs.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sttm import STTMParams, TargetMemory
from tcs import TCSParams, TrackletCoherence, band_of, mask_iou, shift_mask

H, W = 120, 160
rng = np.random.default_rng(1)


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


BASE = unit(rng.normal(size=32))
OTHER = unit(rng.normal(size=32))


def near(base, noise):
    return unit(base + noise * rng.normal(size=base.shape))


def box_mask(x1, y1, x2, y2):
    m = np.zeros((H, W), bool)
    m[y1:y2, x1:x2] = True
    return m


def blob(t, vx=2):
    """30x30 target moving right at vx px/frame, starting at x=20."""
    x = 20 + vx * t
    return box_mask(x, 40, x + 30, 70)


def fresh(**tcs_kw):
    mem = TargetMemory(STTMParams(delta=1), (H, W), unit(rng.normal(size=8)), np.stack([OTHER]))
    mem.set_anchor(0, blob(0), BASE)
    tcs = TrackletCoherence(TCSParams(**tcs_kw))
    tcs.set_anchor(blob(0), 0.30)
    return mem, tcs


def step(mem, tcs, t, mask, emb, sem=0.30, obj=6.0, conf=0.95):
    """What stage2_tcs.py does each frame: TCS first, then the memory update."""
    res = tcs.score(t, mem, mask, obj, conf, emb, sem)
    mem.update(t, mask, emb, obj, conf)
    return res


def warm_up(mem, tcs, n=8):
    last = None
    for t in range(1, n + 1):
        last = step(mem, tcs, t, blob(t), near(BASE, 0.05))
    return last


# ------------------------------------------------------------------ helpers
def test_shift_mask_moves_without_wrap():
    m = box_mask(10, 10, 20, 20)
    s = shift_mask(m, 5, 3)
    assert s.sum() == m.sum() and s[13:23, 15:25].all()
    far = shift_mask(m, W + 5, 0)
    assert far.sum() == 0
    edge = shift_mask(box_mask(W - 10, 10, W, 20), 4, 0)   # pushed partly off the right edge
    assert edge.sum() == 6 * 10


def test_band_of_boundaries():
    p = TCSParams(tau_low=0.5, tau_high=0.7)
    assert band_of(0.7, p) == "HIGH" and band_of(0.699, p) == "MEDIUM"
    assert band_of(0.5, p) == "MEDIUM" and band_of(0.499, p) == "LOW"


# --------------------------------------------------------------- score logic
def test_steady_track_is_high_and_terms_near_one():
    mem, tcs = fresh()
    res = warm_up(mem, tcs)
    assert res["visible"] and res["state"] == "visible"
    assert res["band"] == "HIGH" and res["c"] > 0.85
    for name in ("mask", "motion", "appearance", "semantic"):
        assert res["terms"][name] is not None and res["terms"][name] > 0.8, name


def test_teleporting_mask_hurts_motion_term_only_by_design():
    # Same shape, far away: shape agreement aligns centroids, so the MASK term stays
    # high and the MOTION term is what catches the jump.
    mem, tcs = fresh()
    warm_up(mem, tcs)
    steady = step(mem, tcs, 9, blob(9), near(BASE, 0.05))
    jumped = step(mem, tcs, 10, box_mask(110, 5, 140, 35), near(BASE, 0.05))
    assert jumped["terms"]["motion"] < 0.1 < steady["terms"]["motion"]
    assert abs(jumped["terms"]["mask"] - steady["terms"]["mask"]) < 0.05
    assert jumped["c"] < steady["c"] and jumped["band"] != "HIGH"


def test_arithmetic_mean_hides_a_single_collapsed_term_geometric_does_not():
    # WHY the default is geometric: a teleport collapses only the motion term.
    # The plain average of four terms stays above the HIGH threshold; the geometric mean does not.
    out = {}
    for agg in ("arithmetic", "geometric"):
        mem, tcs = fresh(agg=agg)
        warm_up(mem, tcs)
        out[agg] = step(mem, tcs, 9, box_mask(110, 5, 140, 35), near(BASE, 0.05))
    assert out["arithmetic"]["band"] == "HIGH"        # the weakness
    assert out["geometric"]["band"] == "LOW"          # the fix
    assert out["geometric"]["c"] < out["arithmetic"]["c"] - 0.3


def test_shape_change_hurts_mask_term():
    # Same place, very different shape (30x30 square -> 80x6 sliver of similar centroid).
    mem, tcs = fresh()
    warm_up(mem, tcs)
    steady = step(mem, tcs, 9, blob(9), near(BASE, 0.05))
    x = 20 + 2 * 10
    sliver = box_mask(x - 25, 54, x + 55, 60)        # centroid ~ (x+15, 57), area 480 >= a_min
    res = step(mem, tcs, 10, sliver, near(BASE, 0.05))
    assert res["terms"]["mask"] < steady["terms"]["mask"] - 0.1
    assert res["c"] < steady["c"]


def test_wrong_identity_hurts_appearance_term():
    mem, tcs = fresh()
    warm_up(mem, tcs)
    good = step(mem, tcs, 9, blob(9), near(BASE, 0.05))
    bad = step(mem, tcs, 10, blob(10), near(OTHER, 0.05))
    assert bad["terms"]["appearance"] < 0.4 < good["terms"]["appearance"]
    assert bad["c"] < good["c"]


def test_semantic_term_is_relative_to_frame0():
    mem, tcs = fresh()
    warm_up(mem, tcs)
    same = step(mem, tcs, 9, blob(9), near(BASE, 0.05), sem=0.30)
    half = step(mem, tcs, 10, blob(10), near(BASE, 0.05), sem=0.15)
    better = step(mem, tcs, 11, blob(11), near(BASE, 0.05), sem=0.45)
    assert abs(same["terms"]["semantic"] - 1.0) < 1e-6
    assert abs(half["terms"]["semantic"] - 0.5) < 1e-6
    assert better["terms"]["semantic"] == 1.0          # capped at 1


def test_missing_term_renormalises_weights():
    mem, tcs = fresh()
    warm_up(mem, tcs)
    res = step(mem, tcs, 9, blob(9), near(BASE, 0.05), sem=None)
    assert res["terms"]["semantic"] is None
    vals = [v for v in res["terms"].values() if v is not None]
    assert len(vals) == 3
    assert abs(res["c"] - float(np.exp(np.mean(np.log(vals))))) < 1e-9      # default = geometric mean
    mem2, tcs2 = fresh(agg="arithmetic")
    warm_up(mem2, tcs2)
    res2 = step(mem2, tcs2, 9, blob(9), near(BASE, 0.05), sem=None)
    vals2 = [v for v in res2["terms"].values() if v is not None]
    assert abs(res2["c"] - float(np.mean(vals2))) < 1e-9


def test_zero_weight_removes_a_term():
    mem, tcs = fresh(w_appearance=0.0)
    warm_up(mem, tcs)
    good = step(mem, tcs, 9, blob(9), near(BASE, 0.05))
    swapped = step(mem, tcs, 10, blob(10), near(OTHER, 0.05))
    # appearance collapses, but with weight 0 only the other three terms drive c
    assert swapped["terms"]["appearance"] < 0.4
    assert abs(swapped["c"] - good["c"]) < 0.1


def test_absence_is_not_drift():
    mem, tcs = fresh()
    warm_up(mem, tcs)
    gone = step(mem, tcs, 9, np.zeros((H, W), bool), None, sem=None, obj=-5.0, conf=0.0)
    assert gone["visible"] is False and gone["state"] == "absent"
    assert gone["c"] == 0.0 and gone["band"] == "LOW"
    assert all(v is None for v in gone["terms"].values())
    back = step(mem, tcs, 10, blob(10), near(BASE, 0.05))
    assert back["state"] == "reappeared"
    assert back["terms"]["motion"] is None              # stale prediction after a gap is not used
    assert back["terms"]["appearance"] is not None


def test_score_does_not_change_the_memory():
    mem, tcs = fresh()
    warm_up(mem, tcs)
    before = (mem.last_centroid, mem.last_area, mem.velocity, mem.predicted_next,
              mem.visible_prev, mem.stats["frames"], list(mem.read_frames()))
    tcs.score(9, mem, blob(9), 6.0, 0.95, near(BASE, 0.05), 0.30)
    after = (mem.last_centroid, mem.last_area, mem.velocity, mem.predicted_next,
             mem.visible_prev, mem.stats["frames"], list(mem.read_frames()))
    assert before == after


def test_counts_add_up():
    mem, tcs = fresh()
    warm_up(mem, tcs, 6)
    step(mem, tcs, 7, np.zeros((H, W), bool), None, sem=None, obj=-5.0, conf=0.0)
    s = tcs.summary()
    assert s["counts"]["absent"] == 1
    assert s["counts"]["HIGH"] + s["counts"]["MEDIUM"] + s["counts"]["LOW"] == 7


# ----------------------------------------------- allow_write switch in sttm.py
def test_allow_write_false_blocks_write_but_keeps_tracking_state():
    mem, _ = fresh()
    for t in range(1, 6):
        mem.update(t, blob(t), near(BASE, 0.05), 6.0, 0.95)           # default: V1 behaviour
    bank = list(mem.read_frames())
    d = mem.update(6, blob(6), near(BASE, 0.05), 6.0, 0.95, allow_write=False)
    assert d["written"] is False and d["blocked_by"] == "tcs_uncertain"
    assert mem.read_frames() == bank
    assert mem.stats["blocked"]["tcs_uncertain"] == 1
    assert mem.last_centroid is not None and abs(mem.last_centroid[0] - (blob(6).nonzero()[1].mean())) < 1e-6


def test_allow_write_default_is_v1_behaviour():
    a = TargetMemory(STTMParams(delta=1), (H, W), unit(np.ones(8)), None)
    b = TargetMemory(STTMParams(delta=1), (H, W), unit(np.ones(8)), None)
    a.set_anchor(0, blob(0), BASE)
    b.set_anchor(0, blob(0), BASE)
    r = np.random.default_rng(5)
    for t in range(1, 40):
        e = unit(BASE + 0.3 * r.normal(size=32))
        da = a.update(t, blob(t), e, 5.0, 0.9)
        db = b.update(t, blob(t), e, 5.0, 0.9, allow_write=True)
        assert da == db
    assert a.read_frames() == b.read_frames()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\nall {len(tests)} tests passed")
