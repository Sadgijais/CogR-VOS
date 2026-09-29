"""Unit tests for sttm.py on synthetic data. No GPU, no dataset. Run:  python tools/test_sttm.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sttm import STTMParams, TargetMemory

H, W = 100, 100
rng = np.random.default_rng(0)


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def emb_near(base, noise):
    return unit(base + noise * rng.normal(size=base.shape))


def box_mask(x1, y1, x2, y2):
    m = np.zeros((H, W), bool)
    m[y1:y2, x1:x2] = True
    return m


BASE = unit(rng.normal(size=32))
OTHER = unit(rng.normal(size=32))
GOOD = box_mask(30, 30, 60, 60)         # 900 px, interior


def fresh(**kw):
    p = STTMParams(**kw)
    m = TargetMemory(p, (H, W), unit(rng.normal(size=8)), np.stack([OTHER]))
    m.set_anchor(0, GOOD, BASE)
    return m


def test_anchor_never_evicted():
    m = fresh(delta=1, K=3)
    for t in range(1, 200):
        m.update(t, GOOD, emb_near(BASE, 0.4), obj_score=5.0, pred_iou=0.9)
    assert m.anchor.is_anchor and m.anchor.frame_idx == 0
    assert 0 in m.read_frames()
    assert len(m.entries()) <= 3


def test_bank_never_exceeds_K():
    m = fresh(delta=1, K=4)
    for t in range(1, 300):
        m.update(t, GOOD, emb_near(BASE, 0.5), 5.0, 0.9)
        assert len(m.entries()) <= 4


def test_near_duplicate_is_merged_not_added():
    m = fresh(delta=1, K=4, merge_thr=0.95)
    m.update(5, GOOD, emb_near(BASE, 0.01), 5.0, 0.9)
    n = len(m.curated)
    d = m.update(10, GOOD, emb_near(BASE, 0.01), 5.0, 0.9)
    assert d["written"] and d["action"] == "merged"
    assert len(m.curated) == n


def test_frozen_while_absent():
    m = fresh(delta=1, K=4)
    for t in range(1, 30):
        m.update(t, GOOD, emb_near(BASE, 0.3), 5.0, 0.9)
    before = [(e.frame_idx, e.reliability) for e in m.entries()]
    w0, e0 = m.stats["writes"], m.stats["evictions"]
    empty = np.zeros((H, W), bool)
    for t in range(30, 60):
        d = m.update(t, empty, None, obj_score=-8.0, pred_iou=0.0)
        assert not d["written"] and d["blocked_by"] == "not_visible"
    assert [(e.frame_idx, e.reliability) for e in m.entries()] == before
    assert m.stats["writes"] == w0 and m.stats["evictions"] == e0


def test_low_object_score_counts_as_absent():
    m = fresh(delta=1)
    d = m.update(3, GOOD, emb_near(BASE, 0.1), obj_score=-2.0, pred_iou=0.9)
    assert d["blocked_by"] == "not_visible"


def test_pacing():
    m = fresh(delta=10)
    d = m.update(3, GOOD, emb_near(BASE, 0.1), 5.0, 0.9)
    assert d["blocked_by"] == "paced"
    d = m.update(11, GOOD, emb_near(BASE, 0.1), 5.0, 0.9)
    assert d["written"]


def test_dissimilar_appearance_blocked():
    m = fresh(delta=1, tau_sim=0.5)
    d = m.update(3, GOOD, OTHER, 5.0, 0.9)
    assert d["blocked_by"] == "low_similarity"
    assert len(m.curated) == 0


def test_truncated_at_border_blocked():
    m = fresh(delta=1)
    m.update(1, box_mask(60, 30, 100, 70), emb_near(BASE, 0.1), 5.0, 0.9)  # big, touches right border
    d = m.update(2, box_mask(80, 30, 100, 50), emb_near(BASE, 0.1), 5.0, 0.9)  # shrinking at border
    assert d["blocked_by"] == "truncated"


def test_reappearance_event_logged():
    m = fresh(delta=1)
    empty = np.zeros((H, W), bool)
    m.update(5, empty, None, -8.0, 0.0)
    m.update(6, empty, None, -8.0, 0.0)
    m.update(9, GOOD, emb_near(BASE, 0.1), 5.0, 0.9)
    kinds = [e["event"] for e in m.events]
    assert kinds == ["visible", "absent", "reappeared"], kinds
    assert m.events[-1]["absent_frames"] == 4


def test_evicts_lowest_reliability_times_freshness():
    m = fresh(delta=1, K=3, merge_thr=0.999999, half_life=1e9)   # no decay, no merging
    m.update(1, GOOD, emb_near(BASE, 0.1), 1.0, 0.30)   # low reliability
    m.update(2, GOOD, emb_near(BASE, 0.1), 6.0, 0.95)   # high reliability
    assert len(m.curated) == 2
    low_frame = min(m.curated, key=lambda e: e.reliability).frame_idx
    d = m.update(3, GOOD, emb_near(BASE, 0.1), 6.0, 0.95)
    assert d["written"] and d["action"] == "evicted"
    assert low_frame not in [e.frame_idx for e in m.curated]


def test_context_margin_recorded():
    m = fresh(delta=1)
    m.update(2, GOOD, emb_near(BASE, 0.1), 5.0, 0.9)
    assert m.context["margin"] is not None and m.context["margin"] > 0


def test_motion_velocity_tracks():
    m = fresh(delta=1)
    for t in range(1, 15):
        m.update(t, box_mask(30 + t, 30, 60 + t, 60), emb_near(BASE, 0.1), 5.0, 0.9)
    assert m.velocity[0] > 0.5 and abs(m.velocity[1]) < 0.2


def _rotating_embs(n, step):
    """n embeddings drifting smoothly away from BASE (each step small, total change large)."""
    d = unit(rng.normal(size=BASE.shape))
    d = unit(d - (d @ BASE) * BASE)                    # orthogonal to BASE
    return [unit(np.cos(i * step) * BASE + np.sin(i * step) * d) for i in range(1, n + 1)]


def test_gradual_change_deadlocks_anchor_mode_but_not_recent_mode():
    embs = _rotating_embs(40, 0.15)                     # fast change: > 1 rad within ~8 frames
    old = fresh(delta=5, sim_mode="anchor_proto")
    new = fresh(delta=5, sim_mode="proto_or_recent")
    for t, e in enumerate(embs, start=1):
        old.update(t, GOOD, e, 5.0, 0.9)
        new.update(t, GOOD, e, 5.0, 0.9)
    assert old.stats["blocked"]["low_similarity"] > 15          # the deadlock
    assert new.stats["blocked"]["low_similarity"] == 0          # continuity keeps it alive


def test_sudden_switch_stays_blocked_in_recent_mode():
    m = fresh(delta=1, sim_mode="proto_or_recent")
    for t in range(1, 6):
        m.update(t, GOOD, emb_near(BASE, 0.05), 5.0, 0.9)
    decs = [m.update(t, GOOD, emb_near(OTHER, 0.05), 5.0, 0.9) for t in range(6, 20)]
    assert all(d["blocked_by"] == "low_similarity" for d in decs)   # not fooled one frame later


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\nall {len(tests)} tests passed")
