"""Tests for events.py (the event trigger) and the periodic control. Pure python.
Run (from baseline_v0/):   python tools/test_events.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from events import EventParams, EventTrigger, PeriodicTrigger

OK = (0.95, True, "visible", 0.2)


def run(trig, seq):
    out = {}
    for i, (c, v, s, m) in enumerate(seq, start=1):
        e = trig.step(i, c, v, s, m)
        if e:
            out[i] = e
    return out


def P(**kw):
    base = dict(tau_low=0.81, tau_high=0.92, refractory=10)
    base.update(kw)
    return EventParams(**base)


def test_a_quiet_track_fires_nothing():
    t = EventTrigger(P())
    assert run(t, [OK] * 60) == {} and t.calls == 0 and t.candidates == 0


def test_sudden_drop_fires_coherence_drop():
    seq = [OK] * 10 + [(0.60, True, "visible", 0.2)] + [OK] * 5
    assert run(EventTrigger(P(tau_low=0.5)), seq) == {11: "coherence_drop"}      # delta rule only (0.6 is above tau_low)


def test_entering_low_band_fires_even_without_a_big_drop():
    seq = [(0.90, True, "visible", 0.2)] * 10 + [(0.80, True, "visible", 0.2)]
    assert run(EventTrigger(P()), seq) == {11: "coherence_drop"}


def test_disappearance_fires_once_not_every_absent_frame():
    seq = [OK] * 5 + [(0.0, False, "absent", None)] * 6 + [OK] * 3
    assert run(EventTrigger(P(refractory=1)), seq) == {6: "disappear"}


def test_disappear_can_be_switched_off():
    seq = [OK] * 5 + [(0.0, False, "absent", None)] * 3
    assert run(EventTrigger(P(fire_on_disappear=False)), seq) == {}


def test_reappearance_fires():
    seq = [OK] * 3 + [(0.0, False, "absent", None)] * 2 + [(0.85, True, "reappeared", 0.2)]
    assert run(EventTrigger(P(refractory=1)), seq) == {4: "disappear", 6: "reappear"}


def test_distractor_confusion_uses_the_margin():
    seq = [OK] * 5 + [(0.95, True, "visible", -0.10)] + [OK] * 3
    assert run(EventTrigger(P()), seq) == {6: "distractor"}
    assert run(EventTrigger(P()), [(0.95, True, "visible", None)] * 8) == {}       # no margin logged = ignored


def test_persistent_low_gives_drift_suspect_after_the_refractory_window():
    seq = [OK] * 4 + [(0.60, True, "visible", 0.2)] * 30
    out = run(EventTrigger(P()), seq)
    assert out == {5: "coherence_drop", 15: "drift_suspect", 25: "drift_suspect"}, out


def test_refractory_suppresses_and_counts():
    seq = [OK] * 5 + [(0.5, True, "visible", 0.2)] + [OK] * 2 + [(0.4, True, "visible", 0.2)] + [OK] * 3
    t = EventTrigger(P(refractory=10))
    out = run(t, seq)
    assert out == {6: "coherence_drop"} and t.calls == 1 and t.suppressed >= 1 and t.candidates >= 2


def test_max_calls_caps_cost():
    seq = ([OK] * 3 + [(0.4, True, "visible", 0.2)] + [OK] * 3) * 6
    t = EventTrigger(P(refractory=1, max_calls=2))
    run(t, seq)
    assert t.calls == 2 and t.suppressed >= 1


def test_keepalive_forces_a_check_on_a_long_quiet_stretch():
    out = run(EventTrigger(P(keepalive=20)), [OK] * 45)
    assert out == {20: "keepalive", 40: "keepalive"}, out


def test_first_event_inside_the_first_frames_is_allowed():
    out = run(EventTrigger(P()), [(0.5, True, "visible", 0.2)] + [OK] * 3)
    assert out == {1: "coherence_drop"}


def test_periodic_control_matches_a_fixed_period():
    t2 = PeriodicTrigger(5)
    fired = [f for f in range(1, 21) if t2.step(f)]
    assert fired == [5, 10, 15, 20] and t2.calls == 4
    t3 = PeriodicTrigger(1)
    assert [f for f in range(1, 6) if t3.step(f)] == [1, 2, 3, 4, 5]               # every frame = V5 bound


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\nall {len(tests)} tests passed")
