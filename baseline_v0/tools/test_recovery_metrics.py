#!/usr/bin/env python3
"""CPU tests for tools/recovery_metrics.py (pure logic, no dataset, no GPU, no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery_metrics as R

P = R.Params()


def track(n, absent=(), ious=None, empty=None):
    """present[k] False for k in absent; iou default 0.9 where present; pred_empty False unless given."""
    present = [k not in set(absent) for k in range(n)]
    iou = [0.9 if present[k] else 0.0 for k in range(n)]
    for k, v in (ious or {}).items():
        iou[k] = v
    return present, iou, [False] * n if empty is None else empty


def test_no_absence_means_no_events():
    pr, io, em = track(30)
    r = R.analyze_track(pr, io, em, P)
    assert r["events"] == [] and r["vanish_without_return"] == 0 and r["loss"] == []


def test_instant_recovery_has_latency_zero():
    pr, io, em = track(30, absent=range(10, 14))
    e = R.analyze_track(pr, io, em, P)["events"]
    assert len(e) == 1
    assert e[0]["first_visible_frame"] == 14 and e[0]["length"] == 4
    assert e[0]["recovered_in_window"] and e[0]["latency_frames"] == 0


def test_slow_recovery_latency_is_frames_to_first_good_iou():
    pr, io, em = track(40, absent=range(10, 13), ious={13: 0.2, 14: 0.3, 15: 0.1, 16: 0.2, 17: 0.65})
    e = R.analyze_track(pr, io, em, P)["events"][0]
    assert e["recovered_in_window"] and e["latency_frames"] == 4


def test_window_boundary_is_inclusive():
    # first visible frame 13; frame 23 is exactly 10 frames later, frame 24 is 11 later
    bad = {k: 0.2 for k in range(13, 40)}
    pr, io, em = track(40, absent=range(10, 13), ious={**bad, 23: 0.7})
    e = R.analyze_track(pr, io, em, P)["events"][0]
    assert e["recovered_in_window"] and e["latency_frames"] == 10
    pr, io, em = track(40, absent=range(10, 13), ious={**bad, 24: 0.7})
    e = R.analyze_track(pr, io, em, P)["events"][0]
    assert not e["recovered_in_window"] and e["latency_frames"] is None and e["eventual_latency_frames"] == 11


def test_never_recovered_event():
    bad = {k: 0.05 for k in range(13, 40)}
    pr, io, em = track(40, absent=range(10, 13), ious=bad)
    e = R.analyze_track(pr, io, em, P)["events"][0]
    assert not e["recovered_in_window"] and e["eventual_latency_frames"] is None


def test_one_frame_blip_is_not_a_disappearance():
    pr, io, em = track(30, absent=[10])
    assert R.analyze_track(pr, io, em, P)["events"] == []


def test_vanishing_without_return_is_counted_but_is_not_an_event():
    pr, io, em = track(30, absent=range(20, 30))
    r = R.analyze_track(pr, io, em, P)
    assert r["events"] == [] and r["vanish_without_return"] == 1


def test_target_not_yet_visible_at_the_start_is_not_a_reappearance():
    pr, io, em = track(30, absent=range(0, 6))
    r = R.analyze_track(pr, io, em, P)
    assert r["events"] == [] and r["vanish_without_return"] == 0


def test_target_vanishing_again_inside_the_window_stops_the_scan():
    # back at 13, gone again at 15, good IoU only after that: not recovered in the window
    pr, io, em = track(40, absent=[10, 11, 12, 15, 16], ious={13: 0.1, 14: 0.1})
    e = R.analyze_track(pr, io, em, P)["events"]
    assert e[0]["first_visible_frame"] == 13 and not e[0]["recovered_in_window"]
    assert len(e) == 2          # the second absence (15-16) is also a reappearance event


def test_loss_episode_needs_five_visible_frames_and_reports_recovery():
    ious = {k: 0.05 for k in range(10, 16)}                      # frames 10-15 lost
    ious[20] = 0.8
    ious.update({k: 0.3 for k in range(16, 20)})                 # neither lost nor good
    pr, io, em = track(30, ious=ious)
    ep = R.analyze_track(pr, io, em, P)["loss"]
    assert len(ep) == 1 and ep[0]["length"] == 6 and ep[0]["start_frame"] == 10
    assert ep[0]["recovered"] and ep[0]["latency_frames"] == 10   # frame 10 -> first good frame is 20 (16-19 are only 0.3)


def test_short_loss_run_is_not_an_episode():
    pr, io, em = track(30, ious={k: 0.05 for k in range(10, 14)})
    assert R.analyze_track(pr, io, em, P)["loss"] == []


def test_loss_episode_never_recovered():
    pr, io, em = track(30, ious={k: 0.05 for k in range(5, 30)})
    ep = R.analyze_track(pr, io, em, P)["loss"]
    assert len(ep) == 1 and not ep[0]["recovered"] and ep[0]["latency_frames"] is None


def test_frame_counts_skip_first_and_last_frame():
    n = 10
    empty = [True] * n                                            # every mask empty
    pr, io, _ = track(n, absent=[4, 5])
    c = R.frame_counts(pr, empty, P)
    assert c["scored_frames"] == 8 and c["visible_frames"] == 6 and c["nre_frames"] == 6
    assert c["absent_frames"] == 2 and c["false_presence_frames"] == 0
    c2 = R.frame_counts(pr, empty, R.Params(exclude_first_last=False))
    assert c2["scored_frames"] == 10 and c2["nre_frames"] == 8


def test_false_presence_counts_nonempty_masks_while_absent():
    n = 10
    pr, io, _ = track(n, absent=[4, 5, 6])
    empty = [False] * n
    empty[5] = True
    c = R.frame_counts(pr, empty, P)
    assert c["absent_frames"] == 3 and c["false_presence_frames"] == 2 and c["nre_frames"] == 0


def test_length_mismatch_is_an_error():
    try:
        R.analyze_track([True, True], [0.5], [False, False], P)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_summary_prints_counts_next_to_rates():
    a = R.analyze_track(*track(30, absent=range(10, 13)), P)                                   # recovered
    b = R.analyze_track(*track(40, absent=range(10, 13), ious={k: 0.05 for k in range(13, 40)}), P)  # not recovered
    s, text = R.summarize([("v", "0_1", a), ("v", "0_2", b)], P, "demo")
    assert s["reappearance_events"] == 2 and s["recovered_in_window"] == 1 and s["never_recovered"] == 1
    assert "1 of 2 (50.0%)" in text and "anecdote" in text
    assert s["latency_mean"] == 0 and s["expressions_with_reappearance"] == 2


def test_summary_with_no_events_says_so_instead_of_dividing_by_zero():
    a = R.analyze_track(*track(30), P)
    s, text = R.summarize([("v", "0_1", a)], P, "demo")
    assert "n/a (no events)" in text and s["latency_mean"] is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok   {t.__name__}")
    print(f"all {len(tests)} tests passed")
