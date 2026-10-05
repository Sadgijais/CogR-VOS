"""Tests for stage_events.py (V3 replay): the causal abstain rule, then the whole pipeline on the
synthetic video (fake SAM 2 -> real TCS log -> events -> oracle / scripted VLM -> blanked masks).
No GPU, no dataset, no network.   Run (from baseline_v0/):   python tools/test_stage_events.py"""
import csv
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import stage_events as se
import test_stage2_tcs_smoke as sm
import test_tcs_analysis as tta
import vlm as V
from common.io_utils import read_binary_mask


# ------------------------------------------------------------------ the abstain rule (pure)
def fr(n, c=0.95, visible=True):
    return [{"f": f, "c": c, "visible": visible} for f in range(1, n + 1)]


def idv(verdict, conf=0.9, kind="identity"):
    return {"kind": kind, "verdict": verdict, "confidence": conf}


def test_no_match_blanks_until_tcs_has_been_high_for_recover_frames():
    blank, _ = se.abstain_frames(fr(20), {5: idv("no_match")}, tau_high=0.9, recover_high=3)
    assert blank == [5, 6, 7]


def test_a_later_match_ends_the_abstention():
    blank, _ = se.abstain_frames(fr(20, c=0.4), {5: idv("no_match"), 9: idv("match")}, tau_high=0.9, recover_high=99)
    assert blank == [5, 6, 7, 8]


def test_max_abstain_is_a_safety_cap():
    blank, _ = se.abstain_frames(fr(30, c=0.4), {5: idv("no_match")}, tau_high=0.9, recover_high=99, max_abstain=4)
    assert blank == [5, 6, 7, 8]


def test_low_confidence_unsure_and_disabled_do_nothing():
    f = fr(15)
    assert se.abstain_frames(f, {5: idv("no_match", conf=0.3)}, 0.9)[0] == []
    assert se.abstain_frames(f, {5: idv("unsure")}, 0.9)[0] == []
    assert se.abstain_frames(f, {5: idv("no_match")}, 0.9, enabled=False)[0] == []


def test_presence_match_is_flagged_as_false_absence_but_changes_no_mask():
    blank, fa = se.abstain_frames(fr(10, visible=False, c=0.0), {4: idv("match", kind="presence")}, 0.9)
    assert blank == [] and fa == [4]
    assert se.abstain_frames(fr(10), {4: idv("not_visible", kind="presence")}, 0.9) == ([], [])


# ------------------------------------------------------------------ end to end
def csv_rows(p):
    return list(csv.DictReader(open(p)))


def build(tmp: Path):
    sm.install_fakes(tmp)
    cfg_p = sm.build_dataset(tmp, gate=False)
    sm.run_stage2(cfg_p)                                            # fake SAM 2, real TCS -> tcs_log + PNG masks
    tta.write_gt(tmp)                                               # target hidden at 30-31, distractor 15-25
    cfg = yaml.safe_load(cfg_p.read_text())
    cfg["paths"].update(predictions_out=str(tmp / "pred3"), results_out=str(tmp / "res3"))
    cfg["target"] = {"ref_davis17_jf": 66.2}
    cfg["v3"] = {"source_results": str(tmp / "res"), "source_preds": str(tmp / "pred"), "schedule": "events",
                 "action": {"mode": "abstain", "conf_min": 0.6, "recover_high": 5, "max_abstain": 40},
                 "events": {"drop_delta": 0.30, "drop_window": 3, "persist_frames": 5, "margin_thresh": 0.0,
                            "refractory": 2, "keepalive": 0, "max_calls": 0, "fire_on_disappear": True},
                 "vlm": {"provider": "oracle", "cache_dir": str(tmp / "cache"), "workers": 4}}
    p = tmp / "cfg_v3.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def run(cfg_p, *extra):
    sys.argv = ["stage_events.py", "--config", str(cfg_p), *extra]
    se.main()


def blank_frames(tmp, root="pred3"):
    out = []
    for t in range(sm.N):
        if not read_binary_mask(tmp / root / "vid" / "0" / f"{t:05d}.png").any():
            out.append(t)
    return out


def test_pipeline_oracle_blanks_the_drift_frames_and_leaves_the_rest_identical():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cfg_p = build(tmp)
        src_blank = blank_frames(tmp, "pred")                        # frames the V2 run already left empty (30, 31)
        run(cfg_p, "--vlm", "oracle")
        log = csv_rows(tmp / "res3" / "vlm_log.csv")
        events = {int(r["frame"]): r for r in log}
        assert events[20]["event"] == "coherence_drop" and events[20]["verdict"] == "no_match"
        assert events[30]["event"] == "disappear" and events[30]["kind"] == "presence" and events[30]["verdict"] == "not_visible"
        assert events[32]["event"] == "reappear" and events[32]["verdict"] == "match"
        assert all(r["correct"] == "True" for r in log), [r for r in log if r["correct"] != "True"]
        b = blank_frames(tmp)
        assert {20, 21, 22} <= set(b) and set(src_blank) <= set(b)
        assert 10 not in b and 5 not in b
        for t in range(sm.N):                                        # every non-blanked frame is byte-identical to V2
            if t not in b:
                a = (tmp / "pred" / "vid" / "0" / f"{t:05d}.png").read_bytes()
                c = (tmp / "pred3" / "vid" / "0" / f"{t:05d}.png").read_bytes()
                assert a == c, t
        s = (tmp / "res3" / "v3_summary.txt").read_text()
        assert "GROUND-TRUTH ORACLE" in s and "calls_per_tracked_frame" in s
        row = csv_rows(tmp / "res3" / "events_per_expression.csv")[0]
        assert int(row["calls"]) == len(log) and int(row["blanked_frames"]) >= 3
        print("   (synthetic) calls:", len(log), "| calls per tracked frame:", round(len(log) / (sm.N - 1), 3),
              "| blanked:", row["blanked_frames"])


def test_log_only_changes_no_mask_and_rerun_overwrites_cleanly():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cfg_p = build(tmp)
        run(cfg_p, "--vlm", "oracle")                                # abstain run first, writes blanks
        run(cfg_p, "--vlm", "oracle", "--action", "log_only")        # then log_only into the same folder
        assert blank_frames(tmp) == blank_frames(tmp, "pred")        # back to exactly the V2 masks


def test_schedules_every_frame_and_matched_periodic():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cfg_p = build(tmp)
        run(cfg_p, "--vlm", "oracle")
        n_events = len(csv_rows(tmp / "res3" / "vlm_log.csv"))
        run(cfg_p, "--vlm", "oracle", "--schedule", "every_frame", "--action", "log_only")
        assert len(csv_rows(tmp / "res3" / "vlm_log.csv")) == sm.N - 1           # V5 upper bound: every tracked frame
        run(cfg_p, "--vlm", "oracle", "--schedule", "periodic", "--period", "auto", "--action", "log_only")
        n_per = len(csv_rows(tmp / "res3" / "vlm_log.csv"))
        assert abs(n_per - n_events) <= max(2, n_events), (n_per, n_events)       # same order of budget


def test_match_period_hits_the_event_budget_on_short_tracks():
    logs = [("v", str(i), {"frames": [{"f": f} for f in range(1, 64)]}) for i in range(10)]   # 10 targets x 63 frames
    p = se.match_period(logs, 20)
    total = sum(1 for _, _, e in logs for r in e["frames"] if r["f"] % p == 0)
    assert total == 20, (p, total)
    naive = max(1, round(630 / 20))                                                          # the old rule: 32 -> only 10 calls
    assert sum(1 for _, _, e in logs for r in e["frames"] if r["f"] % naive == 0) == 10
    assert se.match_period([], 5) == 1


def test_oracle_iou_threshold_changes_what_counts_as_a_match():
    row = lambda: {"verdict": "match", "kind": "identity", "gt_present": True, "gt_iou": 0.30}
    old = se.MATCH_IOU
    try:
        se.MATCH_IOU = 0.5
        assert se._score(row())["correct"] is False          # a sloppy mask (IoU 0.30) is "no_match" under the 0.5 oracle
        se.MATCH_IOU = 0.10
        assert se._score(row())["correct"] is True           # but still the right object under the identity oracle
    finally:
        se.MATCH_IOU = old


def test_scripted_vlm_path_parses_answers_caches_and_survives_errors():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cfg_p = build(tmp)
        seen = {"n": 0}

        def script(images, prompt):
            seen["n"] += 1
            if "NOT in this frame" in prompt:
                return '{"verdict": "not_visible", "confidence": 0.9, "reason": "empty"}'
            if len(images) == 2 and seen["n"] == 2:
                raise RuntimeError("API hiccup")                         # one failed call must not kill the run
            return '```json\n{"verdict": "no_match", "confidence": 0.9, "reason": "other object"}\n```'

        orig = V.make_client
        V.make_client = lambda vcfg: V.CachedVLM(V.MockVLM(script), tmp / "cache")
        try:
            run(cfg_p, "--vlm", "mock")
            log = csv_rows(tmp / "res3" / "vlm_log.csv")
            assert {r["verdict"] for r in log} <= {"no_match", "not_visible", "unsure"} and len(log) >= 3
            assert any("error" in r["reason"] for r in log) or seen["n"] >= 1
            first_calls = seen["n"]
            run(cfg_p, "--vlm", "mock")                                  # second run: answers come from the cache
            assert seen["n"] <= first_calls + 1                          # (only the one errored call is retried)
            log2 = csv_rows(tmp / "res3" / "vlm_log.csv")
            assert sum(r["cached"] == "True" for r in log2) >= len(log2) - 1
        finally:
            V.make_client = orig


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\nall {len(tests)} tests passed")
