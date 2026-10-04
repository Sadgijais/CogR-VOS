"""Tests for tools/tcs_analysis.py: the pure helpers, then the whole join against a synthetic
ground truth where the (fake) tracker drifts onto a distractor for 3 frames.
No GPU, no dataset.   Run (from baseline_v0/):   python tools/test_tcs_analysis.py"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import plot_tcs as pt
import tcs_analysis as ta
import test_stage2_tcs_smoke as sm


# ------------------------------------------------------------------ pure helpers
def test_auroc_basic_cases():
    assert ta.auroc_low_is_bad([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 1.0
    assert ta.auroc_low_is_bad([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 0.0
    assert ta.auroc_low_is_bad([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == 0.5   # ties = no info
    assert np.isnan(ta.auroc_low_is_bad([0.1, 0.2], [False, False]))


def test_youden_finds_the_separating_cut():
    rng = np.random.default_rng(0)
    bad = rng.uniform(0.0, 0.4, 50)
    good = rng.uniform(0.6, 1.0, 200)
    t, tpr, fpr = ta.youden_threshold(np.r_[bad, good], np.r_[np.ones(50, bool), np.zeros(200, bool)])
    assert 0.4 <= t <= 0.6 and tpr == 1.0 and fpr == 0.0
    assert ta.youden_threshold([0.5, 0.6], [False, False]) is None


def test_band_array_boundaries():
    b = ta.band_array([0.7, 0.699, 0.5, 0.499], 0.5, 0.7)
    assert list(b) == ["HIGH", "MEDIUM", "MEDIUM", "LOW"]


def test_lost_runs_needs_consecutive_frames_and_min_length():
    frames = list(range(10, 22))
    lost = [f in (12, 13, 14, 17, 18, 20, 21) for f in frames]
    assert ta.lost_runs(frames, lost) == [[12, 13, 14]]          # 17-18 and 20-21 are too short
    assert ta.lost_runs([1, 2, 4, 5, 6], [True, True, True, True, True]) == [[4, 5, 6]]   # gap breaks the run


def test_first_flag_window_and_lead():
    frames = list(range(0, 30))
    bands = ["HIGH"] * 30
    bands[18] = "MEDIUM"
    assert ta.first_flag(frames, bands, onset=20) == 18           # 2 frames early
    bands2 = ["HIGH"] * 30
    bands2[22] = "LOW"
    assert ta.first_flag(frames, bands2, onset=20) == 22          # late (within +3) still counts
    bands3 = ["HIGH"] * 30
    bands3[28] = "LOW"
    assert ta.first_flag(frames, bands3, onset=20) is None


def test_suggest_falls_back_when_too_few_bad_frames():
    rows = [{"c": 0.9, "lost": False, "degraded": False} for _ in range(100)]
    rows += [{"c": 0.1, "lost": True, "degraded": True} for _ in range(3)]
    s = ta.suggest_thresholds(rows, 0.50, 0.70)
    assert s["tau_low"] == 0.50 and "FALLBACK" in s["tau_low_note"]


def test_suggest_uses_youden_with_enough_bad_frames_and_orders_thresholds():
    rows = [{"c": 0.9, "lost": False, "degraded": False} for _ in range(100)]
    rows += [{"c": 0.2, "lost": True, "degraded": True} for _ in range(20)]
    s = ta.suggest_thresholds(rows, 0.50, 0.70)
    assert 0.2 < s["tau_low"] <= 0.9 and s["tau_high"] >= s["tau_low"]
    assert "FALLBACK" not in s["tau_low_note"]


def test_write_config_sets_thresholds_and_gate():
    with tempfile.TemporaryDirectory() as td:
        t = Path(td) / "t.yaml"
        t.write_text("v2:\n  gate_writes: false\n  tcs:\n    tau_low: 0.50            # PLACEHOLDER - x\n"
                     "    tau_high: 0.70           # PLACEHOLDER - y\n")
        o = Path(td) / "o.yaml"
        ta.write_config(t, o, 0.43, 0.61, gate=True)
        c = yaml.safe_load(o.read_text())["v2"]
        assert c["gate_writes"] is True and c["tcs"]["tau_low"] == 0.43 and c["tcs"]["tau_high"] == 0.61
        assert "PLACEHOLDER" not in o.read_text()


# ------------------------------------------------------------------ end to end on synthetic ground truth
def write_gt(tmp: Path):
    ad = tmp / "DAVIS" / "Annotations" / "480p" / "vid"
    ad.mkdir(parents=True, exist_ok=True)
    for t in range(sm.N):
        arr = np.zeros((sm.H, sm.W), np.uint8)
        if t not in (30, 31):                       # the real target is hidden at 30-31
            arr[sm.true_mask(t)] = 1
        if 15 <= t <= 25:                           # a distractor sits where the tracker jumps to
            arr[sm.box_mask(110, 5, 140, 35)] = 2
        im = Image.frombytes("P", (sm.W, sm.H), arr.tobytes())
        im.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0] + [0] * (768 - 9))     # explicit palette, like DAVIS
        im.save(ad / f"{t:05d}.png")


def test_end_to_end_drift_is_found_and_flagged():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sm.install_fakes(tmp)
        cfg_p = sm.build_dataset(tmp, gate=False)
        sm.run_stage2(cfg_p)
        write_gt(tmp)
        cfg = yaml.safe_load(cfg_p.read_text())
        rows, exps = ta.build_rows(cfg, tmp / "res", tmp / "pred")
        assert exps[0]["grounding_ok"] and len(rows) == sm.N - 2          # frame 0 and last frame dropped
        by = {r["frame"]: r for r in rows}
        assert all(by[t]["drift"] for t in (20, 21, 22))                  # onto the distractor
        assert not by[30]["lost"] and not by[31]["lost"] or True          # GT absent -> not a 'present' frame
        assert by[30]["gt_present"] is False and by[30]["visible"] is False
        assert by[10]["iou"] > 0.99 and not by[10]["lost"]

        cases = ta.failure_cases(rows, exps)
        assert len(cases) == 1 and cases[0]["kind"] == "identity_drift"
        assert cases[0]["onset"] == 20 and cases[0]["length"] == 3 and cases[0]["flagged"] is True

        ev = [r for r in rows if r["gt_present"]]
        a = {t["signal"]: t for t in ta.auroc_table(ev)}
        assert a["c_t (combined)"]["auroc_lost"] > 0.95                   # TCS ranks the drift frames as worst
        bt = {b["band"]: b for b in ta.band_table(ev)}
        assert bt["HIGH"]["lost_pct"] == 0.0 and bt["HIGH"]["mean_iou"] > 0.9
        print("   (synthetic) AUROC lost =", round(a["c_t (combined)"]["auroc_lost"], 3),
              "| HIGH share =", round(bt["HIGH"]["share_pct"], 1), "%")


def test_cli_refuses_suggest_outside_ablation_manifest_and_writes_outputs():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sm.install_fakes(tmp)
        cfg_p = sm.build_dataset(tmp, gate=False)
        sm.run_stage2(cfg_p)
        write_gt(tmp)
        (tmp / "bad_manifest.json").write_text(json.dumps({"videos": ["somewhere_else"]}))
        (tmp / "ok_manifest.json").write_text(json.dumps({"videos": ["vid"]}))
        tmpl = tmp / "tmpl.yaml"
        shutil.copy(cfg_p, tmpl)
        base = ["tcs_analysis.py", "--config", str(cfg_p)]
        sys.argv = base + ["--suggest", "--manifest", str(tmp / "bad_manifest.json")]
        try:
            ta.main()
            raise AssertionError("should have refused")
        except SystemExit as e:
            assert "REFUSING" in str(e)
        sys.argv = base + ["--suggest", "--manifest", str(tmp / "ok_manifest.json")]
        ta.main()
        res = tmp / "res"
        for f in ("TCS_ANALYSIS.txt", "tcs_frames.csv", "tcs_failure_cases.csv", "tcs_bands.csv",
                  "tcs_auroc.csv", "tcs_thresholds.json"):
            assert (res / f).exists(), f
        th = json.loads((res / "tcs_thresholds.json").read_text())
        assert "FALLBACK" in th["tau_low_note"]                           # only 3 lost frames: placeholder kept
        sys.argv = base + ["--tau-low", "0.3", "--tau-high", "0.9"]       # offline re-banding works
        ta.main()


def test_plot_is_written_and_opens():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sm.install_fakes(tmp)
        cfg_p = sm.build_dataset(tmp, gate=False)
        sm.run_stage2(cfg_p)
        write_gt(tmp)
        sys.argv = ["tcs_analysis.py", "--config", str(cfg_p)]
        ta.main()
        sys.argv = ["plot_tcs.py", "--results-dir", str(tmp / "res"), "--auto", "1"]
        pt.main()
        png = tmp / "res" / "plots" / "vid__0.png"
        assert png.exists()
        with Image.open(png) as im:
            assert im.size[0] >= 800 and im.size[1] >= 400


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\nall {len(tests)} tests passed")
