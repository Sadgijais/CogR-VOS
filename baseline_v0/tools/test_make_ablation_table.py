#!/usr/bin/env python3
"""CPU test for make_ablation_table.py on tiny fake result folders."""
import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_ablation_table as T


def summ(d, j, f, jf):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        for k, v in [("n_expressions_scored", 61), ("n_videos_scored", 30), ("mean_J", j), ("mean_F", f), ("mean_JF", jf)]:
            w.writerow([k, v])


def timing(d, rows):
    with open(d / "timing_per_video.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "inference_s"])
        w.writerows(rows)


def rec(d, events, recovered):
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"reappearance_events": events, "recovered_in_window": recovered, "latency_mean": 2.0, "loss_episodes": 5,
               "loss_recovered": 2, "nre_frames": 10, "visible_frames": 1000, "false_presence_frames": 5, "absent_frames": 100},
              open(d / "recovery_summary.json", "w"))


def test_table_from_fake_folders():
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        summ(t / "v0", 56.01, 44.94, 50.48)
        (t / "v0" / "timing_summary.txt").write_text("total_inference_s: 2125.7\n")
        summ(t / "v2", 56.48, 45.46, 50.97)
        (t / "v2" / "timing_summary.txt").write_text("total_inference_s: 1000.0\n")
        timing(t / "v2", [("a", 400.0), ("b", 300.0), ("c", 300.0)])
        summ(t / "v3", 56.12, 45.04, 50.58)
        (t / "v3" / "v3_summary.txt").write_text(
            "vlm_calls: 76\ncalls_per_tracked_frame: 0.0194\ncalls_per_video_frame: 0.0380\ntracked_target_frames: 3923\n"
            "video_frames: 1999\nestimated_v3_inference_s: 3139.6\n")
        summ(t / "v4", 57.0, 46.0, 51.5)
        (t / "v4" / "reid_summary.txt").write_text("variant: vlm\nvlm_calls: 39 (cached answers: 0)\ntotal_search_s: 5.0\ntotal_vlm_wait_s: 45.0\n")
        timing(t / "v4", [("a", 500.0), ("b", 350.0)])               # V4 re-ran a and b only
        rec(t / "r2", 3, 2)
        rec(t / "r4", 3, 3)
        rows = [("V0", t / "v0", None), ("V2", t / "v2", t / "r2"), ("V3", t / "v3", None), ("V4", t / "v4", t / "r4")]
        out = T.build(rows, t / "v2")
        by = {r["version"]: r for r in out}
        assert by["V0"]["JF"] == 50.48 and by["V0"]["time_s"] == 2125.7 and by["V0"]["calls"] == 0.0
        assert by["V3"]["calls"] == 76 and abs(by["V3"]["calls_per_tracked"] - 0.0194) < 1e-9 and by["V3"]["time_s"] == 3139.6
        assert abs(by["V4"]["time_s"] - 1050.0) < 1e-9, by["V4"]["time_s"]       # V2 1000 + 5 + 45
        assert by["V4"]["calls"] == 39 and abs(by["V4"]["calls_per_tracked"] - 39 / 3923) < 1e-12
        assert by["V0"]["recovery"] is None and by["V4"]["recovery"]["recovered"] == 3
        md = T.write(out, t / "out")
        assert "2 of 3" in md and "3 of 3" in md and "n/a" in md and (t / "out" / "ablation_table.csv").exists()
        assert "50.48" in md and "1050.0" in md


def test_missing_files_give_na_not_a_crash():
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        (t / "empty").mkdir()
        out = T.build([("V4", t / "empty", None)], None)
        assert out[0]["JF"] is None and "n/a" in T.write(out, t / "o")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok   {fn.__name__}")
    print(f"all {len(tests)} tests passed")
