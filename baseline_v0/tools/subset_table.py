#!/usr/bin/env python3
"""
Task 6: per-version table on a chosen set of videos (CPU only, reads files, writes nothing but a printout).

--variant NAME=RESULTS_DIR[:TIME_DIR]  reads RESULTS_DIR/per_video_results.csv (one row per sentence:
video, exp_id, ..., n_frames_scored, J, F, JF) and TIME_DIR/timing_per_video.csv (video, n_frames, inference_s, ...);
TIME_DIR defaults to RESULTS_DIR. (V4 re-runs only some sentences, so its time is taken from the V2 folder, plus --extra.)
--extra NAME=SECONDS adds measured overhead seconds to that version's time (V4 search time on the full slice: 11.9).
J, F, J&F = mean over the sentences of the chosen videos. Inference time = sum of inference_s over those videos (+ extra).
Tracked target frames = sum over sentences of (n_frames_scored + 1)  [first frame is the prompt, last is tracked but not scored;
on the full slice this gives 4916, matching the Task 6 numbers]. FPS = tracked target frames / inference seconds.
Without --subset it uses every video (use this once to check that it reproduces the full-slice numbers).
"""
import argparse, csv, json, sys


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", help="JSON with a 'videos' list (subsets/ytvos_vlm_subset.json)")
    ap.add_argument("--variant", action="append", required=True, help="NAME=RESULTS_DIR[:TIME_DIR] (repeatable)")
    ap.add_argument("--extra", action="append", default=[], help="NAME=SECONDS overhead added to the time")
    a = ap.parse_args()

    keep = None
    if a.subset:
        keep = set(json.load(open(a.subset))["videos"])
    extra = {k: float(v) for k, v in (e.split("=", 1) for e in a.extra)}
    print(f"videos: {'all' if keep is None else len(keep)}")
    print(f"{'version':12s} {'sent':>5} {'J':>7} {'F':>7} {'J&F':>7} {'infer_s':>9} {'tracked':>8} {'FPS':>6}")
    for item in a.variant:
        name, spec = item.split("=", 1)
        d, _, td = spec.partition(":")
        td = td or d
        pv = [r for r in load(f"{d}/per_video_results.csv") if keep is None or r["video"] in keep]
        tm = [r for r in load(f"{td}/timing_per_video.csv") if keep is None or r["video"] in keep]
        if not pv or not tm:
            print(f"{name:12s} NO ROWS (check the folder and the subset)"); continue
        if keep is not None:
            miss = keep - {r["video"] for r in pv}
            if miss:
                print(f"{name:12s} WARNING videos missing from results: {sorted(miss)}")
        n = len(pv)
        J = sum(float(r["J"]) for r in pv) / n
        F = sum(float(r["F"]) for r in pv) / n
        JF = sum(float(r["JF"]) for r in pv) / n
        secs = sum(float(r["inference_s"]) for r in tm) + extra.get(name, 0.0)
        tracked = sum(int(r["n_frames_scored"]) + 1 for r in pv)
        print(f"{name:12s} {n:>5} {J:7.2f} {F:7.2f} {JF:7.2f} {secs:9.1f} {tracked:>8} {tracked / secs:6.2f}")
    print("note: first video of a run includes CUDA warm-up; single runs; small sample, differences under about 1 J&F are noise")


if __name__ == "__main__":
    main()
