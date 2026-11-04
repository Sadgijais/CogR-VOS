#!/usr/bin/env python3
"""Task 3 extension (Sir, 4 Nov): three views of the all-runs files. CPU only; reads CSV files only (no masks, no model, no Gemini).

Inputs: the runs CSV and its .sentences.csv written by tools/failure_runs_all.py (use the --others all files), and the joined CSV written by
tools/failure_crosstab.py (it carries the old n_drift_events per sentence and arm).
  V1  run category x switch, first runs and later runs separately, per arm and group. switch = another object IoU >= 0.5 in at least one frame of
      the run; switch>=min = in at least --min-run frames of the run.
  V2  old drift label versus the new switch flag, per sentence: old = n_drift_events > 0 (target IoU < 0.10 and another object IoU > 0.50 in some frame),
      new = a failure run with a switch. 2 x 2 per arm and group. The two rules differ in threshold and in length, so they are not expected to match exactly.
  V3  concentration: videos, later runs and switch sentences per video (arm given by --detail-arm), and the largest video's share of later runs per arm.
Descriptive only: single run, subsets, sentences of one video are not independent.

  python -I tools/failure_runs_views.py --runs ~/cogrvos/analysis/failure_runs_all_allobj_mevis.csv --old ~/cogrvos/analysis/crosstab_old_new_mevis.csv \
         --label "MeViS valid_u subset (12 videos, single-target sentences) - not the full benchmark" --out-txt ~/cogrvos/analysis/failure_runs_views_mevis.txt
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ARMS = ["A0", "A1", "A3-5", "A3-10", "largest"]
CATS = [("reid_failure", "reid"), ("initial_grounding_failure", "init"), ("identity_drift", "drift"), ("mask_propagation_failure", "prop")]
GROUPS = ("visible_f0", "absent_f0")


def read(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--sentences")
    ap.add_argument("--old", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--min-run", type=int, default=3)
    ap.add_argument("--detail-arm", default="A0")
    ap.add_argument("--out-txt")
    a = ap.parse_args()
    outp = Path(a.out_txt).expanduser() if a.out_txt else None
    if outp and outp.exists():
        sys.exit(f"refusing to overwrite: {outp}")
    runs = read(a.runs)
    sent = read(a.sentences or a.runs + ".sentences.csv")
    old = {(r["arm"], r["video"], r["exp_id"]): num(r["n_drift_events"]) for r in read(a.old)}
    L = []

    def emit(s=""):
        print(s)
        L.append(s)

    emit(a.label)
    emit(f"runs {a.runs}; sentences {len(sent)}; old-label rows {len(old)}; min run for switch>=min: {a.min_run}. Single run on a subset; descriptive.")
    for g in GROUPS:
        emit(f"\n=== V1. group {g}: run category x switch ===")
        emit(f"   {'arm':8s} {'which':>6} {'category':>9} {'runs':>5} {'switch':>7} {'switch>=min':>12}")
        for arm in ARMS:
            for which in ("first", "later"):
                sel = [r for r in runs if r["arm"] == arm and r["group"] == g and ((num(r["run_idx"]) == 1) == (which == "first"))]
                for cat, short in CATS:
                    s2 = [r for r in sel if r["category"] == cat]
                    if s2:
                        emit(f"   {arm:8s} {which:>6} {short:>9} {len(s2):>5} {sum(num(r['switch_in_run']) for r in s2):>7} "
                             f"{sum(1 for r in s2 if num(r['n_other_frames']) >= a.min_run):>12}")
    for g in GROUPS:
        emit(f"\n=== V2. group {g}: old drift label (n_drift_events > 0) vs new switch (a failure run with another object >= 0.5) ===")
        emit(f"   {'arm':8s} {'n':>4} {'both':>5} {'old only':>9} {'new only':>9} {'neither':>8} {'old total':>10} {'new total':>10} {'no old row':>11}")
        for arm in ARMS:
            ss = [r for r in sent if r["arm"] == arm and r["group"] == g]
            c, miss = Counter(), 0
            for r in ss:
                k = (arm, r["video"], r["exp_id"])
                if k not in old:
                    miss += 1
                    continue
                c[(old[k] > 0, num(r["n_runs_with_switch"]) > 0)] += 1
            emit(f"   {arm:8s} {len(ss):>4} {c[(True, True)]:>5} {c[(True, False)]:>9} {c[(False, True)]:>9} {c[(False, False)]:>8} "
                 f"{c[(True, True)] + c[(True, False)]:>10} {c[(True, True)] + c[(False, True)]:>10} {miss:>11}")
    emit("\n=== V3. concentration (group visible_f0) ===")
    emit(f"   {'arm':8s} {'videos':>6} {'videos with later run':>22} {'later runs':>10} {'largest video':>14} {'its share':>9} {'sentences with switch':>22} {'largest video':>14} {'its share':>9}")
    for arm in ARMS:
        lr = Counter(r["video"] for r in runs if r["arm"] == arm and r["group"] == "visible_f0" and num(r["run_idx"]) > 1)
        sw = Counter(r["video"] for r in sent if r["arm"] == arm and r["group"] == "visible_f0" and num(r["n_runs_with_switch"]) > 0)
        nv = len({r["video"] for r in sent if r["arm"] == arm and r["group"] == "visible_f0"})
        t1, t2 = sum(lr.values()), sum(sw.values())
        v1, n1 = lr.most_common(1)[0] if lr else ("-", 0)
        v2, n2 = sw.most_common(1)[0] if sw else ("-", 0)
        emit(f"   {arm:8s} {nv:>6} {len(lr):>22} {t1:>10} {v1:>14} {(n1 / t1 if t1 else 0):>9.2f} {t2:>22} {v2:>14} {(n2 / t2 if t2 else 0):>9.2f}")
    emit(f"\n=== V3b. per video, arm {a.detail_arm}, group visible_f0 ===")
    emit(f"   {'video':16s} {'sentences':>9} {'with>=1 run':>11} {'runs':>5} {'later runs':>10} {'switch sentences':>17} {'old drift sentences':>20}")
    vids = sorted({r["video"] for r in sent if r["arm"] == a.detail_arm and r["group"] == "visible_f0"})
    for v in vids:
        ss = [r for r in sent if r["arm"] == a.detail_arm and r["group"] == "visible_f0" and r["video"] == v]
        emit(f"   {v:16s} {len(ss):>9} {sum(1 for r in ss if num(r['n_runs']) >= 1):>11} {sum(num(r['n_runs']) for r in ss):>5} "
             f"{sum(max(num(r['n_runs']) - 1, 0) for r in ss):>10} {sum(1 for r in ss if num(r['n_runs_with_switch']) > 0):>17} "
             f"{sum(1 for r in ss if old.get((a.detail_arm, r['video'], r['exp_id']), 0) > 0):>20}")
    if outp:
        outp.write_text("\n".join(L) + "\n")
        print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
