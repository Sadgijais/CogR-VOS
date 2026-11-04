#!/usr/bin/env python3
"""Task 3 verification (Sir, 4 Nov): old sentence-level failure labels versus the new first-failure-run category.

CPU only. Reads existing CSV files, nothing else: the per-sentence CSV written by tools/frame_failure_analysis.py and the
old failure_analysis*.csv files of the run folder. No model, no API, no ground-truth masks. Writes only the files named by
--out-csv / --out-txt.

Question it answers: the new table says "identity drift is the first failure run in 0 sentences". The old label says
identity_drift for some sentences, and n_drift_events counts drift events in ANY frame (target IoU < 0.10 and another
object IoU > 0.50). Where do those sentences sit in the new categories?

Tables per arm and group (visible_f0 / absent_f0):
  1. old failure_mode (rows) x new category of the first failure run (columns)
  2. sentences with at least one old drift event (n_drift_events > 0), by new category and by old failure_mode
"""
import argparse
import csv
from collections import Counter

ARM_FILES = [("A0", "failure_analysis.csv"), ("A1", "failure_analysis_a1_clip.csv"), ("A3-5", "failure_analysis_a3_n5.csv"),
             ("A3-10", "failure_analysis_a3_n10.csv"), ("largest", "failure_analysis_largest.csv")]
OLD = ["grounding_failure", "identity_drift", "track_lost_no_distractor", "poor_mask_quality", "ok"]
NEW = ["no_failure", "reid_failure", "initial_grounding_failure", "identity_drift", "mask_propagation_failure"]
SHORT = {"no_failure": "none", "reid_failure": "reid", "initial_grounding_failure": "init", "identity_drift": "drift",
         "mask_propagation_failure": "prop"}
OUT_FIELDS = ["arm", "video", "exp_id", "group", "old_failure_mode", "n_drift_events", "drift_onsets", "drift_onto_obj",
              "new_category", "onset", "similar_flag", "other_at_onset", "start_hit", "JF_new"]


def to_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def table(rows, rkey, rvals, ckey, cvals, title):
    cnt = Counter((r[rkey], r[ckey]) for r in rows)
    extra_r = sorted({r[rkey] for r in rows} - set(rvals))
    extra_c = sorted({r[ckey] for r in rows} - set(cvals))
    rv, cv = list(rvals) + extra_r, list(cvals) + extra_c
    L = [title, "  %-26s" % (rkey + " \\ " + ckey) + "".join("%8s" % SHORT.get(c, c[:7]) for c in cv) + "%8s" % "total"]
    col = Counter()
    for v in rv:
        n = [cnt[(v, c)] for c in cv]
        if sum(n) == 0:
            continue
        for c, x in zip(cv, n):
            col[c] += x
        L.append("  %-26s" % v + "".join("%8d" % x for x in n) + "%8d" % sum(n))
    L.append("  %-26s" % "total" + "".join("%8d" % col[c] for c in cv) + "%8d" % sum(col.values()))
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-csv", required=True, help="CSV written by tools/frame_failure_analysis.py")
    ap.add_argument("--run", required=True, help="run folder holding failure_analysis*.csv (task6_mevis/baseline_v0)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-csv")
    ap.add_argument("--out-txt")
    a = ap.parse_args()
    new_all = read(a.new_csv)
    L = ["Old sentence labels versus the new first-failure category. %s" % a.label,
         "new CSV: %s" % a.new_csv, "run folder (old labels): %s" % a.run,
         "Single run on a subset; descriptive. Sentences of one video are not independent. n_drift_events counts drift events in ANY frame.",
         "Old identity_drift = at least one drift event, among sentences whose first visible frame IoU is at least 0.10 (old rule order)."]
    joined = []
    for arm, fname in ARM_FILES:
        old = {(r["video"], str(r["exp_id"])): r for r in read("%s/%s" % (a.run.rstrip("/"), fname))}
        new = [r for r in new_all if r["arm"] == arm]
        miss = 0
        rows = []
        for r in new:
            o = old.get((r["video"], str(r["exp_id"])))
            if o is None:
                miss += 1
                continue
            rows.append(dict(arm=arm, video=r["video"], exp_id=r["exp_id"], group=r["group"], old_failure_mode=o["failure_mode"],
                             n_drift_events=to_int(o.get("n_drift_events")), drift_onsets=o.get("drift_onsets", ""),
                             drift_onto_obj=o.get("drift_onto_obj", ""), new_category=r["category"], onset=r["onset"],
                             similar_flag=r["similar_flag"], other_at_onset=r["other_at_onset"], start_hit=r["start_hit"],
                             JF_new=r["JF"]))
        joined += rows
        L += ["", "=" * 78, "ARM %s: %d new rows, %d joined, %d without an old row" % (arm, len(new), len(rows), miss)]
        for g in ("visible_f0", "absent_f0"):
            sub = [r for r in rows if r["group"] == g]
            if not sub:
                continue
            L += ["", "group %s (%d sentences)" % (g, len(sub))]
            L += table(sub, "old_failure_mode", OLD, "new_category", NEW, "  old failure_mode x new category of the first failure run")
            ev = [r for r in sub if r["n_drift_events"] > 0]
            L += ["", "  sentences with at least one old drift event (n_drift_events > 0): %d" % len(ev)]
            if ev:
                L += table(ev, "old_failure_mode", OLD, "new_category", NEW, "  those sentences: old failure_mode x new category")
                same = sum(1 for r in ev if str(r["similar_flag"]) == "1")
                L.append("  of those, with another object at IoU >= 0.5 at the onset of the first failure run (similar_flag): %d" % same)
    text = "\n".join(L)
    print(text)
    if a.out_txt:
        with open(a.out_txt, "w") as f:
            f.write(text + "\n")
    if a.out_csv:
        with open(a.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
            w.writeheader()
            w.writerows(joined)


if __name__ == "__main__":
    main()
