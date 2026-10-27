#!/usr/bin/env python3
"""
Task 6 audit step B4 (CPU only, no API, reads only): does the judge say "match" because the frame-0 reference crop is already a wrong object?
The V3 judge compares the tracker's frame-0 crop (image 1) with the crop of the current mask (image 2). The frame-0 crop is the top-1 detector box,
so it is the right object only when top1_class == correct in audit_grounding_<dataset>.csv. This joins the two files on (video, exp_id).

  python -I tools/audit_judge_reference.py --vlm-log ~/cogrvos/task6_mevis/baseline_v0/results_v3_gemini_events_abstain/vlm_log.csv --audit ~/cogrvos/audit/audit_grounding_mevis.csv --label "MeViS video 7ca2e40c5af3"
"""
import argparse, csv, sys
from collections import Counter, defaultdict
from pathlib import Path


def truth(r):
    if str(r["gt_present"]).strip().lower() != "true":
        return "absent"
    try:
        v = float(r["gt_iou"])
    except ValueError:
        return "unknown"
    return "good" if v >= 0.5 else ("poor" if v >= 0.1 else "lost")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm-log", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    log = list(csv.DictReader(open(Path(a.vlm_log).expanduser())))
    aud = {(r["video"], r["exp_id"]): r for r in csv.DictReader(open(Path(a.audit).expanduser()))}
    ident = [r for r in log if r["kind"] == "identity"]
    miss = [r for r in ident if (r["video"], r["exp_id"]) not in aud]
    ident = [r for r in ident if (r["video"], r["exp_id"]) in aud]
    print(a.label)
    print(f"identity calls {len(ident) + len(miss)}; joined with the audit file: {len(ident)}; no audit row: {len(miss)}")

    cols = ["match", "no_match", "unsure"]
    grp = defaultdict(Counter)
    for r in ident:
        c = aud[(r["video"], r["exp_id"])]["top1_class"]
        grp["reference RIGHT (top1 correct)" if c == "correct" else "reference WRONG (" + c + ")"][r["verdict"]] += 1
    print("\n[1] answer by quality of the frame-0 reference crop")
    print(f"{'reference':44s} {'calls':>6} " + " ".join(f"{c:>9}" for c in cols))
    for k in sorted(grp):
        print(f"{k:44s} {sum(grp[k].values()):>6} " + " ".join(f"{grp[k][c]:>9}" for c in cols))

    lost = [r for r in ident if truth(r) in ("lost", "absent")]
    print(f"\n[2] calls where the target is lost or absent in the current frame: {len(lost)}")
    t = defaultdict(Counter)
    for r in lost:
        right = aud[(r["video"], r["exp_id"])]["top1_class"] == "correct"
        t["reference right" if right else "reference wrong"][r["verdict"]] += 1
    for k in sorted(t):
        print(f"  {k:18s} calls {sum(t[k].values()):>3}  " + "  ".join(f"{c} {t[k][c]}" for c in cols))

    print("\n[3] per sentence")
    print(f"{'video':14s} {'exp':>4} {'top1 class':12s} {'calls':>5} {'match':>6} {'no_match':>9} {'good mask':>10}")
    by = defaultdict(list)
    for r in ident:
        by[(r["video"], r["exp_id"])].append(r)
    for k in sorted(by, key=lambda x: (x[0], int(x[1]))):
        rs = by[k]
        print(f"{k[0]:14s} {k[1]:>4} {aud[k]['top1_class']:12s} {len(rs):>5} {sum(r['verdict'] == 'match' for r in rs):>6} "
              f"{sum(r['verdict'] == 'no_match' for r in rs):>9} {sum(truth(r) == 'good' for r in rs):>10}")
    print("\nSingle run, one video, one model. Sentences of one video are not independent. Read as examples, not rates.")


if __name__ == "__main__":
    sys.exit(main())
