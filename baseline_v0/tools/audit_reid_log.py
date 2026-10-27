#!/usr/bin/env python3
"""
Task 6 audit step B5 (CPU only, no API, reads only): count the searches, cached answers, errors and restart labels of a V4 reid_log.csv,
and how many restarts replaced an already good tracker mask by a worse one.

  python -I tools/audit_reid_log.py --log ~/cogrvos/task6_mevis/baseline_v0/results_v4_vlm/reid_log.csv --label "MeViS video 7ca2e40c5af3: V4 VLM log"
"""
import argparse, csv, sys
from collections import Counter
from pathlib import Path


def f(x):
    return float(x) if str(x).strip() not in ("", "nan", "None") else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(Path(a.log).expanduser())))
    t = lambda x: str(x).strip().lower() == "true"
    print(a.label)
    print("rows (searches):", len(rows), "| variants:", dict(Counter(r["variant"] for r in rows)))
    print("cached rows:", sum(t(r["vlm_cached"]) for r in rows), "| rows with an error text:", sum(1 for r in rows if r["vlm_error"].strip()))
    print("actions:", dict(Counter(r["action"] for r in rows)))
    print("labels:", dict(sorted(Counter(r["label"] for r in rows).items())))
    conf = [f(r["vlm_conf"]) for r in rows if f(r["vlm_conf"]) is not None]
    if conf:
        print("vlm_conf: min %.2f, max %.2f, %d of %d are 0.9 or higher" % (min(conf), max(conf), sum(c >= 0.9 for c in conf), len(conf)))
    rs = [r for r in rows if r["action"] == "restart"]
    good = [r for r in rs if f(r["current_iou"]) is not None and f(r["current_iou"]) >= 0.5]
    worse = [r for r in good if (f(r["chosen_iou"]) or 0.0) < f(r["current_iou"])]
    low = [r for r in good if (f(r["chosen_iou"]) or 0.0) < 0.5]
    print("restarts: %d; on an already good tracker mask (IoU >= 0.5): %d; of these the new mask is worse: %d, below 0.5: %d"
          % (len(rs), len(good), len(worse), len(low)))
    print("Single run. Counts are small.")


if __name__ == "__main__":
    sys.exit(main())
