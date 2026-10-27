#!/usr/bin/env python3
"""
Task 6 candidate analysis (Sir's check 1; CPU only, no GPU, no API, reads only): from audit_grounding_<dataset>.csv
  1. how often is the target box present among the cached Grounding-DINO candidates (pool recall), and at which rank (recall@k)?
  2. when the top-1 box is wrong, how often is the right box still in the pool (a chooser could recover it) and at which rank?
  3. how many sentences have the target ABSENT at frame 0 (no frame-0 chooser can recover them), and how late does it appear?
  4. does the detector score of the top-1 box tell right from wrong (mean, median, AUROC)?
  5. per video: sentences, top-1 correct, target in pool, target absent at frame 0.
A box counts as the target when its IoU with the ground-truth box at frame 0 is >= 0.5 (the --hit value of tools/audit_grounding.py).
Positions of first appearance count in the annotated frame list of the dataset (Long-RVOS: one position = 5 video frames).

  python -I tools/candidate_analysis.py --audit ~/cogrvos/audit/audit_grounding_mevis.csv --label "MeViS valid_u subset (12 videos, single-target sentences) - not the full benchmark"
"""
import argparse, csv, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def auroc(pos, neg):
    if not pos or not neg:
        return None
    s = 0.0
    for p in pos:
        for q in neg:
            s += 1.0 if p > q else (0.5 if p == q else 0.0)
    return s / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(Path(a.audit).expanduser())))
    n = len(rows)
    vis = [r for r in rows if r["visible_at_f0"] == "1"]
    absent = [r for r in rows if r["visible_at_f0"] != "1"]
    later = [r for r in absent if r["first_visible_pos"] != ""]
    never = [r for r in absent if r["first_visible_pos"] == ""]
    rank = lambda r: int(r["best_cand_rank"]) if r["best_cand_rank"] != "" else None

    print(a.label)
    print(f"sentences {n}; target visible at frame 0: {len(vis)}; absent at frame 0: {len(absent)} "
          f"(appears later {len(later)}, never visible {len(never)})")

    print(f"\n[1] candidate pool, sentences with the target visible at frame 0 (n = {len(vis)})")
    print(f"{'k (top-k boxes)':18s} {'target in top-k':>16} {'share':>7}")
    for k in range(1, 6):
        c = sum(1 for r in vis if rank(r) is not None and rank(r) < k)
        print(f"{k:<18d} {c:>16d} {c / max(len(vis), 1):>7.1%}")
    nohit = [r for r in vis if rank(r) is None]
    nocand = [r for r in nohit if r["n_candidates"] == "0"]
    print(f"target not in the pool at all: {len(nohit)} ({len(nohit) / max(len(vis), 1):.1%}); of these the detector returned no box: {len(nocand)}")
    print("pool size (boxes per sentence): " + ", ".join(f"{k} boxes: {v}" for k, v in sorted(Counter(int(r['n_candidates']) for r in rows).items())))

    wrong = [r for r in vis if rank(r) != 0]
    fix = [r for r in wrong if rank(r) is not None]
    print(f"\n[2] top-1 wrong: {len(wrong)}; the target is still in the pool in {len(fix)} of them ({len(fix) / max(len(wrong), 1):.1%})")
    rc = Counter(rank(r) for r in fix)
    print("rank of the target among those (0 = top-1): " + ", ".join(f"rank {k}: {v}" for k, v in sorted(rc.items())))
    cc = Counter(r["top1_class"] for r in vis)
    print("where the wrong top-1 landed: " + ", ".join(f"{k} {cc.get(k, 0)}" for k in ("correct", "partial", "other_obj", "elsewhere", "no_candidate")))

    print(f"\n[3] target absent at frame 0: {len(absent)} of {n} ({len(absent) / max(n, 1):.1%}); no frame-0 chooser can recover these")
    if later:
        pos = sorted(int(r["first_visible_pos"]) for r in later)
        print(f"first appearance position (annotated frames), {len(later)} sentences: min {pos[0]}, median {pos[len(pos) // 2]}, max {pos[-1]}")
        print("appears by position: " + ", ".join(f"<= {k}: {sum(p <= k for p in pos)}" for k in (1, 2, 5, 10, 20, 50)))

    good = [num(r["top1_score"]) for r in vis if rank(r) == 0 and num(r["top1_score"]) is not None]
    bad = [num(r["top1_score"]) for r in vis if rank(r) != 0 and num(r["top1_score"]) is not None]
    print(f"\n[4] detector score of the top-1 box (right vs wrong box)")
    if good and bad:
        au = auroc(good, bad)
        print(f"right: n {len(good)}, mean {np.mean(good):.3f}, median {np.median(good):.3f}; wrong: n {len(bad)}, mean {np.mean(bad):.3f}, "
              f"median {np.median(bad):.3f}; AUROC {au:.3f} (0.5 = the score tells nothing)")
    else:
        print("not enough rows")

    print(f"\n[5] per video")
    print(f"{'video':14s} {'sent':>5} {'visible f0':>10} {'top-1 ok':>9} {'in pool':>8} {'absent f0':>10}")
    byv = defaultdict(list)
    for r in rows:
        byv[r["video"]].append(r)
    for v in sorted(byv):
        rs = byv[v]
        vv = [r for r in rs if r["visible_at_f0"] == "1"]
        print(f"{v:14s} {len(rs):>5d} {len(vv):>10d} {sum(rank(r) == 0 for r in vv):>9d} "
              f"{sum(rank(r) is not None for r in vv):>8d} {len(rs) - len(vv):>10d}")
    print("\nSingle run on a subset. Sentences of one video are not independent. Nothing here is a model result.")


if __name__ == "__main__":
    sys.exit(main())
