#!/usr/bin/env python3
"""
Task 6 diagnostic table (CPU only, reads only): V0 with the normal Grounding-DINO start versus the two ORACLE starts
(gtbox = ground-truth box at frame 0, cand = best cached candidate first), per sentence group, with a video-level bootstrap interval.

Groups come from audit_grounding_<dataset>.csv (tools/audit_grounding.py): where the cached top-1 box landed at frame 0.
Oracle numbers use ground truth: they are diagnostic CEILINGS, not results of the method. Sentences whose target is absent at
frame 0 are unchanged by both oracles and are reported as their own group.

  python -I tools/compare_oracle.py --run ~/cogrvos/task6_mevis/baseline_v0 --audit ~/cogrvos/audit/audit_grounding_mevis.csv \
         --label "MeViS valid_u subset (12 videos, single-target sentences) - not the full benchmark"
"""
import argparse, csv, sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out[(r["video"], r["exp_id"])] = float(r["JF"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder with results/, results_oracle_gtbox/, results_oracle_cand/")
    ap.add_argument("--audit", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    run = Path(a.run).expanduser()
    v0 = load(run / "results" / "per_video_results.csv")
    gt = load(run / "results_oracle_gtbox" / "per_video_results.csv")
    cd = load(run / "results_oracle_cand" / "per_video_results.csv")
    grp = {}
    for r in csv.DictReader(open(Path(a.audit).expanduser())):
        grp[(r["video"], r["exp_id"])] = r["top1_class"]
    keys = sorted(set(v0) & set(gt) & set(cd) & set(grp))
    if not (len(keys) == len(v0) == len(gt) == len(cd) == len(grp)):
        sys.exit(f"sentence sets differ: v0 {len(v0)} gtbox {len(gt)} cand {len(cd)} audit {len(grp)} common {len(keys)}")

    A = np.array([v0[k] for k in keys]); B = np.array([gt[k] for k in keys]); C = np.array([cd[k] for k in keys])
    G = np.array([grp[k] for k in keys]); V = np.array([k[0] for k in keys])

    print(a.label)
    print(f"sentences {len(keys)}; mean J&F recomputed from the per-sentence files: V0 {A.mean():.2f}, gtbox {B.mean():.2f}, cand {C.mean():.2f}"
          f" (must equal the summary.csv values)")
    print(f"\n{'group (cached top-1 box at frame 0)':40s} {'n':>4} {'V0':>7} {'gtbox':>7} {'cand':>7}")
    order = [("all sentences", np.ones(len(keys), bool)),
             ("top-1 correct", G == "correct"),
             ("top-1 partial overlap", G == "partial"),
             ("top-1 on another annotated object", G == "other_obj"),
             ("top-1 elsewhere", G == "elsewhere"),
             ("no candidate", G == "no_candidate"),
             ("target absent at frame 0 (unchanged)", G == "target_absent_f0")]
    for name, m in order:
        if m.sum():
            print(f"{name:40s} {int(m.sum()):>4} {A[m].mean():>7.2f} {B[m].mean():>7.2f} {C[m].mean():>7.2f}")

    vids = sorted(set(V))
    idx = {v: np.where(V == v)[0] for v in vids}
    cnt = np.array([len(idx[v]) for v in vids], float)
    sums = {n: np.array([x[idx[v]].sum() for v in vids]) for n, x in (("a", A), ("b", B), ("c", C))}
    rng = np.random.default_rng(a.seed)
    res = {"gtbox - V0": [], "cand - V0": [], "gtbox - cand": []}
    for _ in range(a.reps):
        s = rng.integers(0, len(vids), len(vids))
        n = cnt[s].sum()
        ma, mb, mc = sums["a"][s].sum() / n, sums["b"][s].sum() / n, sums["c"][s].sum() / n
        res["gtbox - V0"].append(mb - ma); res["cand - V0"].append(mc - ma); res["gtbox - cand"].append(mb - mc)
    print(f"\nJ&F difference over all sentences, 95% interval from {a.reps} resamples of the {len(vids)} videos (seed {a.seed}):")
    for k, d in res.items():
        lo, hi = np.percentile(d, [2.5, 97.5])
        pt = {"gtbox - V0": B.mean() - A.mean(), "cand - V0": C.mean() - A.mean(), "gtbox - cand": B.mean() - C.mean()}[k]
        print(f"  {k:14s} {pt:+7.2f}   [{lo:+.2f}, {hi:+.2f}]")
    print("Single run. The interval covers video sampling only (not repeated runs); sentences of one video are resampled together.")


if __name__ == "__main__":
    sys.exit(main())
