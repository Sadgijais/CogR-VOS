#!/usr/bin/env python3
"""
Paired comparison of two result folders of the SAME sentences (CPU only, reads only): B minus A in J&F, with a video-level bootstrap interval,
sentence win/loss counts, the groups of tools/audit_grounding.py, and the per-video differences (is the gain spread out or in a few videos?).

  python -I tools/compare_pair.py --run ~/cogrvos/task6_longrvos/baseline_v0 --a results_oracle_gtbox --b results_v1_oracle_gtbox \
         --audit ~/cogrvos/audit/audit_grounding_longrvos.csv --label "Long-RVOS valid subset (11 videos, one sentence per type) - not the full benchmark"
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np


def load(path):
    return {(r["video"], r["exp_id"]): float(r["JF"]) for r in csv.DictReader(open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--a", required=True, help="result folder name of A (reference)")
    ap.add_argument("--b", required=True, help="result folder name of B (compared with A)")
    ap.add_argument("--audit", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--same", type=float, default=1.0, help="|difference| below this J&F counts as 'same'")
    a = ap.parse_args()

    run = Path(a.run).expanduser()
    A_, B_ = load(run / a.a / "per_video_results.csv"), load(run / a.b / "per_video_results.csv")
    grp = {(r["video"], r["exp_id"]): r["top1_class"] for r in csv.DictReader(open(Path(a.audit).expanduser()))}
    keys = sorted(set(A_) & set(B_) & set(grp))
    if not (len(keys) == len(A_) == len(B_) == len(grp)):
        sys.exit(f"sentence sets differ: A {len(A_)} B {len(B_)} audit {len(grp)} common {len(keys)}")
    A = np.array([A_[k] for k in keys]); B = np.array([B_[k] for k in keys]); D = B - A
    G = np.array([grp[k] for k in keys]); V = np.array([k[0] for k in keys])

    print(a.label)
    print(f"A = {a.a}   B = {a.b}   sentences {len(keys)}")
    print(f"mean J&F  A {A.mean():.2f}   B {B.mean():.2f}   B - A {D.mean():+.2f}")
    print(f"sentences: B better {int((D >= a.same).sum())}, same (|diff| < {a.same:g}) {int((abs(D) < a.same).sum())}, B worse {int((D <= -a.same).sum())}")

    vids = sorted(set(V))
    idx = {v: np.where(V == v)[0] for v in vids}
    cnt = np.array([len(idx[v]) for v in vids], float)
    sd = np.array([D[idx[v]].sum() for v in vids])
    rng = np.random.default_rng(a.seed)
    boot = [sd[s].sum() / cnt[s].sum() for s in (rng.integers(0, len(vids), len(vids)) for _ in range(a.reps))]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"B - A over all sentences: {D.mean():+.2f}   95% interval [{lo:+.2f}, {hi:+.2f}] from {a.reps} resamples of the {len(vids)} videos (seed {a.seed})")

    print(f"\n{'group (cached top-1 box at frame 0)':40s} {'n':>4} {'A':>7} {'B':>7} {'B-A':>7}")
    for name, m in [("top-1 correct", G == "correct"), ("top-1 partial overlap", G == "partial"),
                    ("top-1 on another annotated object", G == "other_obj"), ("top-1 elsewhere", G == "elsewhere"),
                    ("no candidate", G == "no_candidate"), ("target absent at frame 0", G == "target_absent_f0")]:
        if m.sum():
            print(f"{name:40s} {int(m.sum()):>4} {A[m].mean():>7.2f} {B[m].mean():>7.2f} {D[m].mean():>+7.2f}")

    print(f"\nper video (mean over its sentences):  {'video':14s} {'n':>3} {'A':>7} {'B':>7} {'B-A':>7}")
    rows = sorted(((D[idx[v]].mean(), v) for v in vids), reverse=True)
    for d, v in rows:
        print(f"{'':38s}{v:14s} {len(idx[v]):>3} {A[idx[v]].mean():>7.2f} {B[idx[v]].mean():>7.2f} {d:>+7.2f}")
    tot = D.sum()
    top = np.sort(D)[::-1][:3].sum()
    print(f"\nthe 3 sentences with the largest gain add up to {top:.1f} of the total {tot:.1f} J&F points (sum over sentences)")
    print("Single run. Interval covers video sampling only; sentences of one video are resampled together.")


if __name__ == "__main__":
    sys.exit(main())
