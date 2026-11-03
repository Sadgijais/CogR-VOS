#!/usr/bin/env python3
"""
Amendment 9 items 5-6 (Tasks 2 and 5): one table of every variant against a reference result folder, plus a cost table. CPU only, reads only.

Table 1 (statistics): per variant, on the SAME sentences as the reference: mean J&F, difference, 95% video-level bootstrap interval (same method and
seed as tools/compare_pair.py), videos better / same / worse (per-video mean difference, |diff| < --same counts as same), sentences better / same /
worse, and the number of videos where the variant differs from the reference at all (V3 / V4-vlm may differ in one video only).
Table 2 (cost): from timing_per_video.csv of each folder: videos, live videos (column n_live, when it exists), inference_s, search_s, vlm_wait_s
(kept SEPARATE, never added), overhead against the reference on the same videos (and on the live videos only), whether the overhead is interpreted
(Amendment 8: not under 15%), peak allocated MB, and the log rows of vlm_log.csv / reid_log.csv when they exist. --extra name=seconds adds a
selection-time column (A1 / A3 selection is not part of the V0 tracking time).

  python -I tools/ladder_table.py --run ~/cogrvos/task6_mevis/baseline_v0 --ref results --variants results_a1_clip results_a3_n5 ...
         --out ~/cogrvos/analysis/ladder_mevis --label "MeViS valid_u subset (12 videos) - not the full benchmark"
Writes <out>_stat.csv and <out>_cost.csv (must not exist).
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np

DEFAULT = ["results_a1_clip", "results_a3_n5", "results_a3_n10", "results_largest", "results_v1_final", "results_v2_passive",
           "results_v3_gemini_events_abstain", "results_v4_memory", "results_v4_vlm"]


def load(path):
    return {(r["video"], r["exp_id"]): float(r["JF"]) for r in csv.DictReader(open(path))}


def ci(sd, cnt, reps, seed):
    rng = np.random.default_rng(seed)
    n = len(sd)
    boot = [sd[s].sum() / cnt[s].sum() for s in (rng.integers(0, n, n) for _ in range(reps))]
    return np.percentile(boot, [2.5, 97.5])


def timing(path):
    if not path.exists():
        return None
    return {r["video"]: r for r in csv.DictReader(open(path))}


def fnum(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return None


def log_counts(folder):
    out = {"searches": None, "vlm_calls": None}
    p = folder / "vlm_log.csv"
    if p.exists():
        out["vlm_calls"] = sum(1 for _ in csv.DictReader(open(p)))
    p = folder / "reid_log.csv"
    if p.exists():
        rows = list(csv.DictReader(open(p)))
        out["searches"] = len(rows)
        out["vlm_calls"] = sum(1 for r in rows if (r.get("vlm_letter") or "").strip())
    return out


def fmt(x, f="{:.1f}"):
    return "n/a" if x is None else f.format(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref", default="results", help="result folder name of the reference (V0 normal start)")
    ap.add_argument("--variants", nargs="+", default=DEFAULT)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True, help="prefix; writes <out>_stat.csv and <out>_cost.csv, both must not exist")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--same", type=float, default=1.0)
    ap.add_argument("--extra", action="append", default=[], help="name=seconds of selection time not in the tracking time (repeatable)")
    a = ap.parse_args()

    run = Path(a.run).expanduser()
    out = Path(a.out).expanduser()
    fs, fc = Path(str(out) + "_stat.csv"), Path(str(out) + "_cost.csv")
    if fs.exists() or fc.exists():
        sys.exit(f"{fs} or {fc} exists; refusing to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    for e in a.extra:
        k, v = e.split("=")
        extra[k] = float(v)

    R = load(run / a.ref / "per_video_results.csv")
    keys = sorted(R)
    vids = sorted({k[0] for k in keys})
    idx = {v: np.array([i for i, k in enumerate(keys) if k[0] == v]) for v in vids}
    cnt = np.array([len(idx[v]) for v in vids], float)
    Ra = np.array([R[k] for k in keys])

    print(a.label)
    print(f"reference = {a.ref}   sentences {len(keys)}   videos {len(vids)}   (single runs; sentences of one video are not independent)")
    print(f"\n{'variant':36s} {'J&F':>6} {'diff':>7} {'95% interval':>18} {'videos b/s/w':>13} {'sent b/s/w':>12} {'videos differing':>17}")
    srows = [["variant", "mean_JF", "diff_vs_ref", "ci_lo", "ci_hi", "videos_better", "videos_same", "videos_worse",
              "sent_better", "sent_same", "sent_worse", "videos_differing", "videos_total"]]
    shown = []
    print(f"{a.ref:36s} {Ra.mean():>6.2f} {'(ref)':>7}")
    srows.append([a.ref, f"{Ra.mean():.2f}", "", "", "", "", "", "", "", "", "", "", len(vids)])
    for v in a.variants:
        p = run / v / "per_video_results.csv"
        if not p.exists():
            print(f"{v:36s} no per_video_results.csv: skipped")
            continue
        B_ = load(p)
        if set(B_) != set(R):
            print(f"{v:36s} sentence set differs from the reference ({len(B_)} vs {len(R)}): skipped")
            continue
        shown.append(v)
        B = np.array([B_[k] for k in keys])
        D = B - Ra
        sd = np.array([D[idx[x]].sum() for x in vids])
        lo, hi = ci(sd, cnt, a.reps, a.seed)
        pv = np.array([D[idx[x]].mean() for x in vids])
        vb, vs, vw = int((pv >= a.same).sum()), int((abs(pv) < a.same).sum()), int((pv <= -a.same).sum())
        sb, ss, sw = int((D >= a.same).sum()), int((abs(D) < a.same).sum()), int((D <= -a.same).sum())
        vd = sum(1 for x in vids if np.abs(D[idx[x]]).max() >= 0.01)
        print(f"{v:36s} {B.mean():>6.2f} {D.mean():>+7.2f} {f'[{lo:+.2f}, {hi:+.2f}]':>18} {f'{vb}/{vs}/{vw}':>13} {f'{sb}/{ss}/{sw}':>12} {f'{vd} of {len(vids)}':>17}")
        srows.append([v, f"{B.mean():.2f}", f"{D.mean():+.2f}", f"{lo:+.2f}", f"{hi:+.2f}", vb, vs, vw, sb, ss, sw, vd, len(vids)])

    print(f"\nCOST (timing_per_video.csv; search_s and vlm_wait_s are NOT added to inference_s; overhead is against {a.ref} on the same videos;"
          f" under 15% is not interpreted)")
    print(f"{'variant':36s} {'vids':>4} {'live':>4} {'inference_s':>11} {'search_s':>9} {'vlm_wait_s':>10} {'over_all':>9} {'over_live':>10} {'read?':>6} {'peakMB':>7} "
          f"{'select_s':>9} {'searches':>8} {'vlm_calls':>9}")
    rt = timing(run / a.ref / "timing_per_video.csv")
    crows = [["variant", "videos", "live_videos", "inference_s", "search_s", "vlm_wait_s", "overhead_all_pct", "overhead_live_pct", "interpreted",
              "peak_alloc_mb", "selection_s", "searches", "vlm_calls"]]
    for v in [a.ref] + shown:
        t = timing(run / v / "timing_per_video.csv")
        lc = log_counts(run / v)
        if t is None:
            print(f"{v:36s} {'no timing_per_video.csv':>40s} {fmt(extra.get(v)):>9} {fmt(lc['searches'], '{:d}'):>8} {fmt(lc['vlm_calls'], '{:d}'):>9}")
            crows.append([v, "", "", "", "", "", "", "", "", "", fmt(extra.get(v)), lc["searches"], lc["vlm_calls"]])
            continue
        inf = sum(fnum(r, "inference_s") or 0 for r in t.values())
        srch = sum(fnum(r, "search_s") or 0 for r in t.values()) if any("search_s" in r for r in t.values()) else None
        vw_ = sum(fnum(r, "vlm_wait_s") or 0 for r in t.values()) if any("vlm_wait_s" in r for r in t.values()) else None
        peak = max((fnum(r, "peak_alloc_mb") or 0) for r in t.values())
        over_all = over_live = None
        nl = None
        if rt is not None and v != a.ref:
            same = [x for x in t if x in rt]
            rr = sum(fnum(rt[x], "inference_s") or 0 for x in same)
            vv = sum(fnum(t[x], "inference_s") or 0 for x in same)
            over_all = (vv / rr - 1) * 100 if rr else None
        if any("n_live" in r for r in t.values()):
            live = [x for x, r in t.items() if (fnum(r, "n_live") or 0) > 0]
            nl = len(live)
            if rt is not None and v != a.ref and 0 < nl < len(t):
                rr = sum(fnum(rt[x], "inference_s") or 0 for x in live if x in rt)
                vv = sum(fnum(t[x], "inference_s") or 0 for x in live)
                over_live = (vv / rr - 1) * 100 if rr else None
        ref_o = over_live if over_live is not None else over_all
        read = "ref" if v == a.ref else ("n/a" if ref_o is None else ("yes" if abs(ref_o) >= 15 else "no"))
        print(f"{v:36s} {len(t):>4} {fmt(nl, '{:d}'):>4} {inf:>11.1f} {fmt(srch):>9} {fmt(vw_):>10} {fmt(over_all, '{:+.1f}%'):>9} {fmt(over_live, '{:+.1f}%'):>10} "
              f"{read:>6} {peak:>7.0f} {fmt(extra.get(v)):>9} {fmt(lc['searches'], '{:d}'):>8} {fmt(lc['vlm_calls'], '{:d}'):>9}")
        crows.append([v, len(t), "" if nl is None else nl, f"{inf:.1f}", fmt(srch), fmt(vw_), fmt(over_all, "{:+.1f}"), fmt(over_live, "{:+.1f}"), read,
                      f"{peak:.0f}", fmt(extra.get(v)), "" if lc["searches"] is None else lc["searches"], "" if lc["vlm_calls"] is None else lc["vlm_calls"]])
    print("\nNotes: V3 has no timing_per_video.csv (only v3_summary.txt). 'live' = videos with n_live > 0; for those variants the time of the live videos is the"
          " only one that contains VLM work. A1 / A3 selection time (select_s) is separate from the tracking time. Time differences under 15% are not interpreted.")
    with open(fs, "w", newline="") as f:
        csv.writer(f).writerows(srows)
    with open(fc, "w", newline="") as f:
        csv.writer(f).writerows(crows)
    print(f"\nwrote {fs} and {fc}")


if __name__ == "__main__":
    sys.exit(main())
