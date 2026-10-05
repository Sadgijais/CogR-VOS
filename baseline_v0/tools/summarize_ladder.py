#!/usr/bin/env python3
"""
One table for the ablation ladder: J, F, J&F, failure modes, time, VRAM, and how many expressions
improved / got worse vs the previous row. Reads files the stages already wrote.

    python tools/summarize_ladder.py \
        --run V0 ../results/v0_baseline ../results/v0_baseline/failure_analysis.csv \
        --run V1 ../results/v1_sttm     ../results/v1_sttm/failure_analysis.csv \
        --run V2 results_v2_gated       failure_analysis_v2_gated.csv

Each --run is  NAME  RESULTS_DIR  FAILURE_ANALYSIS_CSV  (the dir holds summary.csv, per_video_results.csv,
timing_summary.txt). Writes ladder_summary.md next to where you run it.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

MODES = ["ok", "poor_mask_quality", "grounding_failure", "track_lost_no_distractor", "identity_drift"]


def read_kv(p: Path) -> dict:
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=3, action="append", metavar=("NAME", "RESULTS_DIR", "FAILURE_CSV"), required=True)
    ap.add_argument("--out", default="ladder_summary.md")
    a = ap.parse_args()

    rows, prev = [], None
    for name, rdir, fcsv in a.run:
        rdir = Path(rdir)
        summ = {r["metric"]: float(r["value"]) for r in csv.DictReader(open(rdir / "summary.csv"))}
        per = {(r["video"], r["exp_id"]): float(r["JF"]) for r in csv.DictReader(open(rdir / "per_video_results.csv"))}
        modes = Counter(r["failure_mode"] for r in csv.DictReader(open(fcsv)))
        t = read_kv(rdir / "timing_summary.txt")
        row = {"name": name, "J": summ["mean_J"], "F": summ["mean_F"], "JF": summ["mean_JF"],
               "n": int(summ["n_expressions_scored"]),
               "modes": [modes.get(m, 0) for m in MODES],
               "time_s": t.get("total_inference_s", "n/a"), "vram": t.get("max_peak_alloc_mb", "n/a")}
        if prev is not None:
            common = set(per) & set(prev)
            d = [per[k] - prev[k] for k in common]
            row["vs_prev"] = (sum(x > 0.05 for x in d), sum(x < -0.05 for x in d), sum(abs(x) <= 0.05 for x in d))
        rows.append(row)
        prev = per

    L = ["| run | J | F | J&F | " + " / ".join(["ok", "poor mask", "grounding", "track lost", "drift"]) +
         " | time (s) | peak VRAM (MB) | vs previous: better / worse / same |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        vp = "-" if "vs_prev" not in r else " / ".join(map(str, r["vs_prev"]))
        L.append(f"| {r['name']} | {r['J']:.2f} | {r['F']:.2f} | {r['JF']:.2f} | {' / '.join(map(str, r['modes']))} | "
                 f"{r['time_s']} | {r['vram']} | {vp} |")
    text = "\n".join(L) + f"\n\n({rows[0]['n']} expressions, Ref-DAVIS17 val; failure modes counted from stage 4)\n"
    Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
