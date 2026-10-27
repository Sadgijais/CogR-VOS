#!/usr/bin/env python3
"""
Task 6 audit step B3 (CPU only, no API, reads only): how many Gemini calls does a V3 log hold, and how reliable was the judge?
Reads vlm_log.csv of a V3 run (columns: video, exp_id, frame, event, kind, c, verdict, confidence, reason, latency_s, cached,
gt_present, gt_iou, correct). The true state of each call comes from the ground-truth columns:
  absent  gt_present is False          good  IoU >= 0.5          poor  0.1 <= IoU < 0.5          lost  IoU < 0.1 (target visible)
Reports: rows (calls), cached rows, verdicts, a table true state x verdict, false no_match (judge rejects a good mask),
missed (judge says match on a lost or absent target), and the no_match answers that would blank a mask (confidence >= 0.6).

  python -I tools/audit_vlm_log.py --log ~/cogrvos/task6_mevis/baseline_v0/results_v3_gemini_events_abstain/vlm_log.csv --label "MeViS video 7ca2e40c5af3"
"""
import argparse, csv, sys
from collections import Counter
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
    ap.add_argument("--log", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--abstain-conf", type=float, default=0.6)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(Path(a.log).expanduser())))
    n = len(rows)
    keys = Counter((r["video"], r["exp_id"], r["frame"]) for r in rows)
    verdicts = Counter(r["verdict"] for r in rows)
    kinds = Counter(r["kind"] for r in rows)
    cached = sum(str(r["cached"]).strip().lower() == "true" for r in rows)
    odd = [r for r in rows if r["verdict"] not in ("match", "no_match", "unsure", "not_visible")]
    print(a.label)
    print(f"log: {a.log}")
    print(f"rows (VLM calls recorded): {n}   duplicate (video, sentence, frame) keys: {sum(1 for k, c in keys.items() if c > 1)}")
    print(f"cached rows (answers reused from an earlier attempt): {cached}   live rows: {n - cached}")
    print("kinds: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    print("verdicts: " + ", ".join(f"{k} {v}" for k, v in sorted(verdicts.items())))
    print(f"rows with an unexpected verdict (error or empty): {len(odd)}")

    ident = [r for r in rows if r["kind"] == "identity"]
    states = ["good", "poor", "lost", "absent", "unknown"]
    table = {s: Counter() for s in states}
    for r in ident:
        table[truth(r)][r["verdict"]] += 1
    cols = ["match", "no_match", "unsure"]
    print(f"\nidentity calls {len(ident)}: true state of the tracker mask against the answer")
    print(f"{'true state':12s} {'calls':>6} " + " ".join(f"{c:>9}" for c in cols))
    for s in states:
        t = sum(table[s].values())
        if t:
            print(f"{s:12s} {t:>6} " + " ".join(f"{table[s][c]:>9}" for c in cols))

    good = sum(table["good"].values())
    bad = sum(table["lost"].values()) + sum(table["absent"].values())
    fnm = table["good"]["no_match"]
    missed = table["lost"]["match"] + table["absent"]["match"]
    print(f"\nfalse no_match (good mask, judge says no_match): {fnm} of {good}")
    print(f"missed (target lost or absent, judge says match): {missed} of {bad}")
    hit = table["lost"]["no_match"] + table["absent"]["no_match"]
    print(f"true no_match (target lost or absent, judge says no_match): {hit} of {bad}")

    conf = lambda r: float(r["confidence"]) if str(r["confidence"]).strip() not in ("", "nan", "None") else 0.0
    strong = [r for r in ident if r["verdict"] == "no_match" and conf(r) >= a.abstain_conf]
    harm = [r for r in strong if truth(r) in ("good", "poor")]
    print(f"\nno_match with confidence >= {a.abstain_conf:g} (these would blank the mask): {len(strong)}; of these on a good or poor mask: {len(harm)}")
    cs = sorted(conf(r) for r in rows if r["verdict"] in ("match", "no_match"))
    if cs:
        print(f"confidence of match/no_match answers: min {cs[0]:.2f}, median {cs[len(cs) // 2]:.2f}, max {cs[-1]:.2f}; "
              f"{sum(c >= 0.9 for c in cs)} of {len(cs)} are 0.9 or higher")

    graded = [r for r in rows if r["verdict"] != "unsure" and str(r["correct"]).strip() != ""]
    ok = sum(str(r["correct"]).strip().lower() == "true" for r in graded)
    print(f"\n'correct' column: {ok} of {len(graded)} graded rows (unsure rows are not graded)")
    lat = [float(r["latency_s"]) for r in rows if str(r["cached"]).strip().lower() != "true" and str(r["latency_s"]).strip() != ""]
    if lat:
        print(f"latency of live calls: mean {sum(lat) / len(lat):.2f} s over {len(lat)} calls")
    print("Single run, one model. Counts are small: read them as examples, not rates.")


if __name__ == "__main__":
    sys.exit(main())
