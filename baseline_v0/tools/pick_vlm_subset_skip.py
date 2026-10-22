#!/usr/bin/env python3
"""
Task 6 (MeViS, Long-RVOS): pick the VLM subset BEFORE any real Gemini call (CPU only, no API, no GPU).

Rule written in TASK6_PLAN.md Amendment 5 (it replaces the stop-at-first-misfit rule of Amendment 2):
take the dataset's videos in the same hash order (sha256(seed_text + video)); go through them one by one;
add a video if the planned V3 'events' calls on the chosen videos stay at or below --max-calls, otherwise SKIP that
video and go on to the next one. The planned calls come from the free oracle replay (log_only):
events_per_expression.csv. Accuracy is never looked at.

Writes one JSON file (the subset list + how it was computed). Reads only.
"""
import argparse, csv, hashlib, json, sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="subsets/<dataset>_subset.json (has videos and seed_text)")
    ap.add_argument("--events-csv", required=True, help="results_v3_oracle_events_log_only/events_per_expression.csv")
    ap.add_argument("--max-calls", type=int, default=40)
    ap.add_argument("--label", required=True, help="value written to the 'dataset' field of the output")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = json.load(open(a.manifest))
    order = sorted(m["videos"], key=lambda v: hashlib.sha256((m["seed_text"] + v).encode()).hexdigest())
    calls, sents = defaultdict(int), defaultdict(int)
    csv_bytes = open(a.events_csv, "rb").read()
    for r in csv.DictReader(csv_bytes.decode().splitlines()):
        calls[r["video"]] += int(r["calls"]); sents[r["video"]] += 1
    total_planned = sum(calls[v] for v in order)

    chosen, skipped, cum = [], [], 0
    print(f"{'rank':>4} {'video':12s} {'sentences':>9} {'calls':>5} {'cumulative':>10}  decision")
    for i, v in enumerate(order, 1):
        c = calls.get(v, 0)
        if cum + c > a.max_calls:
            skipped.append({"video": v, "planned_calls": c})
            print(f"{i:>4} {v:12s} {sents.get(v, 0):>9} {c:>5} {cum + c:>10}  skipped (would exceed {a.max_calls})")
            continue
        chosen.append(v); cum += c
        print(f"{i:>4} {v:12s} {sents.get(v, 0):>9} {c:>5} {cum:>10}  chosen")
    out = {
        "dataset": a.label,
        "rule": f"videos of the dataset in sha256(seed_text+video) order; add a video if the planned V3 events calls on the chosen videos stay <= {a.max_calls}, else skip it and continue (Amendment 5)",
        "seed_text": m["seed_text"], "max_calls": a.max_calls,
        "K": len(chosen), "planned_calls_on_subset": cum, "planned_calls_full_slice": total_planned,
        "sentences_on_subset": sum(sents[v] for v in chosen),
        "events_csv": a.events_csv, "events_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "videos": sorted(chosen),
        "hash_order_of_chosen": chosen,
        "skipped_in_hash_order": skipped,
    }
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nK = {len(chosen)} videos, {out['sentences_on_subset']} sentences, planned calls {cum} (cap {a.max_calls}); full dataset would need {total_planned}")
    print(f"written: {a.out}")


if __name__ == "__main__":
    sys.exit(main())
