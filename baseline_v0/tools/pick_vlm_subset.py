#!/usr/bin/env python3
"""
Task 6 (Ref-YouTube-VOS): pick the VLM subset BEFORE any real Gemini call (CPU only, no API, no GPU).

Rule written in TASK6_PLAN.md section 3: take the slice's videos in the same hash order
(sha256(seed_text + video)), and keep the first K videos, where K is the LARGEST number such that the
planned V3 'events' schedule needs at most --max-calls VLM calls on those K videos.
The planned calls come from the free oracle replay (log_only): events_per_expression.csv.
Accuracy is never looked at.

Writes one JSON file (the subset list + how it was computed). Reads only.
"""
import argparse, csv, hashlib, json, sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="subsets/ref_ytvos_train_slice.json")
    ap.add_argument("--events-csv", required=True, help="results_v3_oracle_events_log_only/events_per_expression.csv")
    ap.add_argument("--max-calls", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = json.load(open(a.manifest))
    order = sorted(m["videos"], key=lambda v: hashlib.sha256((m["seed_text"] + v).encode()).hexdigest())
    calls, sents = defaultdict(int), defaultdict(int)
    csv_bytes = open(a.events_csv, "rb").read()
    for r in csv.DictReader(csv_bytes.decode().splitlines()):
        calls[r["video"]] += int(r["calls"]); sents[r["video"]] += 1
    total_planned = sum(calls[v] for v in order)

    chosen, cum = [], 0
    print(f"{'rank':>4} {'video':12s} {'sentences':>9} {'calls':>5} {'cumulative':>10}")
    for i, v in enumerate(order, 1):
        c = calls.get(v, 0)
        stop = cum + c > a.max_calls
        if i <= 20 or stop:
            print(f"{i:>4} {v:12s} {sents.get(v,0):>9} {c:>5} {cum + c:>10}{'   <- would exceed the cap, stop before this one' if stop else ''}")
        if stop:
            break
        chosen.append(v); cum += c
    out = {
        "dataset": "ref_youtube_vos_train_slice_vlm_subset",
        "rule": f"first K videos of the slice in sha256(seed_text+video) order, K largest with planned V3 events calls <= {a.max_calls}",
        "seed_text": m["seed_text"], "max_calls": a.max_calls,
        "K": len(chosen), "planned_calls_on_subset": cum, "planned_calls_full_slice": total_planned,
        "sentences_on_subset": sum(sents[v] for v in chosen),
        "events_csv": a.events_csv, "events_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "videos": sorted(chosen),
        "hash_order_of_chosen": chosen,
    }
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nK = {len(chosen)} videos, {out['sentences_on_subset']} sentences, planned calls {cum} (cap {a.max_calls}); full slice would need {total_planned}")
    print(f"written: {a.out}")


if __name__ == "__main__":
    sys.exit(main())
