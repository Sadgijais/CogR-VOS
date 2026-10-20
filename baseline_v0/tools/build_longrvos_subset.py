#!/usr/bin/env python3
"""
Task 6, Long-RVOS valid: build the frozen evaluation subset (READ-ONLY, reads only meta_expressions.json).

Why valid: it is the official Long-RVOS split that we have with public ground-truth masks (112 videos, 1326 sentences).
The videos are very long (mean about 379 frames), so running all sentences would take days on this laptop.

Rules (written down BEFORE any model run, no randomness):
  1. For each video take ONE sentence of each type (static, dynamic, hybrid): among the sentences of that type, the one
     with the smallest sha256(seed_text + video + "/" + sentence_id). Videos that lack any of the 3 types are skipped.
  2. Order the remaining videos by sha256(seed_text + video_name).
  3. Walk down that order and keep each video while the running total of tracked frames
     (3 sentences x (frames - 1) per video) stays <= --budget. Stop at the FIRST video that does not fit.
     Then sort the kept list.
"""
import argparse, hashlib, json, sys
from pathlib import Path

TYPES = ("static", "dynamic", "hybrid")


def h(s): return hashlib.sha256(s.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Long-RVOS folder that contains valid/")
    ap.add_argument("--budget", type=int, default=13500, help="max tracked frames (sentence x (frames-1))")
    ap.add_argument("--seed-text", default="cogrvos-task6-longrvos-v1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    meta = json.load(open(Path(a.root) / "valid" / "meta_expressions.json"))["videos"]
    cand, lacking = {}, 0
    for v, d in meta.items():
        pick = {}
        for t in TYPES:
            sids = [s for s, e in d["expressions"].items() if e["type"] == t]
            if sids:
                pick[t] = min(sids, key=lambda s: h(f"{a.seed_text}{v}/{s}"))
        if len(pick) == 3:
            cand[v] = (len(d["frames"]), pick)
        else:
            lacking += 1
    order = sorted(cand, key=lambda v: h(a.seed_text + v))

    kept, total, stopped_at = [], 0, None
    for v in order:
        nf, pick = cand[v]
        cost = 3 * (nf - 1)
        if total + cost > a.budget:
            stopped_at = (v, cost); break
        kept.append(v); total += cost
    kept_sorted = sorted(kept)
    pairs = [f"{v}:{cand[v][1][t]}" for v in kept_sorted for t in TYPES]
    digest = h("\n".join(pairs))
    out = {
        "dataset": "long_rvos_valid_subset",
        "rule": f"valid videos with all 3 sentence types; per video the hash-smallest sentence of each type "
                f"(static, dynamic, hybrid); videos ordered by sha256('{a.seed_text}'+video); keep while running tracked "
                f"frames <= {a.budget}; stop at first video that does not fit",
        "seed_text": a.seed_text, "budget_tracked_frames": a.budget,
        "videos": kept_sorted, "hash_order": kept,
        "sentences": {v: [cand[v][1][t] for t in TYPES] for v in kept_sorted},
        "sentence_types": {v: {cand[v][1][t]: t for t in TYPES} for v in kept_sorted},
        "n_videos": len(kept), "n_sentences": len(pairs), "tracked_frames": total,
        "frames_per_video": {v: cand[v][0] for v in kept_sorted},
        "valid_totals": {"videos": len(meta), "videos_with_all_3_types": len(cand), "videos_skipped_missing_a_type": lacking},
        "content_sha256": digest,
    }
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"valid: {len(meta)} videos; with all 3 types: {len(cand)}; skipped (missing a type): {lacking}")
    print(f"kept: {len(kept)} videos, {len(pairs)} sentences, {total} tracked frames (budget {a.budget})")
    print(f"frames per kept video: min {min(out['frames_per_video'].values())}, max {max(out['frames_per_video'].values())}")
    if stopped_at: print(f"stopped before {stopped_at[0]} (cost {stopped_at[1]} would exceed the budget)")
    print(f"ESTIMATE ONLY: 4 versions x {total} frames x 1.98 s/frame = {total*1.98/3600:.1f} h (speed from the YouTube-VOS run)")
    print(f"first 3 in hash order (pilot): {kept[:3]}")
    print(f"content_sha256: {digest}")
    print(f"manifest written: {a.out}")


if __name__ == "__main__":
    sys.exit(main())
