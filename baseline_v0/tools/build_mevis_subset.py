#!/usr/bin/env python3
"""
Task 6, MeViS (v2) valid_u: build the frozen evaluation subset (READ-ONLY, reads only meta_expressions.json).

Why valid_u: it is the official MeViS split with public ground-truth masks (50 videos). The MeViS `valid` split has no
public masks, so J and F cannot be scored on it.

Rules (written down BEFORE any model run, no randomness):
  1. Only SINGLE-TARGET sentences are run (len(obj_id) == 1). The method follows one frame-0 box, so sentences with no
     target (obj_id empty) or several targets (len(obj_id) > 1) are NOT RUN. They are counted and reported, never hidden.
  2. Candidates = valid_u videos with at least one single-target sentence.
  3. Order candidates by sha256(seed_text + video_name).
  4. Walk down that order and keep each video while the running total of tracked frames
     (sum over its single-target sentences of (frames - 1)) stays <= --budget. Stop at the FIRST video that does not
     fit (no skipping, no choosing). Then sort the kept list.
The manifest lists the exact (video, sentence id) pairs and a content hash.
"""
import argparse, hashlib, json, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="MeVis folder that contains valid_u/")
    ap.add_argument("--budget", type=int, default=12000, help="max tracked frames (sentence x (frames-1))")
    ap.add_argument("--seed-text", default="cogrvos-task6-mevis-v1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    meta = json.load(open(Path(a.root) / "valid_u" / "meta_expressions.json"))["videos"]
    n_all = n_none = n_multi = n_single = 0
    cand = {}
    for v, d in meta.items():
        nf = len(d["frames"])
        singles = []
        for sid, e in d["expressions"].items():
            n_all += 1
            k = len(e["obj_id"])
            if k == 0: n_none += 1
            elif k > 1: n_multi += 1
            else:
                n_single += 1; singles.append(sid)
        if singles:
            cand[v] = (nf, sorted(singles, key=lambda s: int(s)))
    order = sorted(cand, key=lambda v: hashlib.sha256((a.seed_text + v).encode()).hexdigest())

    kept, total, stopped_at = [], 0, None
    for v in order:
        nf, singles = cand[v]
        cost = len(singles) * (nf - 1)
        if total + cost > a.budget:
            stopped_at = (v, cost); break
        kept.append(v); total += cost
    kept_sorted = sorted(kept)
    pairs = [f"{v}:{s}" for v in kept_sorted for s in cand[v][1]]
    digest = hashlib.sha256("\n".join(pairs).encode()).hexdigest()
    hash_order = [v for v in order if v in set(kept)]          # for the pilot: first 3 in hash order

    # facts about the chosen videos (counted on the meta file only)
    kn = km = 0
    for v in kept:
        for e in meta[v]["expressions"].values():
            k = len(e["obj_id"]); kn += (k == 0); km += (k > 1)
    out = {
        "dataset": "mevis_v2_valid_u_subset",
        "rule": f"valid_u videos with >=1 single-target sentence, ordered by sha256('{a.seed_text}'+video); keep while "
                f"running tracked frames <= {a.budget}; stop at first video that does not fit; single-target sentences only",
        "seed_text": a.seed_text, "budget_tracked_frames": a.budget,
        "videos": kept_sorted, "hash_order": hash_order,
        "sentences": {v: cand[v][1] for v in kept_sorted},
        "n_videos": len(kept), "n_sentences": len(pairs), "tracked_frames": total,
        "not_run_in_chosen_videos": {"no_target": kn, "multi_target": km},
        "valid_u_totals": {"sentences": n_all, "no_target": n_none, "multi_target": n_multi, "single_target": n_single},
        "content_sha256": digest,
    }
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"valid_u: {len(meta)} videos, {n_all} sentences = {n_single} single + {n_multi} multi + {n_none} no-target")
    print(f"candidates (>=1 single-target sentence): {len(cand)}")
    print(f"kept: {len(kept)} videos, {len(pairs)} sentences, {total} tracked frames (budget {a.budget})")
    print(f"in the kept videos NOT RUN: {kn} no-target, {km} multi-target sentences")
    if stopped_at: print(f"stopped before {stopped_at[0]} (cost {stopped_at[1]} would exceed the budget)")
    print(f"ESTIMATE ONLY: 4 versions x {total} frames x 1.98 s/frame = {total*1.98/3600:.1f} h (speed from the YouTube-VOS run)")
    print(f"first 3 in hash order (pilot): {hash_order[:3]}")
    print(f"content_sha256: {digest}")
    print(f"manifest written: {a.out}")


if __name__ == "__main__":
    sys.exit(main())
