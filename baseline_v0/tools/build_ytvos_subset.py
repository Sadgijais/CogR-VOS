#!/usr/bin/env python3
"""
Task 6, Ref-YouTube-VOS: build the frozen evaluation slice (READ-ONLY on the dataset).

Why a slice of `train`: the official Ref-YouTube-VOS `valid` split has no public
ground-truth masks (the `valid/Annotations` folder on this laptop is in the
predictions layout <video>/<sentence_id>/<frame>.png), so J&F cannot be scored on it.
`train` has masks, so we score on a fixed slice of `train` and label it that way.

Rule (written down BEFORE any model run):
  1. candidates = train videos with at least --min-frames annotated frames
  2. order candidates by sha256(seed_text + video_name)   (deterministic, no RNG)
  3. take the first --n of them
The chosen video list is hashed (sha256 over the sorted names joined by newlines).

Writes only the manifest file you name with --out. Never writes into the dataset.
Also prints facts about the slice (sizes, objects absent in frame 0, disappearances).
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Ref-YouTube-VOS folder (has train/ and meta_expressions/)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--min-frames", type=int, default=10)
    ap.add_argument("--seed-text", default="cogrvos-task6-ytvos-v1")
    ap.add_argument("--out", required=True, help="manifest json to write")
    a = ap.parse_args()

    root = Path(a.root)
    meta = json.load(open(root / "meta_expressions" / "train" / "meta_expressions.json"))["videos"]

    cands = [v for v, d in meta.items() if len(d["frames"]) >= a.min_frames]
    cands.sort(key=lambda v: hashlib.sha256((a.seed_text + v).encode()).hexdigest())
    chosen = sorted(cands[: a.n])
    digest = hashlib.sha256("\n".join(chosen).encode()).hexdigest()

    n_sent = n_obs = 0
    absent_f0 = []          # (video, sentence_id, obj_id): target not visible in the very first frame
    gone_objs = 0           # objects absent for >= 2 frames and visible again later
    never_vis = []
    objs_total = 0
    for v in chosen:
        d = meta[v]
        frames = d["frames"]
        ids_needed = sorted({int(e["obj_id"]) for e in d["expressions"].values()})
        vis = {i: [] for i in ids_needed}
        for f in frames:
            m = np.array(Image.open(root / "train" / "Annotations" / v / f"{f}.png"))
            for i in ids_needed:
                vis[i].append(bool((m == i).any()))
        objs_total += len(ids_needed)
        for i, row in vis.items():
            if not any(row):
                never_vis.append((v, i))
            # absent run of >= 2 frames followed by a visible frame
            run = 0
            for k, present in enumerate(row):
                if not present:
                    run += 1
                else:
                    if run >= 2 and k - run > 0 and any(row[: k - run]):
                        gone_objs += 1
                        break
                    run = 0
        for sid, e in d["expressions"].items():
            n_sent += 1
            n_obs += max(len(frames) - 2, 0)
            if not vis[int(e["obj_id"])][0]:
                absent_f0.append((v, sid, int(e["obj_id"])))

    out = {
        "videos": chosen,
        "dataset": "ref_youtube_vos_train_slice",
        "rule": f"train videos with >= {a.min_frames} frames, ordered by sha256('{a.seed_text}'+video), first {a.n}, then sorted",
        "seed_text": a.seed_text, "n": a.n, "min_frames": a.min_frames,
        "content_sha256": digest,
    }
    json.dump(out, open(a.out, "w"), indent=2)

    print(f"candidates with >= {a.min_frames} frames: {len(cands)} of {len(meta)} train videos")
    print(f"chosen videos: {len(chosen)}   sentences: {n_sent}   objects: {objs_total}")
    print(f"scored frame-observations (sentence x (frames-2)): {n_obs}")
    print(f"sentences whose target is NOT visible in frame 0: {len(absent_f0)} of {n_sent}")
    print(f"objects that vanish for >= 2 frames and come back: {gone_objs} of {objs_total}")
    print(f"objects never visible in any annotated frame: {len(never_vis)}")
    print(f"content_sha256: {digest}")
    print(f"manifest written: {a.out}")


if __name__ == "__main__":
    sys.exit(main())
