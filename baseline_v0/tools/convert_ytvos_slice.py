#!/usr/bin/env python3
"""
Task 6, Ref-YouTube-VOS: copy the frozen slice into the Ref-DAVIS17 480p folder shape,
so the frozen V0..V4 code runs on it UNCHANGED.

Reads  : <src>/train/JPEGImages/<video>/<frame>.jpg
         <src>/train/Annotations/<video>/<frame>.png   (palette PNG, pixel value = object id)
         <src>/meta_expressions/train/meta_expressions.json
         the slice manifest (list of videos)
Writes : <out>/JPEGImages/480p/<video>/<frame>.jpg
         <out>/Annotations/480p/<video>/<frame>.png
         <out>/meta_expressions/meta_expressions.json

Protocol choice (stated in the Task 6 plan): frames and masks are resized to 480 px height
(1280x720 -> 854x480, the same size Ref-DAVIS17 480p uses) so J, F and the 4 px boundary
tolerance are computed exactly as for Ref-DAVIS17. Images: bilinear/antialias. Masks:
nearest neighbour (object ids are never blended). The folder is called "480p" because the
loader expects that name.

Safety: never writes into <src>. Refuses to run if <out> already exists and is not empty.
"""
import argparse, json, shutil, sys
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Ref-YouTube-VOS folder")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="new folder to create")
    ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"REFUSING: {out} already exists and is not empty")
    videos = json.load(open(a.manifest))["videos"]
    meta = json.load(open(src / "meta_expressions" / "train" / "meta_expressions.json"))["videos"]

    n_img = n_msk = 0
    sizes = set()
    for v in videos:
        frames = meta[v]["frames"]
        (out / "JPEGImages" / "480p" / v).mkdir(parents=True)
        (out / "Annotations" / "480p" / v).mkdir(parents=True)
        for f in frames:
            im = Image.open(src / "train" / "JPEGImages" / v / f"{f}.jpg").convert("RGB")
            w, h = im.size
            new_w = int(round(w * a.height / h / 2)) * 2
            new_size = (new_w, a.height)
            im.resize(new_size, Image.BILINEAR, reducing_gap=None).save(
                out / "JPEGImages" / "480p" / v / f"{f}.jpg", quality=95)
            mk = Image.open(src / "train" / "Annotations" / v / f"{f}.png")
            if mk.size != (w, h):
                sys.exit(f"{v}/{f}: mask size {mk.size} differs from image size {(w, h)}")
            pal = mk.getpalette()
            ids_before = set(np.unique(np.array(mk)).tolist())
            mk2 = mk.resize(new_size, Image.NEAREST)
            if pal is not None:
                mk2.putpalette(pal)
            ids_after = set(np.unique(np.array(mk2)).tolist())
            if not ids_after <= ids_before:
                sys.exit(f"{v}/{f}: object ids changed by resizing: {ids_before} -> {ids_after}")
            mk2.save(out / "Annotations" / "480p" / v / f"{f}.png")
            n_img += 1; n_msk += 1; sizes.add(new_size)

    (out / "meta_expressions").mkdir(parents=True)
    slim = {"videos": {v: {"expressions": meta[v]["expressions"], "frames": meta[v]["frames"]} for v in videos}}
    json.dump(slim, open(out / "meta_expressions" / "meta_expressions.json", "w"))
    n_sent = sum(len(d["expressions"]) for d in slim["videos"].values())
    print(f"videos: {len(videos)}  sentences: {n_sent}  images: {n_img}  masks: {n_msk}")
    print(f"output frame size(s): {sorted(sizes)}")
    print(f"written to: {out}")


if __name__ == "__main__": sys.exit(main())
