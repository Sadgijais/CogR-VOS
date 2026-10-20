#!/usr/bin/env python3
"""
Task 6, MeViS valid_u: copy the frozen subset into the Ref-DAVIS17 480p folder shape, so the frozen V0..V4 code runs UNCHANGED.

Reads  : <src>/valid_u/JPEGImages/<video>/<frame>.jpg
         <src>/valid_u/mask_dict.json   (COCO RLE: mask_dict[str(anno_id)][frame_position] = {"size": [h, w], "counts": str} or None)
         <src>/valid_u/meta_expressions.json
         the subset manifest (subsets/mevis_valid_u_subset.json)
Writes : <out>/JPEGImages/480p/<video>/<frame>.jpg
         <out>/Annotations/480p/<video>/<frame>.png   (palette PNG, pixel value = object label, 0 = background)
         <out>/meta_expressions/meta_expressions.json (ONLY the single-target sentences listed in the manifest)
         <out>/conversion_report.json

Protocol (Amendment 2): 480 px on the SHORT side (even width/height), images bilinear, masks nearest neighbour.
Every object that appears in any sentence of a video gets a label (1..n, in sorted anno_id order), so other objects are
available to the failure analysis. Masks are painted other-objects first and evaluated targets last; the report counts pixels
claimed by two objects (a single label PNG keeps only one), separately for target-vs-target overlap.
The RLE decoder is pure python/numpy (checked against pycocotools on a sample when pycocotools is installed).

Safety: never writes into <src>. Refuses to run if <out> already exists and is not empty.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from pycocotools import mask as _cm
except Exception:
    _cm = None


def counts_from_string(s):
    cnts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1f) << (5 * k)
            more = bool(c & 0x20)
            p += 1; k += 1
            if (not more) and (c & 0x10):
                x |= (-1) << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def decode_rle(rle):
    """COCO RLE -> bool array (h, w). Column-major runs, first run is background."""
    if rle is None:
        return None
    h, w = rle["size"]
    c = rle["counts"]
    cnts = counts_from_string(c) if isinstance(c, str) else list(c)
    vals = (np.arange(len(cnts)) % 2).astype(np.uint8)
    flat = np.repeat(vals, np.array(cnts, dtype=np.int64))
    if flat.size != h * w:
        raise ValueError(f"RLE length {flat.size} != {h}*{w}")
    return flat.reshape(w, h).T.astype(bool)


def voc_palette():
    pal = []
    for i in range(256):
        r = g = b = 0; c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j); g |= ((c >> 1) & 1) << (7 - j); b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        pal += [r, g, b]
    return pal


def target_size(w, h, short):
    if w <= h:
        return (short, int(round(h * short / w / 2)) * 2)
    return (int(round(w * short / h / 2)) * 2, short)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="MeVis folder (has valid_u/)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="new folder to create")
    ap.add_argument("--short-side", type=int, default=480)
    ap.add_argument("--first", type=int, default=0, help="only the first N videos in hash order (pilot)")
    a = ap.parse_args()

    src, out = Path(a.src) / "valid_u", Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"REFUSING: {out} already exists and is not empty")
    man = json.load(open(a.manifest))
    videos = man["hash_order"][: a.first] if a.first else man["videos"]
    meta = json.load(open(src / "meta_expressions.json"))["videos"]
    md = json.load(open(src / "mask_dict.json"))
    pal = voc_palette()

    rep = {"videos": 0, "sentences": 0, "images": 0, "masks": 0, "sizes": set(), "overlap_other_other_px": 0,
           "overlap_target_over_other_px": 0, "overlap_target_target_px_LOST": 0, "target_frames_vanished_by_resize": 0,
           "pycocotools_checked_masks": 0}
    slim = {}
    for v in videos:
        d = meta[v]; frames = d["frames"]; exps = d["expressions"]
        all_ids = sorted({int(x) for e in exps.values() for x in e["anno_id"]})
        if len(all_ids) > 254:
            sys.exit(f"{v}: too many objects for a palette PNG")
        label = {aid: i + 1 for i, aid in enumerate(all_ids)}
        chosen = manifest_sids = man["sentences"][v]
        targets = {int(exps[s]["anno_id"][0]) for s in chosen}
        for aid in all_ids:
            if str(aid) not in md or len(md[str(aid)]) != len(frames):
                sys.exit(f"{v}: mask_dict entry for anno_id {aid} missing or has {len(md.get(str(aid), []))} items, video has {len(frames)} frames")
        (out / "JPEGImages" / "480p" / v).mkdir(parents=True)
        (out / "Annotations" / "480p" / v).mkdir(parents=True)
        order = [x for x in all_ids if x not in targets] + [x for x in all_ids if x in targets]
        for pos, f in enumerate(frames):
            im = Image.open(src / "JPEGImages" / v / f"{f}.jpg").convert("RGB")
            w, h = im.size
            lab = np.zeros((h, w), np.uint8); isT = np.zeros((h, w), bool)
            for aid in order:
                rle = md[str(aid)][pos]
                m = decode_rle(rle)
                if m is None or not m.any():
                    continue
                if m.shape != (h, w):
                    sys.exit(f"{v}/{f}: mask size {m.shape} differs from image size {(h, w)}")
                if _cm is not None and rep["pycocotools_checked_masks"] < 30 * (rep["videos"] + 1):
                    r2 = dict(rle); r2["counts"] = r2["counts"].encode() if isinstance(r2["counts"], str) else r2["counts"]
                    if not np.array_equal(_cm.decode(r2).astype(bool), m):
                        sys.exit(f"{v}/{f}: pure decoder and pycocotools disagree for anno_id {aid}")
                    rep["pycocotools_checked_masks"] += 1
                hit = (lab > 0) & m
                n = int(hit.sum())
                if aid in targets:
                    rep["overlap_target_target_px_LOST"] += int((hit & isT).sum())
                    rep["overlap_target_over_other_px"] += n - int((hit & isT).sum())
                    isT |= m
                else:
                    rep["overlap_other_other_px"] += n
                lab[m] = label[aid]
            nw, nh = target_size(w, h, a.short_side)
            im.resize((nw, nh), Image.BILINEAR, reducing_gap=None).save(out / "JPEGImages" / "480p" / v / f"{f}.jpg", quality=95)
            mk = Image.fromarray(lab, mode="P"); mk.putpalette(pal)
            mk2 = mk.resize((nw, nh), Image.NEAREST); mk2.putpalette(pal)
            ids_b, ids_a = set(np.unique(lab).tolist()), set(np.unique(np.array(mk2)).tolist())
            if not ids_a <= ids_b:
                sys.exit(f"{v}/{f}: object ids changed by resizing")
            rep["target_frames_vanished_by_resize"] += len({label[t] for t in targets} & (ids_b - ids_a))
            mk2.save(out / "Annotations" / "480p" / v / f"{f}.png")
            rep["images"] += 1; rep["masks"] += 1; rep["sizes"].add((nw, nh))
        slim[v] = {"frames": frames, "expressions": {s: {"exp": exps[s]["exp"], "obj_id": label[int(exps[s]["anno_id"][0])]} for s in chosen}}
        rep["videos"] += 1; rep["sentences"] += len(chosen)
        print(f"  {v}: {len(frames)} frames, {len(chosen)} sentences, labels {label}", flush=True)

    (out / "meta_expressions").mkdir(parents=True)
    json.dump({"videos": slim}, open(out / "meta_expressions" / "meta_expressions.json", "w"))
    rep["sizes"] = sorted(rep["sizes"]); rep["manifest_sha256"] = man["content_sha256"]
    json.dump(rep, open(out / "conversion_report.json", "w"), indent=2)
    print(json.dumps(rep, indent=2)); print(f"written to: {out}")


if __name__ == "__main__":
    sys.exit(main())
