#!/usr/bin/env python3
"""
Task 6, Long-RVOS valid: copy the frozen subset into the Ref-DAVIS17 480p folder shape, so the frozen V0..V4 code runs UNCHANGED.

Reads  : <src>/valid/JPEGImages/JPEGImages/<video>/<frame>.jpg          (the data has a doubled folder name)
         <src>/valid/Annotations/Annotations/<video>/<obj_id>/<frame>.png   (binary mask of one object; a missing file = object absent)
         <src>/valid/meta_expressions.json
         the subset manifest (subsets/longrvos_valid_subset.json)
Writes : <out>/JPEGImages/480p/<video>/<frame>.jpg
         <out>/Annotations/480p/<video>/<frame>.png   (palette PNG, pixel value = obj_id, 0 = background)
         <out>/meta_expressions/meta_expressions.json (ONLY the sentences listed in the manifest; type kept as a field)
         <out>/conversion_report.json

Protocol (Amendment 2): 480 px on the SHORT side (even width/height), images bilinear, masks nearest neighbour.
All objects found in the video's annotation folder get a label equal to their folder number (1..255), so other objects are
available to the failure analysis. Other objects are painted first and evaluated targets last; the report counts pixels claimed by
two objects (a single label PNG keeps only one), separately for target-vs-target overlap.

Safety: never writes into <src>. Refuses to run if <out> already exists and is not empty.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image


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
    ap.add_argument("--src", required=True, help="Long-RVOS folder (has valid/)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="new folder to create")
    ap.add_argument("--short-side", type=int, default=480)
    ap.add_argument("--first", type=int, default=0, help="only the first N videos in hash order (pilot)")
    a = ap.parse_args()

    src, out = Path(a.src) / "valid", Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"REFUSING: {out} already exists and is not empty")
    man = json.load(open(a.manifest))
    videos = man["hash_order"][: a.first] if a.first else man["videos"]
    meta = json.load(open(src / "meta_expressions.json"))["videos"]
    pal = voc_palette()
    JP, AN = src / "JPEGImages" / "JPEGImages", src / "Annotations" / "Annotations"

    rep = {"videos": 0, "sentences": 0, "images": 0, "masks": 0, "sizes": set(), "overlap_other_other_px": 0,
           "overlap_target_over_other_px": 0, "overlap_target_target_px_LOST": 0, "target_frames_vanished_by_resize": 0,
           "frames_with_no_target_png": 0}
    slim = {}
    for v in videos:
        d = meta[v]; frames = d["frames"]; exps = d["expressions"]
        chosen = man["sentences"][v]
        objs = sorted(int(p.name) for p in (AN / v).iterdir() if p.is_dir() and p.name.isdigit())
        targets = {int(exps[s]["obj_id"]) for s in chosen}
        if not targets <= set(objs):
            sys.exit(f"{v}: target object(s) {sorted(targets - set(objs))} have no annotation folder (folders: {objs})")
        if max(objs) > 255 or min(objs) < 1:
            sys.exit(f"{v}: object numbers must be 1..255, got {objs}")
        (out / "JPEGImages" / "480p" / v).mkdir(parents=True)
        (out / "Annotations" / "480p" / v).mkdir(parents=True)
        order = [x for x in objs if x not in targets] + [x for x in objs if x in targets]
        for f in frames:
            im = Image.open(JP / v / f"{f}.jpg").convert("RGB")
            w, h = im.size
            lab = np.zeros((h, w), np.uint8); isT = np.zeros((h, w), bool)
            for o in order:
                p = AN / v / str(o) / f"{f}.png"
                if not p.exists():
                    continue
                m = np.array(Image.open(p).convert("L")) > 0
                if m.shape != (h, w):
                    sys.exit(f"{v}/{o}/{f}: mask size {m.shape} differs from image size {(h, w)}")
                if not m.any():
                    continue
                hit = (lab > 0) & m
                n = int(hit.sum())
                if o in targets:
                    rep["overlap_target_target_px_LOST"] += int((hit & isT).sum())
                    rep["overlap_target_over_other_px"] += n - int((hit & isT).sum())
                    isT |= m
                else:
                    rep["overlap_other_other_px"] += n
                lab[m] = o
            if not isT.any():
                rep["frames_with_no_target_png"] += 1
            nw, nh = target_size(w, h, a.short_side)
            im.resize((nw, nh), Image.BILINEAR, reducing_gap=None).save(out / "JPEGImages" / "480p" / v / f"{f}.jpg", quality=95)
            mk = Image.fromarray(lab, mode="P"); mk.putpalette(pal)
            mk2 = mk.resize((nw, nh), Image.NEAREST); mk2.putpalette(pal)
            ids_b, ids_a = set(np.unique(lab).tolist()), set(np.unique(np.array(mk2)).tolist())
            if not ids_a <= ids_b:
                sys.exit(f"{v}/{f}: object ids changed by resizing")
            rep["target_frames_vanished_by_resize"] += len(targets & (ids_b - ids_a))
            mk2.save(out / "Annotations" / "480p" / v / f"{f}.png")
            rep["images"] += 1; rep["masks"] += 1; rep["sizes"].add((nw, nh))
        slim[v] = {"frames": frames, "expressions": {s: {"exp": exps[s]["exp"], "obj_id": int(exps[s]["obj_id"]), "type": exps[s]["type"]} for s in chosen}}
        rep["videos"] += 1; rep["sentences"] += len(chosen)
        print(f"  {v}: {len(frames)} frames, {len(chosen)} sentences, objects {objs}, targets {sorted(targets)}", flush=True)

    (out / "meta_expressions").mkdir(parents=True)
    json.dump({"videos": slim}, open(out / "meta_expressions" / "meta_expressions.json", "w"))
    rep["sizes"] = sorted(rep["sizes"]); rep["manifest_sha256"] = man["content_sha256"]
    json.dump(rep, open(out / "conversion_report.json", "w"), indent=2)
    print(json.dumps(rep, indent=2)); print(f"written to: {out}")


if __name__ == "__main__":
    sys.exit(main())
