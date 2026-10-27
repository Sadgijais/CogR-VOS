#!/usr/bin/env python3
"""
Task 6 audit step A1/A2 (CPU only: no GPU, no API, reads only; writes one CSV).

For every sentence of a dataset (converted DAVIS-style layout) it answers, from the CACHED Grounding-DINO boxes of frame 0
and the ground-truth masks:
  1. Is the target visible at frame 0 at all? (V0 always prompts frame 0.)
  2. If visible at frame 0: where does the top-1 box land?
       correct    box IoU with the target's GT box >= 0.5
       partial    0.1 <= IoU < 0.5
       other_obj  IoU < 0.1 with the target and >= 0.5 with ANOTHER annotated object at frame 0
       elsewhere  none of the above
       no_candidate  Grounding-DINO returned no box
  3. If the top-1 is not correct: is the target's box among the cached candidates (IoU >= 0.5)? (a reranker could fix it)

Box IoU uses the tight bounding box of the GT mask. Thresholds are arguments. Nothing here is a model result.

  python -I tools/audit_grounding.py --root ~/data/MeViS-subset --grounding ~/cogrvos/task6_mevis/baseline_v0/grounding \
         --label "MeViS valid_u subset (12 videos, single-target sentences) - not the full benchmark" --out audit_grounding_mevis.csv
"""
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io_utils import read_palette_mask, unique_palette_ids  # noqa: E402


def bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def box_iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root with Annotations/<tier>/ and meta_expressions/meta_expressions.json")
    ap.add_argument("--grounding", required=True, help="folder with <video>.json (cached Grounding-DINO boxes)")
    ap.add_argument("--tier", default="480p")
    ap.add_argument("--label", required=True, help="dataset label printed with the table")
    ap.add_argument("--hit", type=float, default=0.5, help="box IoU counted as the right object")
    ap.add_argument("--partial", type=float, default=0.1, help="box IoU below this counts as not touching the object")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root, gdir = Path(a.root).expanduser(), Path(a.grounding).expanduser()
    meta = json.load(open(root / "meta_expressions" / "meta_expressions.json"))["videos"]
    ann = root / "Annotations" / a.tier

    rows = []
    for video in sorted(meta):
        gfile = gdir / f"{video}.json"
        if not gfile.exists():
            continue                                  # video not part of this run
        grounding = json.load(open(gfile))
        frames = meta[video]["frames"]
        f0 = ann / video / f"{frames[0]}.png"
        ids0 = {i for i in unique_palette_ids(f0) if i not in (0, 255)}
        for exp_id, e in meta[video]["expressions"].items():
            if exp_id not in grounding:
                continue
            oid = int(e["obj_id"])
            tm = np.asarray(read_palette_mask(f0, oid)).astype(bool)
            tb = bbox(tm)
            # first visible position over the frames that exist in the annotation list
            fv = None
            for pos, fr in enumerate(frames):
                if np.asarray(read_palette_mask(ann / video / f"{fr}.png", oid)).astype(bool).any():
                    fv = pos
                    break
            cands = [list(map(float, b[:5])) for b in grounding[exp_id]["boxes"]]
            row = {"video": video, "exp_id": exp_id, "obj_id": oid, "expression": e["exp"].strip(),
                   "n_frames": len(frames), "first_visible_pos": "" if fv is None else fv,
                   "visible_at_f0": int(tb is not None), "n_candidates": len(cands),
                   "top1_score": round(cands[0][4], 3) if cands else "",
                   "top1_iou_target": "", "best_cand_iou_target": "", "best_cand_rank": "",
                   "top1_class": "", "fixable_by_rerank": ""}
            if not cands:
                row["top1_class"] = "no_candidate"
            elif tb is None:
                row["top1_class"] = "target_absent_f0"
            else:
                ious = [box_iou(c[:4], tb) for c in cands]
                row["top1_iou_target"] = round(ious[0], 3)
                row["best_cand_iou_target"] = round(max(ious), 3)
                hit = [i for i, v in enumerate(ious) if v >= a.hit]
                row["best_cand_rank"] = hit[0] if hit else ""
                if ious[0] >= a.hit:
                    cls = "correct"
                elif ious[0] >= a.partial:
                    cls = "partial"
                else:
                    others = []
                    for o in ids0 - {oid}:
                        ob = bbox(np.asarray(read_palette_mask(f0, o)).astype(bool))
                        if ob is not None:
                            others.append(box_iou(cands[0][:4], ob))
                    cls = "other_obj" if others and max(others) >= a.hit else "elsewhere"
                row["top1_class"] = cls
                if cls != "correct":
                    row["fixable_by_rerank"] = int(bool(hit))
            rows.append(row)

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    objs = len({(r["video"], r["obj_id"]) for r in rows})
    cls = Counter(r["top1_class"] for r in rows)
    vis = [r for r in rows if r["visible_at_f0"]]
    wrong = [r for r in vis if r["top1_class"] not in ("correct",)]
    fix = sum(1 for r in wrong if r["fixable_by_rerank"] == 1)
    late = [r for r in rows if r["first_visible_pos"] not in ("", 0)]
    never = [r for r in rows if r["first_visible_pos"] == ""]
    print(f"{a.label}")
    print(f"sentences {n} (objects {objs}); target visible at frame 0: {len(vis)}; target absent at frame 0: {n - len(vis)} "
          f"(first visible later: {len(late)}, never visible: {len(never)})")
    print(f"{'top-1 box class':22s} {'sentences':>9}")
    for k in ("correct", "partial", "other_obj", "elsewhere", "no_candidate", "target_absent_f0"):
        print(f"{k:22s} {cls.get(k, 0):>9}")
    print(f"visible at frame 0 and top-1 not correct: {len(wrong)}; of these the target is among the cached candidates "
          f"(IoU >= {a.hit}): {fix}")
    print(f"mean candidates per sentence: {sum(r['n_candidates'] for r in rows) / max(n, 1):.2f}; thresholds hit {a.hit}, partial {a.partial}")
    print(f"written: {a.out}")


if __name__ == "__main__":
    sys.exit(main())
