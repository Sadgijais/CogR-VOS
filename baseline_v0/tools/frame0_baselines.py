#!/usr/bin/env python3
"""
Task 6 candidate selection, frame-0-only reference rows (Amendment 6, item 4; CPU only, no GPU, no API, reads only).

For every sentence whose target is visible at frame 0, from the CACHED Grounding-DINO boxes of frame 0 and the ground-truth mask of frame 0:
  A0 top-1            the first cached box (the list is sorted by detector score)
  largest box         the cached box with the largest area (a tie goes to the higher detector score)
  smallest box        the cached box with the smallest area (a tie goes to the higher detector score)
  random pick         EXPECTED value of a uniformly random pick from the pool (no random numbers are drawn)
  pool ceiling        a box with IoU >= --hit exists in the pool (the best any chooser could do)
Reported on the sentences with the target visible at frame 0 (choice accuracy needs a target box): hit rate (IoU >= --hit with the GT box) and mean
IoU of the chosen box. Sentences with no cached box count as a miss with IoU 0. Targets absent at frame 0 are counted but not scored.
Box IoU uses the tight bounding box of the GT mask, as in tools/audit_grounding.py. Nothing here is a model result and no tracker is run.

  python -I tools/frame0_baselines.py --root ~/data/MeViS-subset --grounding ~/cogrvos/task6_mevis/baseline_v0/grounding \
         --label "MeViS valid_u subset (12 videos, single-target sentences) - not the full benchmark" --out frame0_baselines_mevis.csv
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io_utils import read_palette_mask  # noqa: E402


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


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root with Annotations/<tier>/ and meta_expressions/meta_expressions.json")
    ap.add_argument("--grounding", required=True, help="folder with <video>.json (cached Grounding-DINO boxes)")
    ap.add_argument("--tier", default="480p")
    ap.add_argument("--label", required=True)
    ap.add_argument("--hit", type=float, default=0.5, help="box IoU counted as the right object")
    ap.add_argument("--out", required=True, help="CSV with one row per sentence")
    a = ap.parse_args()

    root, gdir = Path(a.root).expanduser(), Path(a.grounding).expanduser()
    meta = json.load(open(root / "meta_expressions" / "meta_expressions.json"))["videos"]
    ann = root / "Annotations" / a.tier

    rows, absent, n_all = [], 0, 0
    for video in sorted(meta):
        gfile = gdir / f"{video}.json"
        if not gfile.exists():
            continue
        grounding = json.load(open(gfile))
        f0 = ann / video / f"{meta[video]['frames'][0]}.png"
        for exp_id, e in meta[video]["expressions"].items():
            if exp_id not in grounding:
                continue
            n_all += 1
            tb = bbox(np.asarray(read_palette_mask(f0, int(e["obj_id"]))).astype(bool))
            if tb is None:
                absent += 1
                continue
            cands = [list(map(float, b[:5])) for b in grounding[exp_id]["boxes"]]
            ious = [box_iou(c[:4], tb) for c in cands]
            hits = [v >= a.hit for v in ious]
            r = {"video": video, "exp_id": exp_id, "n_candidates": len(cands),
                 "pool_has_hit": int(any(hits)), "random_expected_hit": (sum(hits) / len(hits)) if cands else 0.0,
                 "random_expected_iou": (sum(ious) / len(ious)) if cands else 0.0}
            if cands:
                big = max(range(len(cands)), key=lambda i: (area(cands[i]), -i))
                small = min(range(len(cands)), key=lambda i: (area(cands[i]), i))
                pick = {"a0_top1": 0, "largest": big, "smallest": small}
                for k, i in pick.items():
                    r[k + "_hit"], r[k + "_iou"] = int(hits[i]), round(ious[i], 4)
            else:
                for k in ("a0_top1", "largest", "smallest"):
                    r[k + "_hit"], r[k + "_iou"] = 0, 0.0
            rows.append(r)

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    print(a.label)
    print(f"sentences {n_all}; target visible at frame 0 (scored): {n}; absent at frame 0 (not scored): {absent}; hit = box IoU >= {a.hit} with the GT box")
    print(f"{'rule':34s} {'hits':>6} {'of':>5} {'choice accuracy':>16} {'mean IoU':>9}")
    for name, hk, ik in [("A0 top-1 (best detector score)", "a0_top1_hit", "a0_top1_iou"), ("largest box", "largest_hit", "largest_iou"),
                         ("smallest box", "smallest_hit", "smallest_iou")]:
        h = sum(int(r[hk]) for r in rows)
        print(f"{name:34s} {h:>6d} {n:>5d} {h / n:>16.1%} {np.mean([float(r[ik]) for r in rows]):>9.3f}")
    eh = sum(r["random_expected_hit"] for r in rows)
    print(f"{'random pick (expected value)':34s} {eh:>6.1f} {n:>5d} {eh / n:>16.1%} {np.mean([r['random_expected_iou'] for r in rows]):>9.3f}")
    ph = sum(int(r["pool_has_hit"]) for r in rows)
    print(f"{'pool ceiling (a hit is in the pool)':34s} {ph:>6d} {n:>5d} {ph / n:>16.1%} {'':>9s}")
    print("\nPer video, choice accuracy of A0 / largest / smallest / random (expected):")
    vids = sorted({r["video"] for r in rows})
    for v in vids:
        rs = [r for r in rows if r["video"] == v]
        m = len(rs)
        print(f"  {v:14s} n {m:>3d}  {sum(int(r['a0_top1_hit']) for r in rs) / m:6.1%} {sum(int(r['largest_hit']) for r in rs) / m:6.1%} "
              f"{sum(int(r['smallest_hit']) for r in rs) / m:6.1%} {sum(r['random_expected_hit'] for r in rs) / m:6.1%}")
    print("\nSingle run on a subset. Sentences of one video are not independent. Nothing here is a model result; no tracker was run.")
    print(f"written: {a.out}")


if __name__ == "__main__":
    sys.exit(main())
