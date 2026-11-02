#!/usr/bin/env python3
"""
Task 6 candidate selection (Amendment 6 items 5 and 7; CPU only, reads only): choice accuracy of a chosen-box grounding folder against the original
cached folder (A0 = top-1), on the sentences whose target is visible at frame 0. Uses the ground-truth mask of frame 0 ONLY for scoring.

  hit = IoU of boxes[0] with the tight bounding box of the ground-truth mask >= --hit (default 0.5); mean IoU of boxes[0]; both for A0 (--ref) and for the
  arm (--grounding). Also: the difference arm minus A0 with a 95% interval from --reps resamples of the VIDEOS (seed --seed; covers the choice of videos
  only), sentences that gained / kept / lost the hit, the groups of tools/audit_grounding.py, and per video. Checks that the arm only re-ordered the
  cached boxes (a chosen box that is not in the pool stops the run). Sentences with the target absent at frame 0 are counted, not scored.

  python -I tools/choice_accuracy.py --root ~/data/MeViS-subset --ref <run>/grounding --grounding <run>/grounding_a1_clip \
         --audit ~/cogrvos/audit/audit_grounding_mevis.csv --label "MeViS valid_u subset (12 videos, single-target sentences) - not the full benchmark: A1 CLIP re-score"
"""
import argparse, csv, json, sys
from collections import defaultdict
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ref", required=True, help="original cached folder (A0 = boxes[0])")
    ap.add_argument("--grounding", required=True, help="folder of the arm (chosen box in boxes[0])")
    ap.add_argument("--audit", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tier", default="480p")
    ap.add_argument("--hit", type=float, default=0.5)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root, ref, arm = Path(a.root).expanduser(), Path(a.ref).expanduser(), Path(a.grounding).expanduser()
    meta = json.load(open(root / "meta_expressions" / "meta_expressions.json"))["videos"]
    grp = {(r["video"], r["exp_id"]): r["top1_class"] for r in csv.DictReader(open(Path(a.audit).expanduser()))}

    rows, absent, absent_changed, n_all = [], 0, 0, 0
    for rfile in sorted(ref.glob("*.json")):
        video = rfile.stem
        R, B = json.load(open(rfile)), json.load(open(arm / rfile.name))
        f0 = root / "Annotations" / a.tier / video / f"{meta[video]['frames'][0]}.png"
        for exp_id, e in R.items():
            n_all += 1
            rb, bb = [list(map(float, b[:5])) for b in e["boxes"]], [list(map(float, b[:5])) for b in B[exp_id]["boxes"]]
            if sorted(rb) != sorted(bb):
                sys.exit(f"{video} {exp_id}: the arm's boxes are not the same set as the cached pool")
            tb = bbox(np.asarray(read_palette_mask(f0, int(e["obj_id"]))).astype(bool))
            if tb is None:
                absent += 1
                absent_changed += int(bool(rb) and rb[0] != bb[0])
                continue
            ia = box_iou(rb[0][:4], tb) if rb else 0.0
            ib = box_iou(bb[0][:4], tb) if bb else 0.0
            rows.append((video, exp_id, ia, ib, ia >= a.hit, ib >= a.hit))

    n = len(rows)
    ha, hb = np.array([r[4] for r in rows], float), np.array([r[5] for r in rows], float)
    ia, ib = np.array([r[2] for r in rows]), np.array([r[3] for r in rows])
    V = np.array([r[0] for r in rows])
    print(a.label)
    print(f"sentences {n_all}; target visible at frame 0 (scored): {n}; absent at frame 0 (not scored): {absent} (box changed in {absent_changed}); hit = box IoU >= {a.hit}")
    print(f"{'':28s} {'hits':>5} {'of':>4} {'choice accuracy':>16} {'mean IoU':>9}")
    print(f"{'A0 top-1':28s} {int(ha.sum()):>5d} {n:>4d} {ha.mean():>16.1%} {ia.mean():>9.3f}")
    print(f"{'arm':28s} {int(hb.sum()):>5d} {n:>4d} {hb.mean():>16.1%} {ib.mean():>9.3f}")
    gained, kept, lost = int(((hb == 1) & (ha == 0)).sum()), int((hb == ha).sum()), int(((hb == 0) & (ha == 1)).sum())
    print(f"sentences: gained the hit {gained}, same {kept}, lost the hit {lost}")

    vids = sorted(set(V))
    idx = {v: np.where(V == v)[0] for v in vids}
    cnt = np.array([len(idx[v]) for v in vids], float)
    for name, d in (("choice accuracy", hb - ha), ("mean IoU", ib - ia)):
        sd = np.array([d[idx[v]].sum() for v in vids])
        rng = np.random.default_rng(a.seed)
        boot = [sd[s].sum() / cnt[s].sum() for s in (rng.integers(0, len(vids), len(vids)) for _ in range(a.reps))]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        fmt = "{:+.1%}" if name == "choice accuracy" else "{:+.3f}"
        print(f"arm - A0, {name}: {fmt.format(d.mean())}   95% interval [{fmt.format(lo)}, {fmt.format(hi)}] from {a.reps} resamples of the {len(vids)} videos (seed {a.seed})")

    print(f"\n{'group (A0 top-1 box at frame 0)':36s} {'n':>4} {'A0 hits':>8} {'arm hits':>9}")
    G = np.array([grp.get((r[0], r[1]), "?") for r in rows])
    for name, key in [("top-1 correct", "correct"), ("top-1 partial overlap", "partial"), ("top-1 on another annotated object", "other_obj"),
                      ("top-1 elsewhere", "elsewhere"), ("no candidate", "no_candidate")]:
        m = G == key
        if m.sum():
            print(f"{name:36s} {int(m.sum()):>4} {int(ha[m].sum()):>8d} {int(hb[m].sum()):>9d}")
    print("\nper video: sentences, A0 accuracy, arm accuracy")
    for v in vids:
        m = idx[v]
        print(f"  {v:14s} n {len(m):>3d}  {ha[m].mean():6.1%} {hb[m].mean():6.1%}")
    print("\nSingle run on a subset. The interval covers the choice of videos only; sentences of one video are resampled together and are not independent.")


if __name__ == "__main__":
    sys.exit(main())
