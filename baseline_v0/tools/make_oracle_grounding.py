#!/usr/bin/env python3
"""
Task 6 diagnostic (CPU only, no GPU, no API): write a COPY of the cached Grounding-DINO folder in which the frame-0 box that SAM 2
will be prompted with (boxes[0]) is replaced by an ORACLE choice. Stage 2 is not touched: it reads boxes[0] at frame 0.

  --mode gtbox  boxes = [ground-truth box of the target at frame 0 (score 1.0)] if the target is visible at frame 0.
                Tells whether the bottleneck is the initial grounding or the tracking after it.
  --mode cand   the cached candidate with the highest box IoU with the GT box (needs IoU >= --hit) is moved to the front;
                the other candidates keep their order behind it. Ceiling of any chooser (reranker, VLM) among the cached candidates.

A sentence whose target is NOT visible at frame 0 is left unchanged (no box on frame 0 can be the target); so is a cand-mode sentence
with no candidate at or above --hit. The oracle uses ground truth: it is a diagnostic CEILING, never a result of the method.
The original folder is never modified. Also writes oracle_changes.csv in the output folder (one row per sentence).

  python -I tools/make_oracle_grounding.py --root ~/data/MeViS-subset --grounding <run>/grounding --out <run>/grounding_oracle_gtbox --mode gtbox
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
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def box_iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--grounding", required=True, help="source folder with <video>.json (never modified)")
    ap.add_argument("--out", required=True, help="new folder (must not exist yet)")
    ap.add_argument("--mode", required=True, choices=["gtbox", "cand"])
    ap.add_argument("--tier", default="480p")
    ap.add_argument("--hit", type=float, default=0.5)
    a = ap.parse_args()

    root, src, out = Path(a.root).expanduser(), Path(a.grounding).expanduser(), Path(a.out).expanduser()
    if out.exists():
        sys.exit(f"refusing to write into an existing folder: {out}")
    out.mkdir(parents=True)
    meta = json.load(open(root / "meta_expressions" / "meta_expressions.json"))["videos"]
    ann = root / "Annotations" / a.tier

    log, n_changed, n_seen = [], 0, 0
    for gfile in sorted(src.glob("*.json")):
        video = gfile.stem
        data = json.load(open(gfile))
        frames = meta[video]["frames"]
        f0 = ann / video / f"{frames[0]}.png"
        for exp_id, entry in data.items():
            n_seen += 1
            oid = int(entry["obj_id"])
            tb = bbox(np.asarray(read_palette_mask(f0, oid)).astype(bool))
            boxes = entry["boxes"]
            action = "unchanged_target_absent_f0" if tb is None else "unchanged"
            if tb is not None:
                if a.mode == "gtbox":
                    entry["boxes"] = [tb + [1.0]]
                    action = "gt_box"
                else:
                    ious = [box_iou(b[:4], tb) for b in boxes]
                    best = max(range(len(ious)), key=lambda i: ious[i]) if ious else None
                    if best is not None and ious[best] >= a.hit:
                        if best != 0:
                            entry["boxes"] = [boxes[best]] + boxes[:best] + boxes[best + 1:]
                            action = f"moved_candidate_{best}_to_front"
                        else:
                            action = "top1_already_best"
                    else:
                        action = "unchanged_no_candidate_hit"
            n_changed += int(action.startswith(("gt_box", "moved")))
            log.append({"video": video, "exp_id": exp_id, "obj_id": oid, "action": action})
        json.dump(data, open(out / gfile.name, "w"), indent=2)

    with open(out / "oracle_changes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["video", "exp_id", "obj_id", "action"])
        w.writeheader()
        w.writerows(log)
    kinds = {}
    for r in log:
        k = "moved_candidate_to_front" if r["action"].startswith("moved") else r["action"]
        kinds[k] = kinds.get(k, 0) + 1
    print(f"mode {a.mode}: {n_seen} sentences, {n_changed} changed; written {out}")
    for k in sorted(kinds):
        print(f"  {k:32s} {kinds[k]:>5}")


if __name__ == "__main__":
    sys.exit(main())
