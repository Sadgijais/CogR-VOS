#!/usr/bin/env python3
"""
Quick per-expression comparison of two prediction folders against ground truth.
J only (region IoU, first/last frame excluded like the official protocol).
This is a FAST CHECK for tuning; official J, F, J&F still come from stage3_evaluate.py.

    python tools/compare_v0_v1.py --config config_v1.yaml --v0 predictions --v1 predictions_v1 \
        --video blackswan --video india
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.dataset import DavisLayout
from common.io_utils import load_json, read_binary_mask, read_palette_mask


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v1.yaml")
    ap.add_argument("--v0", default="predictions")
    ap.add_argument("--v1", default="predictions_v1")
    ap.add_argument("--video", action="append", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    layout = DavisLayout(root=Path(cfg["dataset"]["root"]), require_480p=False)
    excl = cfg["evaluation"].get("exclude_first_last", True)
    gdir = Path(cfg["paths"]["grounding_out"])
    v0d, v1d = Path(a.v0), Path(a.v1)

    print(f"{'video/exp':<26}{'J_v0':>7}{'J_v1':>7}{'delta':>7}{'%frames_differ':>16}{'IoU(v0,v1)':>12}")
    rows = []
    for gf in sorted(gdir.glob("*.json")):
        video = gf.stem
        if a.video and video not in a.video:
            continue
        annos = layout.anno_paths(video)
        for exp_id, e in load_json(gf).items():
            d0, d1 = v0d / video / exp_id, v1d / video / exp_id
            if not d1.exists():
                continue
            p0, p1 = sorted(d0.glob("*.png")), sorted(d1.glob("*.png"))
            n = min(len(p0), len(p1), len(annos))
            idx = range(1, n - 1) if excl and n > 2 else range(n)
            j0, j1, diff, agree = [], [], 0, []
            for i in idx:
                m0, m1 = read_binary_mask(p0[i]), read_binary_mask(p1[i])
                gt = read_palette_mask(annos[i], e["obj_id"])
                j0.append(iou(m0, gt)); j1.append(iou(m1, gt))
                diff += int((m0 != m1).any()); agree.append(iou(m0, m1))
            r = (f"{video}/{exp_id}", 100 * np.mean(j0), 100 * np.mean(j1), 100 * diff / len(idx), 100 * np.mean(agree))
            rows.append(r)
            print(f"{r[0]:<26}{r[1]:>7.1f}{r[2]:>7.1f}{r[2]-r[1]:>+7.1f}{r[3]:>16.1f}{r[4]:>12.1f}")
    if rows:
        m = np.mean([[r[1], r[2], r[3], r[4]] for r in rows], axis=0)
        print(f"{'MEAN over ' + str(len(rows)) + ' exprs':<26}{m[0]:>7.1f}{m[1]:>7.1f}{m[1]-m[0]:>+7.1f}{m[2]:>16.1f}{m[3]:>12.1f}")


if __name__ == "__main__":
    main()
