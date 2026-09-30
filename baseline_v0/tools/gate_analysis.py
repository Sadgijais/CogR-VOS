#!/usr/bin/env python3
"""
Does the similarity gate separate RIGHT frames from WRONG frames?
Joins per-frame similarity (results_v1/sttm_log/<video>.json) with per-frame J of the V1 mask
against ground truth. For each candidate tau_sim: share of correct frames (J>=0.5) that pass,
and share of wrong frames (J<0.5) that pass. Ref-DAVIS17 only; used to SET tau_sim, then frozen.

    python tools/gate_analysis.py --config config_v1.yaml --pred predictions_v1 --log results_v1/sttm_log
"""
import argparse
import json
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
    ap.add_argument("--pred", default="predictions_v1")
    ap.add_argument("--log", default="results_v1/sttm_log")
    ap.add_argument("--video", action="append", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    layout = DavisLayout(root=Path(cfg["dataset"]["root"]), require_480p=False)
    gdir = Path(cfg["paths"]["grounding_out"])

    ok_sims, bad_sims = [], []
    print(f"{'video/exp':<26}{'n_ok':>6}{'n_bad':>6}{'sim_ok(med)':>13}{'sim_bad(med)':>14}")
    for lf in sorted(Path(a.log).glob("*.json")):
        video = lf.stem
        if a.video and video not in a.video:
            continue
        annos = layout.anno_paths(video)
        g = load_json(gdir / f"{video}.json")
        for exp_id, v in json.load(open(lf)).items():
            preds = sorted((Path(a.pred) / video / exp_id).glob("*.png"))
            j0 = iou(read_binary_mask(preds[0]), read_palette_mask(annos[0], g[exp_id]["obj_id"]))
            if j0 < 0.5:      # wrong from frame 0: the anchor itself is wrong, so 'consistent with anchor' says nothing
                print(f"{video + '/' + exp_id:<26}  skipped: anchor wrong at frame 0 (J0={j0:.2f})")
                continue
            ok, bad = [], []
            for fr in v["frames"]:
                if fr["sim"] is None:
                    continue
                i = fr["f"]
                j = iou(read_binary_mask(preds[i]), read_palette_mask(annos[i], g[exp_id]["obj_id"]))
                (ok if j >= 0.5 else bad).append(fr["sim"])
            ok_sims += ok; bad_sims += bad
            med = lambda x: f"{np.median(x):.2f}" if x else "-"
            print(f"{video + '/' + exp_id:<26}{len(ok):>6}{len(bad):>6}{med(ok):>13}{med(bad):>14}")
    print(f"\nall frames: {len(ok_sims)} correct, {len(bad_sims)} wrong")
    print(f"{'tau_sim':>8}{'correct pass %':>16}{'wrong pass %':>14}")
    for tau in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        cp = 100 * np.mean(np.array(ok_sims) >= tau) if ok_sims else float("nan")
        wp = 100 * np.mean(np.array(bad_sims) >= tau) if bad_sims else float("nan")
        print(f"{tau:>8.1f}{cp:>16.1f}{wp:>14.1f}")


if __name__ == "__main__":
    main()
