#!/usr/bin/env python3
"""
Stage 2 — propagation.

Reads the grounding cache from stage 1, takes the TOP-1 box per expression
(V0 does not use the other 4 — they are cached for V4's re-identification
step later), prompts SAM2's video predictor on frame 0, and propagates the
mask to the end of the video. Writes one palette-style PNG per frame under
predictions/<video>/<exp_id>/.

This is a SEPARATE process from stage 1 on purpose: Grounding-DINO (~2-3 GB)
and SAM2 (~2.5-3 GB) cannot be co-resident on a 4 GB card, so stage 1 must
fully exit before this stage loads its model — see
CogR-VOS_V0_Baseline_Spec.md §3.

Requires the `sam2` package (facebookresearch/sam2) installed with
SAM2_BUILD_CUDA=0 (the optional CUDA extension only affects a hole-filling
post-process — skip it, it's the most common Windows/WSL install failure).
Like stage 1, this has been written carefully against the documented SAM2
video-predictor API but has NOT been run against real data in this
environment (no GPU, no dataset here) — debug it against your real files
first with the `smoke` tier before trusting it on `full`.

Usage:
    python stage2_propagation.py --config config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from common.dataset import DavisLayout
from common.io_utils import load_json, write_binary_mask


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def box_to_prompt(box):
    """[x1,y1,x2,y2,score] -> the (x1,y1,x2,y2) box SAM2 expects."""
    x1, y1, x2, y2, _score = box
    return [x1, y1, x2, y2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    layout = DavisLayout(root=Path(cfg["dataset"]["root"]), require_480p=cfg["evaluation"]["require_480p"])
    grounding_dir = Path(cfg["paths"]["grounding_out"])
    pred_dir = Path(cfg["paths"]["predictions_out"])

    grounding_files = sorted(grounding_dir.glob("*.json"))
    if not grounding_files:
        raise FileNotFoundError(
            f"No grounding cache found in {grounding_dir}/. Run "
            "stage1_grounding.py first — stage 2 only reads its output, it "
            "never calls Grounding-DINO itself."
        )

    import torch
    from sam2.build_sam import build_sam2_video_predictor

    print(f"Loading SAM2 ({cfg['propagation']['model_cfg']}) ...")
    predictor = build_sam2_video_predictor(
        cfg["propagation"]["model_cfg"],
        cfg["propagation"]["checkpoint"],
        device=cfg["propagation"]["device"],
    )

    for gfile in grounding_files:
        video = gfile.stem
        video_grounding = load_json(gfile)
        frames_dir = layout.jpeg_dir / video
        frame_paths = layout.frame_paths(video)
        n_frames = len(frame_paths)

        print(f"Propagating {video}: {len(video_grounding)} expression(s), {n_frames} frames")

        # `offload_video_to_cpu` / `offload_state_to_cpu`: required to fit
        # SAM2's video memory bank on a 4 GB card. See
        # CogR-VOS_V0_Baseline_Spec.md §3 for the ~22% speed cost of the
        # latter — worth it, VRAM is the binding constraint, not time.
        state = predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=cfg["propagation"]["offload_video_to_cpu"],
            offload_state_to_cpu=cfg["propagation"]["offload_state_to_cpu"],
        )

        # `batch_objects: true` — add every expression's box as a distinct
        # object id in the SAME SAM2 state, so the (expensive) image
        # encoder runs once per frame instead of once per expression.
        # SAM2 keeps objects independent in the batch dimension, so this
        # produces identical masks to running each expression separately.
        exp_ids = list(video_grounding.keys())
        sam_obj_id_of = {}
        for i, exp_id in enumerate(exp_ids, start=1):
            entry = video_grounding[exp_id]
            if not entry["boxes"]:
                print(f"  WARNING: {video}/{exp_id} has no grounding boxes at all — "
                      f"Grounding-DINO found nothing above threshold for "
                      f"'{entry['expression']}'. Emitting an empty mask for "
                      f"every frame (this IS the correct V0 behaviour for "
                      f"'no candidate' — see config.yaml grounding thresholds "
                      f"if this looks wrong for too many expressions).")
                sam_obj_id_of[exp_id] = None
                continue
            top1 = box_to_prompt(entry["boxes"][0])
            sam_obj_id = i
            sam_obj_id_of[exp_id] = sam_obj_id
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=sam_obj_id,
                box=np.array(top1, dtype=np.float32),
            )
            if not cfg["propagation"]["batch_objects"]:
                # One state per expression instead — simpler, slower.
                # (Left here for clarity; default path is batched.)
                pass

        # Propagate and write out masks.
        video_masks: dict[int, dict[int, np.ndarray]] = {}
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            for obj_id, logits in zip(obj_ids, mask_logits):
                mask = (logits > 0.0).squeeze().cpu().numpy()
                video_masks.setdefault(obj_id, {})[frame_idx] = mask

        for exp_id in exp_ids:
            sam_obj_id = sam_obj_id_of[exp_id]
            out_dir = pred_dir / video / exp_id
            for frame_idx, frame_path in enumerate(frame_paths):
                if sam_obj_id is None:
                    mask = np.zeros(_frame_hw(frame_path), dtype=bool)
                else:
                    mask = video_masks.get(sam_obj_id, {}).get(
                        frame_idx, np.zeros(_frame_hw(frame_path), dtype=bool)
                    )
                out_name = frame_path.stem + ".png"
                write_binary_mask(out_dir / out_name, mask)

        predictor.reset_state(state)

    del predictor
    torch.cuda.empty_cache()
    print(f"Stage 2 done. Wrote predictions under {pred_dir}/")


def _frame_hw(frame_path: Path):
    from PIL import Image

    with Image.open(frame_path) as img:
        w, h = img.size
    return (h, w)


if __name__ == "__main__":
    main()
