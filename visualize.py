#!/usr/bin/env python3
"""
Overlays + contact sheets, for eyeballing what actually went wrong on a
specific (video, expression) — the fastest way to sanity-check a failure
label from failure_analysis.csv before trusting it.

For each scored expression, writes:
  visualizations/<video>/<exp_id>/overlay_XXXXX.png   — every Nth frame,
      predicted mask in one translucent colour, GT contour outlined
  visualizations/<video>/<exp_id>/contact_sheet.png    — a grid of evenly
      sampled frames, side by side, for a one-glance failure review

No GPU dependency. Run this after stage 3/4 so you can jump straight to the
expressions stage 4 flagged as identity_drift or grounding_failure.

Usage:
    python visualize.py --config config.yaml --video lab-coat --exp-id 0
    python visualize.py --config config.yaml --failure-mode identity_drift
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from common.dataset import DavisLayout
from common.io_utils import read_palette_mask, read_binary_mask

OVERLAY_COLOR = (255, 60, 60)   # predicted mask
GT_COLOR = (60, 220, 60)        # ground-truth contour
STRIDE = 5                      # every Nth frame gets a saved overlay
CONTACT_SHEET_N = 12


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def overlay_frame(frame_path: Path, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Image.Image:
    img = Image.open(frame_path).convert("RGB")
    arr = np.array(img).astype(np.float32)

    if pred_mask.any():
        overlay = np.zeros_like(arr)
        overlay[pred_mask] = OVERLAY_COLOR
        arr = arr * 0.6 + overlay * 0.4

    out = Image.fromarray(arr.astype(np.uint8))

    if gt_mask.any():
        # Draw GT boundary as a thin outline so it's visible even when it
        # overlaps the prediction.
        from scipy.ndimage import binary_erosion

        boundary = np.logical_and(gt_mask, np.logical_not(binary_erosion(gt_mask)))
        draw = ImageDraw.Draw(out)
        ys, xs = np.where(boundary)
        for x, y in zip(xs, ys):
            draw.point((int(x), int(y)), fill=GT_COLOR)

    return out


def make_contact_sheet(frames: list[Image.Image], labels: list[str]) -> Image.Image:
    thumb_w = 200
    thumbs = []
    for f in frames:
        ratio = thumb_w / f.width
        thumbs.append(f.resize((thumb_w, int(f.height * ratio))))
    thumb_h = max(t.height for t in thumbs) if thumbs else 0
    sheet = Image.new("RGB", (thumb_w * len(thumbs), thumb_h + 16), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, (t, label) in enumerate(zip(thumbs, labels)):
        sheet.paste(t, (i * thumb_w, 16))
        draw.text((i * thumb_w + 4, 2), label, fill=(255, 255, 255))
    return sheet


def visualize_one(layout: DavisLayout, video: str, exp_id: str, obj_id: int, pred_dir: Path, out_dir: Path):
    frame_paths = layout.frame_paths(video)
    anno_paths = layout.anno_paths(video)
    pred_exp_dir = pred_dir / video / exp_id
    exp_out = out_dir / video / exp_id
    exp_out.mkdir(parents=True, exist_ok=True)

    sampled_frames, sampled_labels = [], []
    stride_indices = list(range(0, len(frame_paths), STRIDE))
    contact_indices = np.linspace(0, len(frame_paths) - 1, CONTACT_SHEET_N).astype(int)

    for i, (frame_p, anno_p) in enumerate(zip(frame_paths, anno_paths)):
        pred_p = pred_exp_dir / (anno_p.stem + ".png")
        pred_mask = read_binary_mask(pred_p) if pred_p.exists() else None
        gt_mask = read_palette_mask(anno_p, obj_id)
        if pred_mask is None:
            pred_mask = np.zeros_like(gt_mask)

        if i in stride_indices or i in contact_indices:
            img = overlay_frame(frame_p, pred_mask, gt_mask)
            if i in stride_indices:
                img.save(exp_out / f"overlay_{i:05d}.png")
            if i in contact_indices:
                sampled_frames.append(img)
                iou_val = float(np.logical_and(pred_mask, gt_mask).sum()) / max(
                    1, float(np.logical_or(pred_mask, gt_mask).sum())
                )
                sampled_labels.append(f"f{i} iou={iou_val:.2f}")

    if sampled_frames:
        sheet = make_contact_sheet(sampled_frames, sampled_labels)
        sheet.save(exp_out / "contact_sheet.png")

    print(f"  wrote overlays + contact sheet for {video}/{exp_id} -> {exp_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video")
    ap.add_argument("--exp-id")
    ap.add_argument("--failure-mode", help="only visualize rows with this failure_analysis.csv label")
    args = ap.parse_args()
    cfg = load_config(args.config)

    layout = DavisLayout(root=Path(cfg["dataset"]["root"]), require_480p=cfg["evaluation"]["require_480p"])
    pred_dir = Path(cfg["paths"]["predictions_out"])
    out_dir = Path(cfg["paths"]["visualizations_out"])
    fa_path = Path(cfg["paths"]["failure_analysis_out"])

    targets = []
    if args.video and args.exp_id:
        # obj_id isn't known here without re-reading expressions; look it
        # up from failure_analysis.csv if present, else from meta_expressions.
        obj_id = None
        if fa_path.exists():
            with open(fa_path) as f:
                for row in csv.DictReader(f):
                    if row["video"] == args.video and row["exp_id"] == args.exp_id:
                        obj_id = int(row["obj_id"])
        if obj_id is None:
            from common.dataset import load_expressions

            for e in load_expressions(layout, cfg["dataset"].get("annotator_set")):
                if e.video == args.video and e.exp_id == args.exp_id:
                    obj_id = e.obj_id
        if obj_id is None:
            raise SystemExit(f"could not find obj_id for {args.video}/{args.exp_id}")
        targets.append((args.video, args.exp_id, obj_id))
    elif args.failure_mode:
        if not fa_path.exists():
            raise SystemExit(f"{fa_path} not found — run stage4_failure_analysis.py first")
        with open(fa_path) as f:
            for row in csv.DictReader(f):
                if row["failure_mode"] == args.failure_mode:
                    targets.append((row["video"], row["exp_id"], int(row["obj_id"])))
    else:
        raise SystemExit("pass either --video/--exp-id or --failure-mode")

    print(f"Visualizing {len(targets)} expression(s)...")
    for video, exp_id, obj_id in targets:
        visualize_one(layout, video, exp_id, obj_id, pred_dir, out_dir)


if __name__ == "__main__":
    main()
