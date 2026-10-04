#!/usr/bin/env python3
"""
Stage 1b — V1 embedding cache (one pass per video, disk-cached).

For every expression in grounding/<video>.json it stores, in
embeddings/<video>.npz:
    <exp_id>__text    (512,)   CLIP text embedding of the expression (semantic field)
    <exp_id>__boxes   (k,384)  DINOv2 embedding of each cached top-k box crop at frame 0
                               (context field: appearance of the distractor candidates)

Separate process from stage 2 (4 GB budget). Reads the V0 grounding cache;
never changes it.

Usage:
    python stage1b_embeddings.py --config config_v1.yaml
    python stage1b_embeddings.py --config config_v1.yaml --video blackswan
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common.dataset import DavisLayout
from common.embed import CropEmbedder, TextEmbedder
from common.io_utils import load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v1.yaml")
    ap.add_argument("--video", default=None, help="only this video")
    ap.add_argument("--force", action="store_true", help="recompute existing caches")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    layout = DavisLayout(root=Path(cfg["dataset"]["root"]),
                         require_480p=cfg["evaluation"]["require_480p"])
    grounding_dir = Path(cfg["paths"]["grounding_out"])
    out_dir = Path(cfg.get("v1", {}).get("embeddings_out", "embeddings/"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg["grounding"]["device"]

    files = sorted(grounding_dir.glob("*.json"))
    if args.video:
        files = [f for f in files if f.stem == args.video]
    if not files:
        raise FileNotFoundError(f"no grounding cache in {grounding_dir}/ (run stage 1 first)")

    text_emb = TextEmbedder(device=device)
    crop_emb = CropEmbedder(device=device)

    for gfile in files:
        video = gfile.stem
        out_path = out_dir / f"{video}.npz"
        if out_path.exists() and not args.force:
            print(f"skip {video} (cached)")
            continue
        vg = load_json(gfile)
        frames = layout.frame_paths(video)
        img0 = Image.open(frames[0]).convert("RGB")
        arrays = {}
        for exp_id, entry in vg.items():
            arrays[f"{exp_id}__text"] = text_emb([entry["expression"]])[0]
            boxes = entry["boxes"]
            if boxes:
                arrays[f"{exp_id}__boxes"] = np.stack([crop_emb.embed_box(img0, b) for b in boxes])
            else:
                arrays[f"{exp_id}__boxes"] = np.zeros((0, 384), dtype=np.float32)
        np.savez_compressed(out_path, **arrays)
        print(f"{video}: {len(vg)} expression(s) -> {out_path}")

    print("Stage 1b done.")


if __name__ == "__main__":
    main()
