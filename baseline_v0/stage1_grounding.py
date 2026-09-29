#!/usr/bin/env python3
"""
Stage 1 — grounding.

For every video's frame 0 and every referring expression, run Grounding-DINO
and save the top-K candidate boxes to grounding/<video>.json. This is the
ONLY stage that touches Grounding-DINO; it runs once, the model is then
unloaded, and every later stage (and every later ablation variant, V1..V5)
re-uses this cache instead of re-running the frontend.

Requires a GPU with the `transformers` and `torch` packages installed — see
README.md "Environment setup". This script does nothing useful without the
actual Ref-DAVIS17 files on disk, so it has not been run in the environment
that wrote this code; it has been written carefully against the documented
HuggingFace GroundingDINO API and should be treated as a first draft to
debug against your real data, not as pre-verified.

Usage:
    python stage1_grounding.py --config config.yaml
    python stage1_grounding.py --config config.yaml --tier smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from common.dataset import DavisLayout, load_expressions, cross_check_obj_ids
from common.io_utils import dump_json


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_tier_videos(cfg: dict) -> set[str] | None:
    """
    Returns the set of video names to restrict to, based on
    dataset.tier (smoke / ablation / full). Looks for frozen manifests
    under subsets/ (see CogR-VOS_Dataset_Subset_Protocol.md); if you have
    not built those yet, `full` is the only tier available and
    smoke/ablation will raise a clear error rather than silently running
    on everything.
    """
    tier = cfg["dataset"].get("tier", "full")
    manifest_path = Path("subsets") / f"ref_davis17_{tier}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"dataset.tier is '{tier}' but {manifest_path} does not exist. "
            "Either build the frozen subset manifest first (see "
            "CogR-VOS_Dataset_Subset_Protocol.md), or set dataset.tier: full "
            "in config.yaml to run every video (slower, but always works)."
        )
    import json

    with open(manifest_path) as f:
        manifest = json.load(f)
    return set(manifest["videos"])


def run_grounding_dino(image_path: Path, text: str, cfg: dict):
    """
    Runs HuggingFace's GroundingDinoForObjectDetection on one image/text
    pair and returns up to top_k boxes as [x1, y1, x2, y2, score] in pixel
    coordinates, sorted by score descending.

    Uses the HF port, NOT the IDEA-Research reference repo, specifically
    because the reference repo requires compiling a custom
    MultiScaleDeformableAttention CUDA extension that is where most
    Windows/WSL installs fail (same weights either way) — see
    CogR-VOS_V0_Baseline_Spec.md §2.
    """
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    device = cfg["grounding"]["device"]
    model_id = cfg["grounding"]["model_id"]

    # Cached at module level by the caller (see main()) so this function is
    # only responsible for one forward pass.
    processor = run_grounding_dino._processor
    model = run_grounding_dino._model

    image = Image.open(image_path).convert("RGB")
    # Grounding-DINO expects lowercase text ending in a period.
    prompt = text.strip().lower()
    if not prompt.endswith("."):
        prompt += "."

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=cfg["grounding"]["box_threshold"],
        text_threshold=cfg["grounding"]["text_threshold"],
        target_sizes=[image.size[::-1]],  # (height, width)
    )[0]

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()

    order = scores.argsort()[::-1]
    top_k = cfg["grounding"]["top_k_boxes"]
    out = []
    for i in order[:top_k]:
        x1, y1, x2, y2 = boxes[i].tolist()
        out.append([float(x1), float(y1), float(x2), float(y2), float(scores[i])])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    layout = DavisLayout(root=Path(cfg["dataset"]["root"]), require_480p=cfg["evaluation"]["require_480p"])

    annotator_set = cfg["dataset"].get("annotator_set")
    expressions = load_expressions(layout, annotator_set)

    problems = cross_check_obj_ids(layout, expressions)
    if problems:
        print("WARNING — obj_id / palette cross-check found problems:")
        for p in problems:
            print("  -", p)
        print(
            "Fix these before trusting any downstream number — this is "
            "silent-killer #3 from the working primer. Continuing anyway "
            "so you can see stage 1 run, but do not evaluate on this data "
            "until it is clean."
        )

    tier_videos = load_tier_videos(cfg)
    if tier_videos is not None:
        expressions = [e for e in expressions if e.video in tier_videos]

    videos = sorted({e.video for e in expressions})
    print(f"Stage 1: grounding {len(expressions)} expressions across {len(videos)} videos.")

    # Load the model ONCE, reuse across every expression, then unload —
    # this is the "staged, disk-cached pipeline" from the master brief.
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    print(f"Loading {cfg['grounding']['model_id']} ...")
    run_grounding_dino._processor = AutoProcessor.from_pretrained(cfg["grounding"]["model_id"])
    run_grounding_dino._model = AutoModelForZeroShotObjectDetection.from_pretrained(
        cfg["grounding"]["model_id"]
    ).to(cfg["grounding"]["device"]).eval()

    out_dir = Path(cfg["paths"]["grounding_out"])
    by_video: dict[str, dict] = {}

    for video in videos:
        video_exprs = [e for e in expressions if e.video == video]
        frame0 = layout.frame_paths(video)[0]
        layout.check_resolution(video)

        video_result = {}
        for e in video_exprs:
            boxes = run_grounding_dino(frame0, e.text, cfg)
            video_result[e.exp_id] = {
                "expression": e.text,
                "obj_id": e.obj_id,
                "frame0": frame0.name,
                "boxes": boxes,  # top-K, [x1,y1,x2,y2,score], V0 uses only boxes[0]
            }
            top1 = boxes[0] if boxes else None
            print(f"  {video}/{e.exp_id} ('{e.text}') -> top1={top1}")

        by_video[video] = video_result
        dump_json(out_dir / f"{video}.json", video_result)

    # Free the GPU before stage 2 loads SAM2 — they cannot be co-resident
    # on a 4 GB card.
    del run_grounding_dino._model
    torch.cuda.empty_cache()
    print(f"Stage 1 done. Wrote {len(videos)} files under {out_dir}/")


if __name__ == "__main__":
    main()
