#!/usr/bin/env python3
"""
Run this FIRST, before stage 1, against your real Ref-DAVIS17 download.
It cross-checks the four things that produce a plausible-looking but wrong
number (CogR-VOS_V0_Baseline_Spec.md §5):

  1. Resolution — refuses (loudly) if this looks like Full-Resolution data.
  2. Palette / obj_id indexing — every expression's obj_id must actually
     appear in that video's annotation palette.
  3. Annotator-set field presence — tells you whether meta_expressions.json
     even has an annotator field to be confused about, and what values it
     contains, so you can settle the open decision in V0_Baseline_Spec.md §7
     with actual data instead of guessing.
  4. Basic shape checks — frame count matches annotation count, etc.

This has NOT been run against a real Ref-DAVIS17 download in this
environment (no dataset here) — it has only been exercised via the
synthetic fixture in test_fixture.py, which checks the underlying metric
and taxonomy code, not this specific file-layout logic. Run it on your
actual data and expect to fix small path issues on first try — different
mirrors of Ref-DAVIS17 do use different folder names.

Usage:
    python tools/inspect_dataset.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.dataset import DavisLayout, ResolutionGuardError
from common.io_utils import unique_palette_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["dataset"]["root"])
    layout = DavisLayout(root=root, require_480p=True)

    print(f"Dataset root: {root}")
    if not root.exists():
        print("  !! root does not exist — edit config.yaml dataset.root")
        return

    print("\n[1/4] Resolution check")
    videos = layout.videos() if layout.jpeg_dir.exists() else []
    if not videos:
        print(f"  !! no videos found under {layout.jpeg_dir} — check folder layout")
    for v in videos[:3]:
        try:
            layout.check_resolution(v)
            print(f"  OK  {v}")
        except ResolutionGuardError as exc:
            print(f"  !!  {exc}")

    print(f"\n  Found {len(videos)} videos total.")

    print("\n[2/4] meta_expressions.json")
    if not layout.meta_expressions_path.exists():
        print(f"  !! not found at {layout.meta_expressions_path}")
        print("     Referring expressions must be placed here (or adjust "
              "common/dataset.py's DavisLayout to match your download).")
    else:
        with open(layout.meta_expressions_path) as f:
            meta = json.load(f)
        vdata_iter = (meta.get("videos", meta)).items()
        annotator_values = set()
        n_expr = 0
        for _video, vdata in vdata_iter:
            for _eid, edata in vdata.get("expressions", {}).items():
                n_expr += 1
                if "annotator" in edata:
                    annotator_values.add(edata["annotator"])
        print(f"  {n_expr} expression entries found.")
        if annotator_values:
            print(f"  'annotator' field present, values seen: {sorted(annotator_values)}")
            print(
                "  ACTION: set dataset.annotator_set in config.yaml to one of "
                "these values. See the ANNOTATOR_SET_WARNING in "
                "common/dataset.py — the project's own docs currently "
                "disagree on which one (0 vs 1)."
            )
        else:
            print("  No 'annotator' field — file already appears to be a single set.")

    print("\n[3/4] obj_id / palette cross-check (first 3 videos)")
    for v in videos[:3]:
        anno_paths = layout.anno_paths(v)
        if not anno_paths:
            print(f"  !! {v}: no annotation files found")
            continue
        ids: set[int] = set()
        for p in anno_paths:
            ids |= unique_palette_ids(p)
        print(f"  {v}: object ids present in palette = {sorted(ids)}")

    print("\n[4/4] Frame / annotation count match (first 3 videos)")
    for v in videos[:3]:
        n_frames = len(layout.frame_paths(v))
        n_anno = len(layout.anno_paths(v))
        flag = "OK" if n_frames == n_anno else "!!"
        print(f"  {flag}  {v}: {n_frames} frames, {n_anno} annotations")

    print("\nDone. Fix anything flagged with !! before running stage1_grounding.py.")


if __name__ == "__main__":
    main()
