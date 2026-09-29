#!/usr/bin/env python3
"""
tools/build_meta_expressions.py

Converts the plain-text Ref-DAVIS17 referring-expression files
(davis_text_annotations/Davis17_annot1.txt, Davis17_annot2.txt --
format: `video_name object_id "referring expression"`, one line per
(video, object) pair) into the meta_expressions.json schema that
common/dataset.py's load_expressions() actually reads:

    {"videos": {"<video>": {"expressions": {"<exp_id>": {
        "obj_id": int, "exp": str, "annotator": int}}}}}

Annotator-set resolution (closes CogR-VOS_V0_Baseline_Spec.md Sec 7):
    Davis17_annot1.txt -> annotator 0
    Davis17_annot2.txt -> annotator 1
This matches config.yaml's existing `dataset.annotator_set: 0` default and
CogR-VOS_Dataset_Subset_Protocol.md's "annotator set 0". The apparent
disagreement with CogR-VOS_Dataset_Audit_Findings.md's "annotator set 1"
was two conventions naming the same file: a 0-indexed array position vs.
the file's own 1-indexed suffix. Only two files exist on disk, so there is
no third option to have meant.

Usage:
    python tools/build_meta_expressions.py --root /home/<user>/data/Ref-DAVIS17/DAVIS
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LINE_RE = re.compile(r'^(\S+)\s+(\d+)\s+"(.*)"\s*$')

SOURCE_FILES = [
    ("Davis17_annot1.txt", 0),
    ("Davis17_annot2.txt", 1),
]


def parse_file(path: Path, annotator: int) -> list[tuple[str, int, str]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("video_name") or low.startswith("e.g."):
                continue  # header / example line from format.txt-style content
            m = LINE_RE.match(line)
            if not m:
                print(f"  !! {path.name}:{lineno}: could not parse line: {line!r} -- skipped")
                continue
            video, obj_id, text = m.group(1), int(m.group(2)), m.group(3)
            out.append((video, obj_id, text))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Ref-DAVIS17-480p dataset root (the folder containing JPEGImages/, Annotations/)")
    ap.add_argument("--text-dir", default=None,
                     help="Folder containing Davis17_annot*.txt "
                          "(default: tries <root>/davis_text_annotations, then <root>/../davis_text_annotations)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    candidates = []
    if args.text_dir:
        candidates.append(Path(args.text_dir).expanduser())
    candidates += [root / "davis_text_annotations", root.parent / "davis_text_annotations"]
    text_dir = next((c for c in candidates if c.exists()), None)
    if text_dir is None:
        raise SystemExit(
            "Could not find davis_text_annotations/ near "
            f"{root} -- pass --text-dir explicitly. Tried: "
            + ", ".join(str(c) for c in candidates)
        )

    print(f"Dataset root:  {root}")
    print(f"Reading annotations from: {text_dir}\n")

    videos: dict[str, dict] = {}
    total = 0
    for fname, annotator in SOURCE_FILES:
        fpath = text_dir / fname
        if not fpath.exists():
            print(f"  !! {fpath} not found -- skipping annotator {annotator}")
            continue
        entries = parse_file(fpath, annotator)
        print(f"  {fname} -> annotator {annotator}: {len(entries)} (video, object) lines")
        for video, obj_id, text in entries:
            vdata = videos.setdefault(video, {"expressions": {}})
            exp_id = f"{annotator}_{obj_id}"
            vdata["expressions"][exp_id] = {
                "obj_id": obj_id,
                "exp": text,
                "annotator": annotator,
            }
            total += 1

    out = {"videos": videos}
    out_dir = root / "meta_expressions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "meta_expressions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {out_path}")
    print(f"  {len(videos)} videos, {total} (video, object, annotator) expressions total")
    print("\nNext: dataset.annotator_set: 0 in config.yaml selects Davis17_annot1.txt "
          "(already the default -- no change needed).")
    print("Then run: python tools/inspect_dataset.py --config config.yaml")


if __name__ == "__main__":
    main()
