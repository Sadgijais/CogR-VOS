#!/usr/bin/env python3
"""
Task 6 candidate selection, arm A1 (Amendment 6 item 4): CLIP re-scoring of the cached frame-0 boxes. Needs no ground truth and does no tracking.

For every sentence of the cached Grounding-DINO folder: crop each cached box from frame 0 (the box plus --pad pixels of padding, no masking), take the
CLIP ViT-B/32 image embedding (openai/clip-vit-base-patch32, frozen, common/embed.py), take its cosine with the CLIP text embedding of the sentence text
exactly as stored in the grounding file (no prompt template), and choose the box with the highest cosine. A tie goes to the higher detector score (the
cached list is sorted by detector score, so the lower list position). The chosen box is moved to the front of a COPY of the grounding folder; the other
boxes keep their order behind it. Stage 2 reads boxes[0] at frame 0 only, so the copy is what the V0 run is prompted with.
Sentences whose target is absent at frame 0 are treated like all others (a real system cannot know). The ground truth is never opened here.
The source folder is never modified. Also writes selection_a1.csv (one row per sentence: cosines, detector scores, chosen position).

  python -I tools/select_a1_clip.py --root ~/data/MeViS-subset --grounding <run>/grounding --out <run>/grounding_a1_clip
"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def box_crop(image, box, pad=8):
    """RGB crop of the box [x1, y1, x2, y2(, score)] with `pad` pixels of padding, clamped to the image; never smaller than 2 x 2 pixels."""
    w, h = image.size
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    x1, y1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
    x2, y2 = min(w, max(int(x2) + pad, x1 + 2)), min(h, max(int(y2) + pad, y1 + 2))
    return image.convert("RGB").crop((x1, y1, x2, y2))


def choose(cosines):
    """Index of the highest cosine; an exact tie goes to the lower index (= higher detector score in the sorted cache)."""
    best = 0
    for i, c in enumerate(cosines):
        if c > cosines[best]:
            best = i
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root with JPEGImages/<tier>/")
    ap.add_argument("--grounding", required=True, help="source folder with <video>.json (never modified)")
    ap.add_argument("--out", required=True, help="new grounding folder (must not exist yet)")
    ap.add_argument("--tier", default="480p")
    ap.add_argument("--pad", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda; the choice is recorded in the log line")
    a = ap.parse_args()

    from common.embed import ClipCropEmbedder, TextEmbedder

    root, src, out = Path(a.root).expanduser(), Path(a.grounding).expanduser(), Path(a.out).expanduser()
    if out.exists():
        sys.exit(f"refusing to write into an existing folder: {out}")
    out.mkdir(parents=True)
    t0 = time.time()
    text_emb, img_emb = TextEmbedder(device=a.device), ClipCropEmbedder(device=a.device)

    log, n_seen, n_changed = [], 0, 0
    for gfile in sorted(src.glob("*.json")):
        video = gfile.stem
        data = json.load(open(gfile))
        frames = sorted((root / "JPEGImages" / a.tier / video).glob("*.jpg"))
        img0 = Image.open(frames[0]).convert("RGB")
        for exp_id, entry in data.items():
            n_seen += 1
            boxes = entry["boxes"]
            row = {"video": video, "exp_id": exp_id, "n_candidates": len(boxes), "chosen_pos": 0, "cosines": "", "detector_scores": ""}
            if len(boxes) > 1:
                t = text_emb([entry["expression"]])[0]
                im = img_emb.embed_images([box_crop(img0, b, a.pad) for b in boxes])
                cos = [float(x) for x in (im @ t)]
                best = choose(cos)
                row["chosen_pos"] = best
                row["cosines"] = ";".join(f"{c:.5f}" for c in cos)
                row["detector_scores"] = ";".join(f"{float(b[4]):.3f}" for b in boxes)
                if best != 0:
                    entry["boxes"] = [boxes[best]] + boxes[:best] + boxes[best + 1:]
                    n_changed += 1
            log.append(row)
        json.dump(data, open(out / gfile.name, "w"), indent=2)

    with open(out / "selection_a1.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["video", "exp_id", "n_candidates", "chosen_pos", "cosines", "detector_scores"])
        w.writeheader()
        w.writerows(log)
    text_emb.free(); img_emb.free()
    secs = time.time() - t0
    print(f"A1 CLIP re-score ({a.device}): {n_seen} sentences, box moved to the front in {n_changed}; {secs:.1f} s in total, {secs / max(n_seen, 1):.3f} s per sentence "
          f"(includes model loading); written {out}")


if __name__ == "__main__":
    sys.exit(main())
