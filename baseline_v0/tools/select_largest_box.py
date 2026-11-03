#!/usr/bin/env python3
"""
Task 6 reference arm "largest box" (Amendment 8; additional reference / exploratory experiment, NOT a candidate method; needs no image, no text, no ground
truth, no GPU). For every sentence of the cached Grounding-DINO folder the box with the largest area (x2 - x1) * (y2 - y1) among the cached boxes is chosen;
an exact tie goes to the higher detector score (the cached list is sorted by detector score, so the lower list position). The chosen box is moved to the front
of a COPY of the grounding folder; the other boxes keep their order behind it. Stage 2 reads boxes[0] at frame 0 only, so the copy is what the V0 run is
prompted with. Sentences whose target is absent at frame 0 are treated like all others. The source folder is never modified. Also writes selection_largest.csv
(one row per sentence: areas, detector scores, chosen position).

  python -I tools/select_largest_box.py --grounding <run>/grounding --out <run>/grounding_largest
"""
import argparse, csv, json, sys
from pathlib import Path


def area(box):
    """Area of [x1, y1, x2, y2(, score)]; 0 for a degenerate box."""
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def choose(areas):
    """Index of the largest area; an exact tie goes to the lower index (= higher detector score in the sorted cache)."""
    best = 0
    for i, a in enumerate(areas):
        if a > areas[best]:
            best = i
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grounding", required=True, help="source folder with <video>.json (never modified)")
    ap.add_argument("--out", required=True, help="new grounding folder (must not exist yet)")
    a = ap.parse_args()

    src, out = Path(a.grounding).expanduser(), Path(a.out).expanduser()
    if out.exists():
        sys.exit(f"refusing to write into an existing folder: {out}")
    out.mkdir(parents=True)

    log, n_seen, n_changed = [], 0, 0
    for gfile in sorted(src.glob("*.json")):
        data = json.load(open(gfile))
        for exp_id, entry in data.items():
            n_seen += 1
            boxes = entry["boxes"]
            row = {"video": gfile.stem, "exp_id": exp_id, "n_candidates": len(boxes), "chosen_pos": 0, "areas": "", "detector_scores": ""}
            if len(boxes) > 1:
                areas = [area(b) for b in boxes]
                best = choose(areas)
                row["chosen_pos"] = best
                row["areas"] = ";".join(f"{x:.0f}" for x in areas)
                row["detector_scores"] = ";".join(f"{float(b[4]):.3f}" for b in boxes)
                if best != 0:
                    entry["boxes"] = [boxes[best]] + boxes[:best] + boxes[best + 1:]
                    n_changed += 1
            log.append(row)
        json.dump(data, open(out / gfile.name, "w"), indent=2)

    with open(out / "selection_largest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["video", "exp_id", "n_candidates", "chosen_pos", "areas", "detector_scores"])
        w.writeheader()
        w.writerows(log)
    print(f"largest box: {n_seen} sentences, box moved to the front in {n_changed}; written {out}")


if __name__ == "__main__":
    sys.exit(main())
