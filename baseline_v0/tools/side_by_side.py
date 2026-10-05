"""V0 vs V1 side-by-side contact sheets (run from baseline_v0/).
Usage: python tools/side_by_side.py loading/0_2 india/0_1 motocross-jump/0_2
Per expression: 6 frames (first, last, and the 4 frames where V0 and V1 differ most in J).
Columns: [image + GT outline] [V0 mask] [V1 mask]; per-frame J printed on each mask panel.
Writes visualizations_v1_final/side_by_side/<video>_<exp>.png"""
import csv, sys
from pathlib import Path
import numpy as np
import yaml
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parents[1]
V0_PRED = HERE / "predictions"
V1_PRED = HERE / "predictions_v1_final"
OUT = HERE / "visualizations_v1_final" / "side_by_side"
GT_COLOR, PRED_COLOR = (0, 255, 0), (255, 60, 60)


def sub(root, kinds, video):
    for k in kinds:
        d = root / k / video
        if d.is_dir():
            return d
    sys.exit(f"ABORT: {video} not found under {root}")


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def edge(m):
    e = np.zeros_like(m)
    e[1:-1, 1:-1] = m[1:-1, 1:-1] & ~(m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:])
    return e


def overlay(img, mask, color, alpha=0.5):
    a = np.array(img, float)
    a[mask] = (1 - alpha) * a[mask] + alpha * np.array(color)
    return a.astype(np.uint8)


def load_mask(p, size):
    m = Image.open(p)
    if m.size != size:
        m = m.resize(size, Image.NEAREST)
    return np.array(m) > 0


def main():
    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    root = Path(cfg["dataset"]["root"])
    objs = {(r["video"], r["exp_id"]): int(r["obj_id"])
            for r in csv.DictReader(open(HERE.parent / "results/v0_baseline/per_video_results.csv"))}
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in sys.argv[1:]:
        video, exp = spec.split("/")
        obj = objs[(video, exp)]
        jd = sub(root, ("JPEGImages/480p", "JPEGImages/Full-Resolution", "JPEGImages"), video)
        ad = sub(root, ("Annotations/480p", "Annotations/Full-Resolution", "Annotations"), video)
        frames, annos = sorted(jd.glob("*.jpg")), sorted(ad.glob("*.png"))
        p0, p1 = sorted((V0_PRED / video / exp).glob("*.png")), sorted((V1_PRED / video / exp).glob("*.png"))
        n = min(len(frames), len(annos), len(p0), len(p1))
        size = Image.open(frames[0]).size
        gts = [np.array(Image.open(annos[i]).resize(size, Image.NEAREST)) == obj for i in range(n)]
        m0 = [load_mask(p0[i], size) for i in range(n)]
        m1 = [load_mask(p1[i], size) for i in range(n)]
        j0 = [iou(m0[i], gts[i]) for i in range(n)]
        j1 = [iou(m1[i], gts[i]) for i in range(n)]
        diff = sorted(range(1, n - 1), key=lambda i: -abs(j1[i] - j0[i]))[:4]
        idx = sorted({0, n - 1, *diff})
        w, h = size
        scale = 0.6
        pw, ph = int(w * scale), int(h * scale)
        sheet = Image.new("RGB", (3 * pw + 40, len(idx) * (ph + 6) + 40), (20, 20, 20))
        dr = ImageDraw.Draw(sheet)
        dr.text((5, 5), f"{video}/{exp} obj {obj}   J mean: V0 {100*np.mean(j0):.1f}   V1 {100*np.mean(j1):.1f}", fill=(255, 255, 255))
        for c, t in enumerate(["image + GT (green)", "V0 (red)", "V1 (red)"]):
            dr.text((10 + c * pw, 22), t, fill=(200, 200, 200))
        for r, i in enumerate(idx):
            img = Image.open(frames[i]).convert("RGB").resize(size)
            panels = [
                overlay(img, edge(gts[i]), GT_COLOR, 1.0),
                overlay(overlay(img, m0[i], PRED_COLOR), edge(gts[i]), GT_COLOR, 1.0),
                overlay(overlay(img, m1[i], PRED_COLOR), edge(gts[i]), GT_COLOR, 1.0),
            ]
            y = 40 + r * (ph + 6)
            for c, pn in enumerate(panels):
                sheet.paste(Image.fromarray(pn).resize((pw, ph)), (10 + c * pw, y))
            dr.text((14, y + 4), f"frame {i}", fill=(255, 255, 0))
            dr.text((14 + pw, y + 4), f"J {100*j0[i]:.0f}", fill=(255, 255, 0))
            dr.text((14 + 2 * pw, y + 4), f"J {100*j1[i]:.0f}", fill=(255, 255, 0))
        out = OUT / f"{video}_{exp}.png"
        sheet.save(out)
        print("wrote", out, f"(J V0 {100*np.mean(j0):.1f} -> V1 {100*np.mean(j1):.1f})")


if __name__ == "__main__":
    main()
