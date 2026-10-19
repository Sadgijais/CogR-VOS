#!/usr/bin/env python3
"""
Task 6: draw frame strips for chosen failure cases. CPU only, reads files a finished run already wrote, writes PNGs.
Green outline = ground truth of the target object, red fill + outline = the system's mask (one results folder).
Each tile is labelled with the frame file name and the IoU of that frame. Nothing is re-run, no Gemini.
Case spec:  VIDEO:EXP   or   VIDEO:EXP:i1,i2,...   (i = positions in the sorted list of frames shared by
the image, ground-truth and prediction folders; without indices, 8 evenly spread frames are shown).
"""
import argparse, csv, os
import numpy as np
import cv2
from PIL import Image

GREEN = (0, 200, 0)
RED = (0, 0, 230)
TILE_W = 320
COLS = 4


def read_mask(path):
    im = Image.open(path)
    if im.mode not in ("P", "L", "1", "I", "I;16"):
        im = im.convert("L")
    return np.array(im).astype(np.int64)


def contours(m):
    res = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return res[-2]


def ascii_(s):
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in s)


def tile(img, gt, pred, name):
    h, w = img.shape[:2]
    if pred.shape != (h, w):
        pred = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    if gt.shape != (h, w):
        gt = cv2.resize(gt.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    out = img.copy()
    fill = out.copy()
    fill[pred] = RED
    out = cv2.addWeighted(fill, 0.45, out, 0.55, 0)
    cv2.drawContours(out, contours(pred), -1, RED, 2)
    cv2.drawContours(out, contours(gt), -1, GREEN, 2)
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    iou = "both empty" if union == 0 else "IoU %.2f" % (inter / union)
    th = int(round(h * TILE_W / w))
    out = cv2.resize(out, (TILE_W, th), interpolation=cv2.INTER_AREA)
    bar = np.full((22, TILE_W, 3), 30, np.uint8)
    cv2.putText(bar, "%s   %s" % (name, iou), (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([out, bar])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root that holds JPEGImages/480p and Annotations/480p")
    ap.add_argument("--preds", required=True, help="predictions_x folder: <video>/<exp_id>/<frame>.png")
    ap.add_argument("--results", required=True, help="results_x/per_video_results.csv (gives obj_id and scores)")
    ap.add_argument("--selected", default=None, help="selected_cases.csv (optional, gives tags and failure mode)")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    res = {(r["video"], r["exp_id"]): r for r in csv.DictReader(open(a.results, newline=""))}
    sel = {}
    if a.selected and os.path.exists(a.selected):
        sel = {(r["video"], r["exp_id"]): r for r in csv.DictReader(open(a.selected, newline=""))}
    os.makedirs(a.out_dir, exist_ok=True)

    for spec in a.cases:
        parts = spec.split(":")
        video, exp = parts[0], parts[1]
        idx = [int(x) for x in parts[2].split(",")] if len(parts) > 2 and parts[2] else None
        r = res[(video, exp)]
        obj = int(r["obj_id"])
        jd = os.path.join(a.data, "JPEGImages", "480p", video)
        ad = os.path.join(a.data, "Annotations", "480p", video)
        pd = os.path.join(a.preds, video, exp)
        stems = sorted(set(f[:-4] for f in os.listdir(jd) if f.endswith(".jpg"))
                       & set(f[:-4] for f in os.listdir(ad) if f.endswith(".png"))
                       & set(f[:-4] for f in os.listdir(pd) if f.endswith(".png")))
        if idx is None:
            idx = sorted(set(int(round(x)) for x in np.linspace(0, len(stems) - 1, a.n)))
        idx = [i for i in idx if 0 <= i < len(stems)]
        tiles = []
        for i in idx:
            s = stems[i]
            img = cv2.imread(os.path.join(jd, s + ".jpg"))
            gt = read_mask(os.path.join(ad, s + ".png")) == obj
            pred = read_mask(os.path.join(pd, s + ".png")) > 0
            tiles.append(tile(img, gt, pred, "#%d  frame %s" % (i, s)))
        th = max(t.shape[0] for t in tiles)
        tiles = [np.vstack([t, np.zeros((th - t.shape[0], TILE_W, 3), np.uint8)]) if t.shape[0] < th else t for t in tiles]
        while len(tiles) % COLS:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[k:k + COLS]) for k in range(0, len(tiles), COLS)]
        body = np.vstack(rows)
        s = sel.get((video, exp), {})
        tags = s.get("tags", "")
        mode = s.get("failure_mode", "")
        head = np.full((64, body.shape[1], 3), 20, np.uint8)
        cv2.putText(head, ascii_("%s  exp %s  J&F %.1f  %s  %s" % (video, exp, float(r["JF"]), mode, tags))[:150], (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(head, ascii_('"%s"' % r["expression"])[:150], (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(head, "green outline = ground truth, red = prediction. Reserved subset from the training split - not official validation.",
                    (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        out = os.path.join(a.out_dir, "%s_exp%s.png" % (video, exp))
        cv2.imwrite(out, np.vstack([head, body]))
        print("wrote", out, "frames:", ", ".join(stems[i] for i in idx))


if __name__ == "__main__":
    main()
