#!/usr/bin/env python3
"""
Per-frame picture of the Tracklet Coherence Score for one (video, expression). PIL only (no matplotlib).

  top panel     c_t over frames (black), its four terms (thin colours), band thresholds (dashed),
                background = band of each frame (green HIGH, yellow MEDIUM, red LOW, grey = target absent)
  bottom panel  true IoU of the prediction with the target (black) and with the best OTHER object (red dotted),
                red background = lost frames (IoU < 0.10), grey = target not in the ground truth

Needs results/tcs_frames.csv, which tools/tcs_analysis.py writes. Usage (from baseline_v0/):
    python tools/plot_tcs.py --results-dir results_v2_passive --video lab-coat --exp 0_1
    python tools/plot_tcs.py --results-dir results_v2_passive --auto 4        # picks failure cases / least stable tracks
Output: <results-dir>/plots/<video>__<exp>.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

COL = {"HIGH": (214, 240, 214), "MEDIUM": (252, 243, 196), "LOW": (248, 207, 207), "ABSENT": (222, 222, 222)}
TERM_COL = {"t_mask": (31, 119, 180), "t_motion": (255, 127, 14), "t_appearance": (44, 160, 44), "t_semantic": (148, 103, 189)}


def _f(x):
    return None if x in ("", None) else float(x)


def _b(x):
    return str(x).lower() == "true"


def load_rows(results_dir: Path):
    out = []
    for r in csv.DictReader(open(results_dir / "tcs_frames.csv")):
        out.append({"video": r["video"], "exp_id": r["exp_id"], "frame": int(r["frame"]), "c": float(r["c"]),
                    "band": r["band"], "visible": _b(r["visible"]), "gt_present": _b(r["gt_present"]),
                    "iou": float(r["iou"]), "iou_other": float(r["iou_other"]), "lost": _b(r["lost"]),
                    **{k: _f(r[k]) for k in TERM_COL}})
    return out


def _dashed(d, x0, y, x1, color, dash=6):
    x = x0
    while x < x1:
        d.line([(x, y), (min(x + dash, x1), y)], fill=color, width=1)
        x += dash * 2


def plot_one(rows, tau_low, tau_high, title, out_path: Path, width=1100):
    rows = sorted(rows, key=lambda r: r["frame"])
    frames = [r["frame"] for r in rows]
    f0, f1 = min(frames), max(frames)
    L, R, T1, B1, T2, B2 = 70, width - 20, 74, 300, 352, 520
    H = B2 + 40
    img = Image.new("RGB", (width, H), "white")
    d = ImageDraw.Draw(img)
    span = max(f1 - f0 + 1, 1)
    X = lambda f: L + (f - f0) * (R - L) / span
    wf = (R - L) / span

    d.text((L, 12), title, fill="black")
    d.text((L, 28), "top: coherence c_t and its terms; background = band.   bottom: true IoU; red = lost, grey = target not in GT",
           fill=(90, 90, 90))
    for (T, B) in ((T1, B1), (T2, B2)):
        d.rectangle([L, T, R, B], outline=(120, 120, 120))
    # ---- top panel
    for r in rows:
        band = "ABSENT" if not r["visible"] else r["band"]
        d.rectangle([X(r["frame"]), T1 + 1, X(r["frame"]) + wf + 0.5, B1 - 1], fill=COL[band])
    Y1 = lambda v: B1 - v * (B1 - T1)
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        d.text((L - 30, Y1(v) - 5), f"{v:.2f}", fill=(60, 60, 60))
    for tau, name in ((tau_high, "tau_high"), (tau_low, "tau_low")):
        _dashed(d, L, Y1(tau), R, (60, 60, 60))
        d.text((R - 58, Y1(tau) - 11), f"{name} {tau:.2f}", fill=(60, 60, 60))
    for key, color in TERM_COL.items():
        pts = [(X(r["frame"]) + wf / 2, Y1(r[key])) for r in rows if r[key] is not None]
        if len(pts) > 1:
            d.line(pts, fill=color, width=1)
    pts = [(X(r["frame"]) + wf / 2, Y1(r["c"])) for r in rows]
    if len(pts) > 1:
        d.line(pts, fill="black", width=3)
    lx = L
    for name, color in (("c_t", "black"), ("mask", TERM_COL["t_mask"]), ("motion", TERM_COL["t_motion"]),
                        ("appearance", TERM_COL["t_appearance"]), ("semantic", TERM_COL["t_semantic"])):
        d.line([(lx, T1 - 12), (lx + 16, T1 - 12)], fill=color, width=3)
        d.text((lx + 20, T1 - 18), name, fill="black")
        lx += 20 + 8 * len(name) + 18
    # ---- bottom panel
    for r in rows:
        if not r["gt_present"]:
            col = COL["ABSENT"]
        elif r["lost"]:
            col = COL["LOW"]
        else:
            continue
        d.rectangle([X(r["frame"]), T2 + 1, X(r["frame"]) + wf + 0.5, B2 - 1], fill=col)
    Y2 = lambda v: B2 - v * (B2 - T2)
    for v in (0.0, 0.5, 1.0):
        d.text((L - 30, Y2(v) - 5), f"{v:.2f}", fill=(60, 60, 60))
    _dashed(d, L, Y2(0.5), R, (150, 150, 150))
    other = [(X(r["frame"]) + wf / 2, Y2(r["iou_other"])) for r in rows]
    own = [(X(r["frame"]) + wf / 2, Y2(r["iou"])) for r in rows]
    for i in range(0, len(other) - 1, 2):
        d.line([other[i], other[i + 1]], fill=(200, 40, 40), width=2)
    if len(own) > 1:
        d.line(own, fill="black", width=2)
    d.text((L + 6, T2 + 4), "IoU with target (black)   IoU with best other object (red dotted)", fill="black")
    # ---- x axis
    step = max(1, span // 10)
    for f in range(f0, f1 + 1, step):
        d.line([(X(f) + wf / 2, B2), (X(f) + wf / 2, B2 + 4)], fill="black")
        d.text((X(f) - 4, B2 + 8), str(f), fill=(60, 60, 60))
    d.text((R - 40, B2 + 22), "frame", fill=(60, 60, 60))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def thresholds(results_dir: Path, video: str, exp: str):
    p = results_dir / "tcs_log" / f"{video}.json"
    e = json.load(open(p))[exp]
    return e["tau_low"], e["tau_high"], e["expression"]


def pick_auto(results_dir: Path, rows, n):
    picks = []
    fc = results_dir / "tcs_failure_cases.csv"
    if fc.exists():
        for r in csv.DictReader(open(fc)):
            if r["kind"] in ("identity_drift", "track_lost") and (r["video"], r["exp_id"]) not in picks:
                picks.append((r["video"], r["exp_id"]))
    by = {}
    for r in rows:
        by.setdefault((r["video"], r["exp_id"]), []).append(r["c"])
    for k, _ in sorted(by.items(), key=lambda kv: np.mean(kv[1])):          # least stable first
        if k not in picks:
            picks.append(k)
    return picks[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--video")
    ap.add_argument("--exp")
    ap.add_argument("--auto", type=int, default=0, help="plot the N most interesting expressions")
    a = ap.parse_args()
    rd = Path(a.results_dir)
    rows = load_rows(rd)
    targets = pick_auto(rd, rows, a.auto) if a.auto else [(a.video, a.exp)]
    if not a.auto and not (a.video and a.exp):
        raise SystemExit("give --video and --exp, or --auto N")
    for video, exp in targets:
        sub = [r for r in rows if r["video"] == video and r["exp_id"] == exp]
        if not sub:
            print(f"no frames for {video}/{exp}")
            continue
        tl, th, text = thresholds(rd, video, exp)
        out = plot_one(sub, tl, th, f"{video}/{exp}: \"{text}\"", rd / "plots" / f"{video}__{exp}.png")
        print("wrote", out)


if __name__ == "__main__":
    main()
