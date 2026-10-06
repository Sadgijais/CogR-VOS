#!/usr/bin/env python3
"""
Recovery metrics for ANY version's predictions (V0 to V4). CPU only: no model, no GPU, no network.

It reads predicted masks and ground-truth masks and answers: when the target vanished and came back, or the tracker
lost it while it was still visible, did the prediction get it back, and how fast?

Definitions (fixed in V4_PLAN.md before any V4 run):
  disappearance         the true target is absent for >= 2 consecutive frames AFTER it was visible at least once
  reappearance event    a disappearance that ends (the target is visible again before the video ends)
  Recovery Rate         reappearance events where some visible frame within `window` frames of the first visible
                        frame has IoU >= recovery_iou, divided by all reappearance events (counts are always printed)
  Recovery Latency      frames from the first visible frame to the first frame with IoU >= recovery_iou
                        (0 = instantly). Mean and median over events recovered inside the window
  loss episode          >= loss_len consecutive VISIBLE frames with IoU < loss_iou on the target
                        (lost it, or on the wrong object). Recovered if IoU >= recovery_iou returns on a later visible frame.
                        Latency counted from the first frame of the episode
  NRE                   frames where the target is visible but the predicted mask is empty
  false presence        frames where the target is absent but the predicted mask is not empty
Frame-level counts (NRE, false presence, loss episodes) skip the first and last frame of each track when
evaluation.exclude_first_last is true (the DAVIS convention used by stage3_evaluate.py). Events use every frame.
A missing prediction PNG counts as an empty mask, exactly as in stage3_evaluate.py.

Usage (from baseline_v0/):
    python tools/recovery_metrics.py --config config_v2_passive.yaml --pred-dir predictions_v2_passive \
        --out-dir recovery_v2_passive --label v2_passive
    python tools/recovery_metrics.py --config config_v2_passive.yaml --pred-dir predictions_v2_passive \
        --out-dir /tmp/rec_test --video india --video kite-surf
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EVENT_FIELDS = ["label", "kind", "video", "exp_id", "start_frame", "end_frame", "length", "first_visible_frame",
                "iou_at_return", "recovered_in_window", "latency_frames", "eventual_latency_frames", "recovered"]


@dataclass
class Params:
    min_absent: int = 2            # frames the target must be gone to count as a disappearance
    window: int = 10               # frames after the first visible frame in which it must be back
    recovery_iou: float = 0.50
    loss_iou: float = 0.10
    loss_len: int = 5
    exclude_first_last: bool = True


# ------------------------------------------------------------------ pure logic (unit-tested, no files)
def _first_good(present, iou, start, last, thr):
    """First frame k in [start, last] with the target visible and iou[k] >= thr, scanning only while the target
    stays visible. Returns k or None."""
    for k in range(start, last + 1):
        if not present[k]:
            return None
        if iou[k] >= thr:
            return k
    return None


def reappearance_events(present, iou, p: Params):
    """-> (events, vanish_without_return). present[k]: target visible in frame k. iou[k]: IoU of the prediction
    with the true mask (only used where present[k])."""
    n = len(present)
    events, no_return = [], 0
    seen, i = False, 0
    while i < n:
        if present[i]:
            seen = True
            i += 1
            continue
        j = i
        while j < n and not present[j]:
            j += 1
        # absent run is [i, j-1]
        if seen and (j - i) >= p.min_absent:
            if j >= n:
                no_return += 1
            else:
                k_win = _first_good(present, iou, j, min(n - 1, j + p.window), p.recovery_iou)
                k_any = _first_good(present, iou, j, n - 1, p.recovery_iou)
                events.append({
                    "start_frame": i, "end_frame": j - 1, "length": j - i, "first_visible_frame": j,
                    "iou_at_return": round(float(iou[j]), 4),
                    "recovered_in_window": k_win is not None,
                    "latency_frames": None if k_win is None else k_win - j,
                    "eventual_latency_frames": None if k_any is None else k_any - j,
                })
        i = j
    return events, no_return


def loss_episodes(present, iou, p: Params):
    n = len(present)
    lo, hi = (1, n - 1) if (p.exclude_first_last and n > 2) else (0, n)
    eps, k = [], lo
    while k < hi:
        if present[k] and iou[k] < p.loss_iou:
            e = k
            while e + 1 < hi and present[e + 1] and iou[e + 1] < p.loss_iou:
                e += 1
            if (e - k + 1) >= p.loss_len:
                back = next((m for m in range(e + 1, n) if present[m] and iou[m] >= p.recovery_iou), None)
                eps.append({"start_frame": k, "end_frame": e, "length": e - k + 1, "recovered": back is not None,
                            "latency_frames": None if back is None else back - k})
            k = e + 1
        else:
            k += 1
    return eps


def frame_counts(present, pred_empty, p: Params):
    n = len(present)
    lo, hi = (1, n - 1) if (p.exclude_first_last and n > 2) else (0, n)
    c = {"scored_frames": hi - lo, "visible_frames": 0, "nre_frames": 0, "absent_frames": 0, "false_presence_frames": 0}
    for k in range(lo, hi):
        if present[k]:
            c["visible_frames"] += 1
            c["nre_frames"] += int(pred_empty[k])
        else:
            c["absent_frames"] += 1
            c["false_presence_frames"] += int(not pred_empty[k])
    return c


def analyze_track(present, iou, pred_empty, p: Params):
    if not (len(present) == len(iou) == len(pred_empty)):
        raise ValueError("present, iou and pred_empty must have the same length")
    ev, no_return = reappearance_events(present, iou, p)
    return {"events": ev, "vanish_without_return": no_return, "loss": loss_episodes(present, iou, p),
            "counts": frame_counts(present, pred_empty, p)}


def _ratio(a, b):
    return f"{a} of {b} ({100.0 * a / b:.1f}%)" if b else "n/a (no events)"


def _stats(xs):
    return "n/a" if not xs else f"mean {statistics.mean(xs):.2f}, median {statistics.median(xs):.1f} (n={len(xs)})"


def summarize(tracks, p: Params, label="", skipped=0):
    """tracks: list of (video, exp_id, analyze_track result). -> (summary dict, text)."""
    ev = [(v, x, e) for v, x, r in tracks for e in r["events"]]
    ls = [(v, x, e) for v, x, r in tracks for e in r["loss"]]
    tot = {k: sum(r["counts"][k] for _, _, r in tracks)
           for k in ("scored_frames", "visible_frames", "nre_frames", "absent_frames", "false_presence_frames")}
    n_rec = sum(e["recovered_in_window"] for _, _, e in ev)
    lat = [e["latency_frames"] for _, _, e in ev if e["recovered_in_window"]]
    later = [e for _, _, e in ev if (not e["recovered_in_window"]) and e["eventual_latency_frames"] is not None]
    never = [e for _, _, e in ev if e["eventual_latency_frames"] is None]
    l_rec = [e for _, _, e in ls if e["recovered"]]
    s = {
        "label": label, "expressions_analyzed": len(tracks), "expressions_skipped_no_predictions": skipped,
        "expressions_with_reappearance": sum(1 for _, _, r in tracks if r["events"]),
        "expressions_with_vanish_no_return": sum(1 for _, _, r in tracks if r["vanish_without_return"]),
        "reappearance_events": len(ev), "recovered_in_window": n_rec, "recovered_later": len(later),
        "never_recovered": len(never),
        "latency_mean": statistics.mean(lat) if lat else None, "latency_median": statistics.median(lat) if lat else None,
        "loss_episodes": len(ls), "loss_recovered": len(l_rec), "loss_never_recovered": len(ls) - len(l_rec),
        **tot,
    }
    lines = [
        f"label: {label}",
        f"expressions_analyzed: {len(tracks)}   skipped (no prediction folder): {skipped}",
        f"settings: min_absent {p.min_absent}, window {p.window}, recovery_iou {p.recovery_iou}, loss_iou {p.loss_iou}, "
        f"loss_len {p.loss_len}, exclude_first_last {p.exclude_first_last}",
        "",
        f"expressions with a disappearance that returns: {s['expressions_with_reappearance']}   "
        f"with a disappearance that never returns: {s['expressions_with_vanish_no_return']}",
        f"reappearance_events: {len(ev)}",
        f"recovery_rate (back at IoU >= {p.recovery_iou} within {p.window} frames): {_ratio(n_rec, len(ev))}",
        f"recovery_latency_frames (events recovered in the window): {_stats(lat)}",
        f"recovered later than the window: {len(later)}    never recovered: {len(never)}",
        "",
        f"loss_episodes (>= {p.loss_len} visible frames at IoU < {p.loss_iou}): {len(ls)}",
        f"loss_episodes_recovered (IoU >= {p.recovery_iou} returns later): {_ratio(len(l_rec), len(ls))}",
        f"loss_recovery_latency_frames: {_stats([e['latency_frames'] for e in l_rec])}",
        f"loss_episodes_never_recovered: {len(ls) - len(l_rec)}   (many of these are frame-0 grounding failures)",
        "",
        f"scored_frames: {tot['scored_frames']}   visible: {tot['visible_frames']}   absent: {tot['absent_frames']}",
        f"NRE (visible but empty mask): {_ratio(tot['nre_frames'], tot['visible_frames'])}",
        f"false_presence (absent but mask not empty): {_ratio(tot['false_presence_frames'], tot['absent_frames'])}",
        "",
        "Small samples: report these counts next to every rate. A rate over 3 events is an anecdote, not a statistic.",
    ]
    return s, "\n".join(lines) + "\n"


# ------------------------------------------------------------------ files
def load_track(layout, video, exp_id, obj_id, pred_dir: Path):
    """Per frame: target visible?, IoU(pred, gt), prediction empty? Missing prediction PNG = empty mask."""
    import numpy as np
    from common.io_utils import read_binary_mask, read_palette_mask

    present, ious, empty = [], [], []
    for a in layout.anno_paths(video):
        gt = read_palette_mask(a, obj_id)
        pp = pred_dir / video / exp_id / (a.stem + ".png")
        pred = read_binary_mask(pp) if pp.exists() else np.zeros_like(gt)
        if pred.shape != gt.shape:
            raise ValueError(f"{pp}: prediction shape {pred.shape} differs from ground truth {gt.shape}")
        u = np.logical_or(pred, gt).sum()
        present.append(bool(gt.any()))
        ious.append(1.0 if u == 0 else float(np.logical_and(pred, gt).sum() / u))
        empty.append(not bool(pred.any()))
    return present, ious, empty


def main():
    import yaml
    from common.dataset import DavisLayout
    from common.io_utils import load_json

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--pred-dir", required=True, help="folder with <video>/<exp_id>/<frame>.png predictions")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--video", action="append", default=None)
    ap.add_argument("--window", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    fa, ev = cfg.get("failure_analysis", {}), cfg.get("evaluation", {})
    p = Params(window=args.window or int(fa.get("recovery_window", 10)),
               recovery_iou=float(fa.get("recovery_iou", 0.5)), loss_iou=float(fa.get("drift_low", 0.10)),
               exclude_first_last=bool(ev.get("exclude_first_last", True)))
    layout = DavisLayout(root=Path(cfg["dataset"]["root"]), require_480p=False)
    gdir, pred_dir, out_dir = Path(cfg["paths"]["grounding_out"]), Path(args.pred_dir), Path(args.out_dir)
    label = args.label or pred_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks, skipped, rows = [], 0, []
    for gf in sorted(gdir.glob("*.json")):
        video = gf.stem
        if args.video and video not in args.video:
            continue
        for exp_id, e in load_json(gf).items():
            if not (pred_dir / video / exp_id).exists():
                skipped += 1
                print(f"skip {video}/{exp_id}: no prediction folder")
                continue
            present, ious, empty = load_track(layout, video, exp_id, int(e["obj_id"]), pred_dir)
            r = analyze_track(present, ious, empty, p)
            tracks.append((video, exp_id, r))
            for x in r["events"]:
                rows.append({"label": label, "kind": "reappearance", "video": video, "exp_id": exp_id,
                             "recovered": x["recovered_in_window"], **x})
            for x in r["loss"]:
                rows.append({"label": label, "kind": "loss_episode", "video": video, "exp_id": exp_id, **x})
    if not tracks:
        raise SystemExit(f"nothing to analyze: no predictions found under {pred_dir}/ for the grounding cache {gdir}/")

    s, text = summarize(tracks, p, label, skipped)
    with open(out_dir / "recovery_events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out_dir / "recovery_summary.txt").write_text(text)
    (out_dir / "recovery_summary.json").write_text(json.dumps(s, indent=2))
    print(text)
    print(f"Wrote {out_dir}/recovery_summary.txt, recovery_summary.json, recovery_events.csv")


if __name__ == "__main__":
    main()
