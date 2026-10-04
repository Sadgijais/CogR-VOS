#!/usr/bin/env python3
"""
Stage V3 — event-driven VLM verification, run as an OFFLINE REPLAY of a finished V2 run.
CPU only, no SAM 2, no GPU. Minutes, not an hour.

Why a replay is exact for this design: the VLM never feeds back into SAM 2, memory or TCS here.
Its only action is to blank a few output masks ("abstain"). So the V2 run already holds everything the
trigger needs (c_t, visible, state, margin per frame), and the schedule of VLM calls can be computed up
front, the calls run in parallel, and the verdicts applied causally afterwards. (A live version that also
feeds verdicts back into memory writes would need a SAM 2 rerun; that is NOT done here and is stated as a limit.)

Pipeline
  1. read <source_results>/tcs_log/*.json and <source_preds>/ from the V2 run
  2. schedule: events (default) | periodic (matched-budget control; --period N or auto) | every_frame
  3. build the questions (identity crops, or a full frame when the mask is empty) and ask the VLM
     (provider oracle = ground-truth ceiling, no API; gemini | openai | anthropic = real VLM)
  4. apply verdicts causally: no_match -> blank this frame and the following ones until the verdict flips
     to match, TCS is HIGH for `recover_high` frames, or `max_abstain` frames pass
  5. write predictions_v3/ (copy of V2 masks + blanked frames) and the logs, then run stage3 on predictions_v3

Outputs in <results_out>/:
  vlm_log.csv               one row per VLM call: event, kind, verdict, confidence, latency, cached, GT diagnostics
  events_per_expression.csv events fired / suppressed / calls / blanked frames per expression
  v3_summary.txt            calls, calls per frame, verdict accuracy vs ground truth, latency, reference costs

Usage (from baseline_v0/):
    python stage_events.py --config config_v3.yaml --vlm oracle                 # free ceiling run
    python stage_events.py --config config_v3.yaml --vlm gemini                 # real VLM (needs GEMINI_API_KEY)
    python stage_events.py --config config_v3.yaml --vlm oracle --action log_only
    python stage_events.py --config config_v3.yaml --vlm oracle --schedule periodic --period auto
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import vlm as V
from common.dataset import DavisLayout
from common.io_utils import load_json, read_binary_mask, read_palette_mask, write_binary_mask
from events import EVENT_TYPES, EventParams, EventTrigger, PeriodicTrigger

MATCH_IOU = 0.50          # a call's verdict is "correct" if it agrees with IoU(pred, GT) >= 0.5
LOG_FIELDS = ["video", "exp_id", "frame", "event", "kind", "c", "verdict", "confidence", "reason",
              "latency_s", "cached", "gt_present", "gt_iou", "correct"]
EXP_FIELDS = ["video", "exp_id", "expression", "frames", "candidates", "calls", "suppressed",
              *EVENT_TYPES, "periodic", "blanked_frames", "false_absence_flags"]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


# ------------------------------------------------------------------ 1-2. schedule
def plan(logs, trigger_factory):
    """logs: [(video, exp_id, entry)] -> (queries, stats). Pure: reads only the logged TCS values."""
    queries, stats = [], {}
    for video, exp_id, e in logs:
        trig = trigger_factory(e)
        for r in sorted(e["frames"], key=lambda r: r["f"]):
            ev = trig.step(r["f"], r["c"], r["visible"], r["state"], r.get("margin"))
            if ev:
                queries.append({"video": video, "exp_id": exp_id, "f": r["f"], "event": ev,
                                "kind": "identity" if r["visible"] else "presence",
                                "expression": e["expression"], "c": r["c"]})
        stats[(video, exp_id)] = {"frames": len(e["frames"]), "candidates": trig.candidates,
                                  "calls": trig.calls, "suppressed": trig.suppressed, "fired": dict(trig.fired)}
    return queries, stats


# ------------------------------------------------------------------ 4. apply verdicts causally
def abstain_frames(frames, verdicts, tau_high, conf_min=0.6, recover_high=5, max_abstain=40, enabled=True):
    """Which frames to blank. verdicts: {frame: {"kind","verdict","confidence"}}.
    Returns (sorted list of blanked frames, list of frames where a presence call said 'visible' = false absence)."""
    blank, false_abs = [], []
    active, start, high_run = False, 0, 0
    for r in sorted(frames, key=lambda r: r["f"]):
        f, v, fresh = r["f"], verdicts.get(r["f"]), False
        if v and v["confidence"] >= conf_min:
            if v["kind"] == "identity" and enabled:
                if v["verdict"] == "no_match":
                    if not active:
                        active, start = True, f
                    high_run, fresh = 0, True
                elif v["verdict"] == "match":
                    active = False
            elif v["kind"] == "presence" and v["verdict"] == "match":
                false_abs.append(f)
        if active and not fresh:
            high_run = high_run + 1 if (r["visible"] and r["c"] >= tau_high) else 0
            if high_run >= recover_high or (f - start) >= max_abstain:
                active = False
        if active:
            blank.append(f)
    return blank, false_abs


# ------------------------------------------------------------------ 3. ask
def _gt_info(layout, video, f, obj_id, pred):
    try:
        gt = read_palette_mask(layout.anno_paths(video)[f], obj_id)
    except Exception:
        return None, None
    return bool(gt.any()), iou(pred, gt)


def answer_query(q, ctx, client, oracle):
    video, exp_id, f = q["video"], q["exp_id"], q["f"]
    fps, layout, src_pred, obj_id = ctx["frame_paths"][video], ctx["layout"], ctx["src_pred"], ctx["obj_id"][(video, exp_id)]
    pdir = src_pred / video / exp_id
    pred = read_binary_mask(pdir / (fps[f].stem + ".png"))
    gt_present, gt_iou = _gt_info(layout, video, f, obj_id, pred)
    row = {"video": video, "exp_id": exp_id, "frame": f, "event": q["event"], "kind": q["kind"],
           "c": round(q["c"], 4), "gt_present": gt_present, "gt_iou": None if gt_iou is None else round(gt_iou, 4)}
    if oracle:
        if gt_present is None:
            v = {"verdict": "unsure", "confidence": 0.0, "reason": "no ground truth"}
        elif q["kind"] == "presence":
            v = {"verdict": "match" if gt_present else "not_visible", "confidence": 1.0, "reason": "oracle"}
        else:
            v = {"verdict": "match" if (gt_present and gt_iou >= MATCH_IOU) else "no_match",
                 "confidence": 1.0, "reason": "oracle"}
        row.update(v, latency_s=0.0, cached=False)
    else:
        img = Image.open(fps[f]).convert("RGB")
        if q["kind"] == "presence":
            images, prompt = [V.small_frame(img)], V.presence_prompt(q["expression"])
        else:
            ref_mask = read_binary_mask(pdir / (fps[0].stem + ".png"))
            ref = V.masked_crop(Image.open(fps[0]).convert("RGB"), ref_mask)
            cur = V.masked_crop(img, pred)
            if ref is None or cur is None:
                row.update({"verdict": "unsure", "confidence": 0.0, "reason": "empty crop", "latency_s": 0.0,
                            "cached": False})
                return _score(row)
            images, prompt = [ref, cur], V.identity_prompt(q["expression"])
        try:
            text, lat, cached = client.ask_timed(images, prompt)
            row.update(V.parse_verdict(text), latency_s=round(lat, 3), cached=cached)
        except Exception as ex:                       # one failed call must not kill the replay
            row.update({"verdict": "unsure", "confidence": 0.0, "reason": f"error: {str(ex)[:120]}",
                        "latency_s": 0.0, "cached": False})
    return _score(row)


def _score(row):
    v, ok = row["verdict"], None
    if row["gt_present"] is not None:
        if row["kind"] == "identity" and v in ("match", "no_match"):
            ok = (v == "match") == (row["gt_iou"] >= MATCH_IOU)
        elif row["kind"] == "presence" and v in ("match", "not_visible"):
            ok = (v == "match") == bool(row["gt_present"])
    row["correct"] = ok
    return row


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v3.yaml")
    ap.add_argument("--vlm", default=None, help="oracle | gemini | openai | anthropic (overrides config)")
    ap.add_argument("--schedule", default=None, help="events | periodic | every_frame")
    ap.add_argument("--period", default=None, help="frames between calls for periodic, or 'auto' = match the events budget")
    ap.add_argument("--action", default=None, help="abstain | log_only")
    ap.add_argument("--video", action="append", default=None)
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    v3 = cfg["v3"]
    provider = args.vlm or v3["vlm"].get("provider", "oracle")
    schedule = args.schedule or v3.get("schedule", "events")
    action = args.action or v3["action"].get("mode", "abstain")
    src_res, src_pred = Path(v3["source_results"]), Path(v3["source_preds"])
    out_pred = Path(args.pred_dir or cfg["paths"]["predictions_out"])
    out_res = Path(args.results_dir or cfg["paths"]["results_out"])
    out_res.mkdir(parents=True, exist_ok=True)

    layout = DavisLayout(root=Path(cfg["dataset"]["root"]), require_480p=False)
    gdir = Path(cfg["paths"]["grounding_out"])
    logs, frame_paths, obj_id, video_frames = [], {}, {}, {}
    files = sorted((src_res / "tcs_log").glob("*.json"))
    if args.video:
        files = [f for f in files if f.stem in set(args.video)]
    if not files:
        raise FileNotFoundError(f"no TCS logs in {src_res}/tcs_log (run the V2 stage first)")
    for lf in files:
        video = lf.stem
        vlog, vg = load_json(lf), load_json(gdir / f"{video}.json")
        frame_paths[video] = layout.frame_paths(video)
        video_frames[video] = len(frame_paths[video])
        for exp_id, e in vlog.items():
            logs.append((video, exp_id, e))
            obj_id[(video, exp_id)] = int(vg[exp_id]["obj_id"])

    evp = dict(v3.get("events", {}))

    def events_factory(e):
        return EventTrigger(EventParams(tau_low=e["tau_low"], tau_high=e["tau_high"], **evp))

    if schedule == "events":
        queries, stats = plan(logs, events_factory)
    else:
        period = 1 if schedule == "every_frame" else args.period or v3.get("period", "auto")
        if str(period) == "auto":
            n_ev = sum(s["calls"] for s in plan(logs, events_factory)[1].values())
            tracked = sum(len(e["frames"]) for _, _, e in logs)
            period = max(1, round(tracked / max(n_ev, 1)))
            print(f"periodic schedule matched to the events budget: {n_ev} calls -> one call every {period} frames")
        period = int(period)
        queries, stats = plan(logs, lambda e: PeriodicTrigger(period))
    print(f"schedule={schedule}  vlm={provider}  action={action}  targets={len(logs)}  VLM calls planned={len(queries)}")

    ctx = {"frame_paths": frame_paths, "layout": layout, "src_pred": src_pred, "obj_id": obj_id}
    oracle = provider == "oracle"
    client = None if oracle else V.make_client({**v3["vlm"], "provider": provider})
    workers = 1 if oracle else int(v3["vlm"].get("workers", 8))
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(lambda q: answer_query(q, ctx, client, oracle), queries))
    wall_s = time.perf_counter() - t0

    # ---- apply verdicts causally and write predictions_v3
    act = v3["action"]
    by_exp: dict = {}
    for r in rows:
        by_exp.setdefault((r["video"], r["exp_id"]), {})[r["frame"]] = r
    exp_rows, n_blank, n_false_abs = [], 0, 0
    for video in sorted({v for v, _, _ in logs}):
        shutil.copytree(src_pred / video, out_pred / video, dirs_exist_ok=True)
    for video, exp_id, e in logs:
        verdicts = {f: {"kind": r["kind"], "verdict": r["verdict"], "confidence": r["confidence"]}
                    for f, r in by_exp.get((video, exp_id), {}).items()}
        blank, false_abs = abstain_frames(
            e["frames"], verdicts, tau_high=e["tau_high"], conf_min=float(act.get("conf_min", 0.6)),
            recover_high=int(act.get("recover_high", 5)), max_abstain=int(act.get("max_abstain", 40)),
            enabled=(action == "abstain"))
        fps = frame_paths[video]
        for f in blank:
            m = read_binary_mask(out_pred / video / exp_id / (fps[f].stem + ".png"))
            write_binary_mask(out_pred / video / exp_id / (fps[f].stem + ".png"), np.zeros_like(m))
        n_blank += len(blank)
        n_false_abs += len(false_abs)
        s = stats[(video, exp_id)]
        exp_rows.append({"video": video, "exp_id": exp_id, "expression": e["expression"], "frames": s["frames"],
                         "candidates": s["candidates"], "calls": s["calls"], "suppressed": s["suppressed"],
                         **{k: s["fired"].get(k, 0) for k in (*EVENT_TYPES, "periodic")},
                         "blanked_frames": len(blank), "false_absence_flags": len(false_abs)})

    with open(out_res / "vlm_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    with open(out_res / "events_per_expression.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXP_FIELDS); w.writeheader(); w.writerows(exp_rows)

    # ---- summary
    tracked = sum(len(e["frames"]) for _, _, e in logs)
    vid_frames = sum(video_frames[v] for v in {v for v, _, _ in logs})
    n = len(rows)
    lat = [r["latency_s"] for r in rows]
    cnt = {k: sum(r["verdict"] == k for r in rows) for k in V.VERDICTS}
    graded = [r for r in rows if r["correct"] is not None]
    acc = (sum(r["correct"] for r in graded) / len(graded)) if graded else float("nan")
    src_t = {}
    tp = src_res / "timing_summary.txt"
    if tp.exists():
        for line in tp.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1); src_t[k.strip()] = v.strip()
    base_s = float(src_t.get("total_inference_s", "nan"))
    ev_tot = {k: sum(s["fired"].get(k, 0) for s in stats.values()) for k in (*EVENT_TYPES, "periodic")}
    lines = [
        f"schedule: {schedule}", f"vlm: {provider}" + ("  (GROUND-TRUTH ORACLE: ceiling, not a result)" if oracle else ""),
        f"model: {getattr(client, 'model', 'n/a')}", f"action: {action}",
        f"targets: {len(logs)}", f"videos: {len({v for v, _, _ in logs})}",
        f"tracked_target_frames: {tracked}", f"video_frames: {vid_frames}",
        f"vlm_calls: {n}", f"calls_per_tracked_frame: {n / max(tracked, 1):.4f}",
        f"calls_per_video_frame: {n / max(vid_frames, 1):.4f}",
        f"reference_cost_every_frame_calls: {tracked}", f"reference_cost_one_shot_calls: {len(logs)}",
        f"candidate_events: {sum(s['candidates'] for s in stats.values())}",
        f"suppressed_by_pacing: {sum(s['suppressed'] for s in stats.values())}",
        "events_fired: " + ", ".join(f"{k}={v}" for k, v in ev_tot.items() if v),
        "verdicts: " + ", ".join(f"{k}={v}" for k, v in cnt.items()),
        f"verdict_accuracy_vs_ground_truth: {acc:.3f}  (n={len(graded)} graded calls, 'unsure' not graded)",
        f"blanked_frames: {n_blank}  ({100.0 * n_blank / max(tracked, 1):.2f}% of tracked frames)",
        f"false_absence_flags: {n_false_abs}",
        f"vlm_latency_sum_s: {sum(lat):.1f}  (what a live sequential system would pay)",
        f"vlm_latency_mean_s: {np.mean(lat) if lat else 0.0:.2f}", f"vlm_cached_calls: {sum(bool(r['cached']) for r in rows)}",
        f"replay_wall_s: {wall_s:.1f}  ({workers} parallel workers)",
        f"source_v2_inference_s: {base_s:.1f}",
        f"estimated_v3_inference_s: {base_s + sum(lat):.1f}  (V2 time + sequential VLM latency)",
    ]
    (out_res / "v3_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Stage V3 done. Predictions under {out_pred}/  -> next: python stage3_evaluate.py --config {args.config}")


if __name__ == "__main__":
    main()
