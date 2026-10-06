#!/usr/bin/env python3
"""
Task 5 ablation table: V0 .. V4 side by side, built ONLY from files the stages already wrote (nothing typed by hand).
CPU only, runs in a second.

  python tools/make_ablation_table.py --v2-dir results_v2_passive \
      --row V0 results         recovery_v0 \
      --row V1 results_v1_final recovery_v1 \
      --row V2 results_v2_passive recovery_v2_passive \
      --row V3 results_v3_gemini_events_abstain recovery_v3_gemini \
      --row V4 results_v4_vlm  recovery_v4_vlm

Each --row is NAME RESULTS_DIR RECOVERY_DIR (use - when there is no recovery folder). From RESULTS_DIR it reads
summary.csv (J, F, J&F), timing_summary.txt (V0-V2), v3_summary.txt (V3: VLM calls and estimated time) or reid_summary.txt
(V4: VLM calls and live-rerun time). From RECOVERY_DIR it reads recovery_summary.json (tools/recovery_metrics.py).

V4 time = V2 total time - V2 time on the videos V4 re-ran + V4 time on those videos (everything else is copied from V2).
Writes ablation_table.md and ablation_table.csv in --out-dir (default: current folder).
"""
import argparse
import csv
import json
from pathlib import Path


def read_kv(p):
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def num(s, default=None):
    try:
        return float(str(s).split()[0])
    except (ValueError, IndexError, TypeError):
        return default


def read_metrics(d):
    out = {}
    p = d / "summary.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            out[r["metric"]] = r["value"]
    return {"J": num(out.get("mean_J")), "F": num(out.get("mean_F")), "JF": num(out.get("mean_JF")),
            "n_expr": num(out.get("n_expressions_scored")), "n_videos": num(out.get("n_videos_scored"))}


def timing_rows(d):
    p = d / "timing_per_video.csv"
    return {r["video"]: float(r["inference_s"]) for r in csv.DictReader(open(p))} if p.exists() else {}


def cost_columns(d, v2_dir, frames_tracked, frames_video):
    """-> dict(time_s, calls, calls_per_tracked, calls_per_video_frame, time_note)"""
    v3 = read_kv(d / "v3_summary.txt")
    reid = read_kv(d / "reid_summary.txt")
    t = read_kv(d / "timing_summary.txt")
    if v3:
        return {"time_s": num(v3.get("estimated_v3_inference_s")), "calls": num(v3.get("vlm_calls"), 0),
                "calls_per_tracked": num(v3.get("calls_per_tracked_frame")), "calls_per_video": num(v3.get("calls_per_video_frame")),
                "tracked": num(v3.get("tracked_target_frames")), "frames": num(v3.get("video_frames")),
                "note": "V2 time + sequential VLM latency"}
    if reid:
        live = timing_rows(d)
        v2 = timing_rows(v2_dir) if v2_dir else {}
        v2_total = num(read_kv(v2_dir / "timing_summary.txt").get("total_inference_s")) if v2_dir else None
        if v2_total is None or not live or not v2:
            return {"time_s": None, "calls": num(reid.get("vlm_calls"), 0), "note": "needs --v2-dir with timing_per_video.csv"}
        removed = sum(v2.get(v, 0.0) for v in live)
        calls = num(reid.get("vlm_calls"), 0)
        return {"time_s": v2_total - removed + sum(live.values()), "calls": calls,
                "calls_per_tracked": calls / frames_tracked if frames_tracked else None,
                "calls_per_video": calls / frames_video if frames_video else None,
                "note": f"V2 total - V2 on {len(live)} re-run videos + V4 on them"}
    return {"time_s": num(t.get("total_inference_s")), "calls": 0.0, "calls_per_tracked": 0.0, "calls_per_video": 0.0,
            "note": "measured"}


def read_recovery(d):
    p = d / "recovery_summary.json" if d else None
    if not p or not p.exists():
        return None
    s = json.load(open(p))
    return {"events": s["reappearance_events"], "recovered": s["recovered_in_window"], "lat": s.get("latency_mean"),
            "loss": s["loss_episodes"], "loss_rec": s["loss_recovered"], "nre": s["nre_frames"], "visible": s["visible_frames"],
            "falsep": s["false_presence_frames"], "absent": s["absent_frames"]}


def fmt(x, nd=2, dash="n/a"):
    return dash if x is None else f"{x:.{nd}f}"


def build(rows, v2_dir):
    # frame counts for the per-frame call rates come from the V3 summary if any row has one (same videos, same expressions)
    tracked = frames = None
    for _, d, _ in rows:
        v3 = read_kv(d / "v3_summary.txt")
        if v3:
            tracked, frames = num(v3.get("tracked_target_frames")), num(v3.get("video_frames"))
    out = []
    for name, d, rdir in rows:
        m = read_metrics(d)
        c = cost_columns(d, v2_dir, tracked, frames)
        r = read_recovery(rdir)
        out.append({"version": name, **m, **{k: c.get(k) for k in ("time_s", "calls", "calls_per_tracked", "calls_per_video", "note")},
                    "recovery": r})
    return out


def write(rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    head = ["Version", "J", "F", "J&F", "Inference time (s)", "VLM calls", "VLM calls / tracked frame",
            "Recovery rate (events)", "Recovery latency (frames)", "Loss episodes recovered", "NRE", "False presence"]
    lines, csvrows = [], []
    for r in rows:
        rec = r["recovery"]
        if rec:
            rr = f"{rec['recovered']} of {rec['events']}"
            lat = fmt(rec["lat"], 1)
            lo = f"{rec['loss_rec']} of {rec['loss']}"
            nre = f"{100 * rec['nre'] / rec['visible']:.1f}%" if rec["visible"] else "n/a"
            fp = f"{100 * rec['falsep'] / rec['absent']:.1f}%" if rec["absent"] else "n/a"
        else:
            rr = lat = lo = nre = fp = "n/a"
        cells = [r["version"], fmt(r["J"]), fmt(r["F"]), fmt(r["JF"]), fmt(r["time_s"], 1), fmt(r["calls"], 0),
                 fmt(r["calls_per_tracked"], 4), rr, lat, lo, nre, fp]
        csvrows.append(cells)
        lines.append("| " + " | ".join(cells) + " |")
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head), *lines, "",
          "Notes:", *[f"- {r['version']} time: {r['note']}" for r in rows],
          "- Recovery = back at IoU >= 0.5 within 10 frames of the first visible frame. Always read it with the event count: "
          "Ref-DAVIS17 has only a handful of reappearance events, so these are anecdotes, not statistics.",
          "- NRE = visible but empty mask; false presence = absent but mask not empty. One run each, no significance test.",
          "- V4 VLM answers served from the disk cache cost no live waiting time here, so the V4 time is a lower bound "
          "when answers were cached.", ""]
    (out_dir / "ablation_table.md").write_text("\n".join(md))
    with open(out_dir / "ablation_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(csvrows)
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", nargs=3, action="append", metavar=("NAME", "RESULTS_DIR", "RECOVERY_DIR"), required=True)
    ap.add_argument("--v2-dir", default=None, help="V2 passive results folder (needed to compute the V4 time)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--time-from", nargs=2, action="append", default=[], metavar=("NAME", "DIR"),
                    help="take this row's time from DIR/timing_summary.txt (V0: results_timed)")
    a = ap.parse_args()
    rows = [(n, Path(d), None if r == "-" else Path(r)) for n, d, r in a.row]
    out = build(rows, Path(a.v2_dir) if a.v2_dir else None)
    for name, d in a.time_from:
        t = num(read_kv(Path(d) / "timing_summary.txt").get("total_inference_s"))
        for r in out:
            if r["version"] == name and t is not None:
                r["time_s"], r["note"] = t, f"measured in {d}"
    print(write(out, Path(a.out_dir)))


if __name__ == "__main__":
    main()
