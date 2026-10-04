#!/usr/bin/env python3
"""
Stage 2 (V2) — SAM 2 propagation + Semantic-Temporal Target Memory (V1) + Tracklet Coherence Score.

Same prompts, same model, same memory settings as V1 (stage2_sttm.py). The additions:

  * after every frame, a TrackletCoherence monitor scores the prediction (appearance, motion,
    mask and semantic consistency -> c_t in [0,1]) and puts the frame in a band:
    HIGH = stable, MEDIUM / LOW = uncertain. Absent targets are flagged separately, not as drift.
  * gate_writes (config v2.gate_writes, or --no-gate to force it off):
        false = PASSIVE. TCS only measures. Masks must equal V1's exactly (check this!).
        true  = a frame is written to memory ONLY if its band is HIGH. This is the V2 action.
  * No VLM, no re-identification.

Writes, next to the PNG predictions:
  <results>/tcs_log/<video>.json        per-frame terms, c_t, band, state, memory decision
  <results>/tcs_per_expression.csv      band shares, absent frames, writes, blocked_tcs_uncertain
  <results>/timing_per_video.csv        wall time, ms/frame, peak VRAM, STTM and TCS overhead
  <results>/timing_summary.txt
  <results>/sttm_per_expression.csv, <results>/sttm_log/<video>.json   (same as V1)

Usage (from baseline_v0/):
    python stage2_tcs.py --config config_v2.yaml
    python stage2_tcs.py --config config_v2.yaml --video blackswan --video india
    python stage2_tcs.py --config config_v2_final.yaml --no-gate --pred-dir predictions_v2_passive --results-dir results_v2_passive
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common.dataset import DavisLayout
from common.io_utils import load_json, write_binary_mask
from sttm import STTMParams, TargetMemory
from sttm_sam2 import object_score, sync_bank
from tcs import TCSParams, TrackletCoherence

TIMING_FIELDS = ["video", "n_frames", "n_objects", "init_s", "propagate_s", "inference_s",
                 "sttm_overhead_s", "tcs_overhead_s", "ms_per_frame", "peak_alloc_mb", "peak_reserved_mb"]
STTM_FIELDS = ["video", "exp_id", "expression", "frames", "writes", "merges", "evictions",
               "blocked_not_visible", "blocked_truncated", "blocked_low_similarity",
               "blocked_paced", "blocked_not_better", "blocked_tcs_uncertain",
               "n_absent_events", "n_reappeared_events", "final_bank_frames", "min_margin",
               "missing_frames"]
TCS_FIELDS = ["video", "exp_id", "expression", "frames", "n_high", "n_medium", "n_low",
              "n_absent_frames", "pct_high", "mean_c", "writes", "blocked_tcs_uncertain"]


def box_to_prompt(box):
    x1, y1, x2, y2, _ = box
    return [x1, y1, x2, y2]


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _frame_hw(p: Path):
    with Image.open(p) as im:
        w, h = im.size
    return (h, w)


def mask_confidence(logits_np: np.ndarray, mask: np.ndarray) -> float:
    """Stand-in for SAM2's IoU prediction (not stored in the video state):
    mean sigmoid(logit) over the predicted-mask pixels = how certain the mask is."""
    if not mask.any():
        return 0.0
    return float((1.0 / (1.0 + np.exp(-logits_np[mask]))).mean())


def _round_terms(terms: dict) -> dict:
    return {k: (None if v is None else round(float(v), 4)) for k, v in terms.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v2.yaml")
    ap.add_argument("--video", action="append", default=None, help="only these videos (repeatable)")
    ap.add_argument("--no-gate", action="store_true", help="passive TCS: measure only, never block a write")
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    v1 = cfg.get("v1", {})
    v2 = cfg.get("v2", {})
    params = STTMParams(**v1.get("params", {}))
    tcs_params = TCSParams(**v2.get("tcs", {}))
    gate = bool(v2.get("gate_writes", False)) and not args.no_gate

    layout = DavisLayout(root=Path(cfg["dataset"]["root"]), require_480p=cfg["evaluation"]["require_480p"])
    grounding_dir = Path(cfg["paths"]["grounding_out"])
    emb_dir = Path(v1.get("embeddings_out", "embeddings/"))
    pred_dir = Path(args.pred_dir or cfg["paths"]["predictions_out"])
    results_dir = Path(args.results_dir or cfg["paths"]["results_out"])
    (results_dir / "sttm_log").mkdir(parents=True, exist_ok=True)
    (results_dir / "tcs_log").mkdir(parents=True, exist_ok=True)

    files = sorted(grounding_dir.glob("*.json"))
    if args.video:
        files = [f for f in files if f.stem in set(args.video)]
    if not files:
        raise FileNotFoundError(f"no grounding cache in {grounding_dir}/ (or --video matched nothing)")

    import torch
    from sam2.build_sam import build_sam2_video_predictor

    print(f"STTM ON; params={params}")
    print(f"TCS ON; gate_writes={'ON (write only on HIGH frames)' if gate else 'OFF (passive, measure only)'}; {tcs_params}")
    print(f"Loading SAM2 ({cfg['propagation']['model_cfg']}) ...")
    predictor = build_sam2_video_predictor(
        cfg["propagation"]["model_cfg"], cfg["propagation"]["checkpoint"],
        device=cfg["propagation"]["device"])
    from common.embed import ClipCropEmbedder, CropEmbedder
    crop_emb = CropEmbedder(device=cfg["propagation"]["device"])
    clip_crop = ClipCropEmbedder(device=cfg["propagation"]["device"])

    timing_rows, sttm_rows, tcs_rows = [], [], []

    for gfile in files:
        video = gfile.stem
        vg = load_json(gfile)
        frame_paths = layout.frame_paths(video)
        n_frames = len(frame_paths)
        H, W = _frame_hw(frame_paths[0])
        print(f"Propagating {video}: {len(vg)} expression(s), {n_frames} frames")

        cache = np.load(emb_dir / f"{video}.npz")

        torch.cuda.reset_peak_memory_stats()
        _sync()
        t_start = time.perf_counter()
        state = predictor.init_state(
            video_path=str(layout.jpeg_dir / video),
            offload_video_to_cpu=cfg["propagation"]["offload_video_to_cpu"],
            offload_state_to_cpu=cfg["propagation"]["offload_state_to_cpu"])
        _sync()
        init_s = time.perf_counter() - t_start

        exp_ids = list(vg.keys())
        sam_obj_id_of = {}
        for i, exp_id in enumerate(exp_ids, start=1):
            entry = vg[exp_id]
            if not entry["boxes"]:
                print(f"  WARNING: {video}/{exp_id} has no grounding boxes — empty mask (V0 behaviour)")
                sam_obj_id_of[exp_id] = None
                continue
            sam_obj_id_of[exp_id] = i
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=i,
                box=np.array(box_to_prompt(entry["boxes"][0]), dtype=np.float32))

        # one memory + one coherence monitor per expression that has a SAM2 object; obj_idx = obj_id - 1
        mems: dict[int, TargetMemory] = {}
        tcss: dict[int, TrackletCoherence] = {}
        logs: dict[int, list] = {}
        frame_logs: dict[int, list] = {}
        tcs_logs: dict[int, list] = {}
        missing_counts: dict[int, int] = {}
        for exp_id in exp_ids:
            oid = sam_obj_id_of[exp_id]
            if oid is None:
                continue
            boxes_emb = cache[f"{exp_id}__boxes"]
            mems[oid - 1] = TargetMemory(
                params, (H, W), cache[f"{exp_id}__text"],
                boxes_emb[1:] if len(boxes_emb) > 1 else None)   # [0] is the target's own box
            tcss[oid - 1] = TrackletCoherence(tcs_params)
            logs[oid - 1] = []
            frame_logs[oid - 1] = []
            tcs_logs[oid - 1] = []
            missing_counts[oid - 1] = 0

        _sync()
        t_prop = time.perf_counter()
        overhead = 0.0
        tcs_overhead = 0.0
        video_masks: dict[int, dict[int, np.ndarray]] = {}
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            img = None
            for k, (obj_id, logits) in enumerate(zip(obj_ids, mask_logits)):
                lg = logits.squeeze()
                mask = (lg > 0.0).cpu().numpy()
                video_masks.setdefault(obj_id, {})[frame_idx] = mask
                mem = mems.get(k)
                if mem is None:
                    continue
                _sync()
                t0 = time.perf_counter()
                if img is None:
                    img = Image.open(frame_paths[frame_idx]).convert("RGB")
                emb = crop_emb.embed_masked(img, mask) if mask.any() else None
                if frame_idx == 0:
                    if emb is None:
                        del mems[k]          # empty prompt mask: no memory, plain V0 behaviour
                        tcss.pop(k, None)
                    else:
                        mem.set_anchor(0, mask, emb)
                        cemb0 = clip_crop.embed_masked(img, mask)
                        tcss[k].set_anchor(mask, float(cemb0 @ mem.semantic))
                else:
                    od = state["output_dict_per_obj"][k]
                    out = od["non_cond_frame_outputs"].get(frame_idx) or od["cond_frame_outputs"].get(frame_idx)
                    osc = object_score(out)
                    conf = mask_confidence(lg.float().cpu().numpy(), mask)
                    # ---- TCS first (reads memory state of the previous frame), then the memory update
                    _sync()
                    t1 = time.perf_counter()
                    cemb = clip_crop.embed_masked(img, mask) if mask.any() else None
                    sem = float(cemb @ mem.semantic) if cemb is not None else None
                    res = tcss[k].score(frame_idx, mem, mask, osc, conf, emb, sem)
                    _sync()
                    tcs_overhead += time.perf_counter() - t1
                    allow = (not gate) or res["band"] == "HIGH"
                    dec = mem.update(frame_idx, mask, emb, osc, conf, allow_write=allow)
                    frame_logs[k].append({"f": frame_idx, "obj_score": round(osc, 2), "conf": round(conf, 3),
                                          "sim": None if dec.get("sim") is None else round(dec["sim"], 3),
                                          "result": dec.get("action") or dec.get("blocked_by")})
                    tcs_logs[k].append({"f": frame_idx, "c": round(res["c"], 4), "band": res["band"],
                                        "state": res["state"], "visible": res["visible"],
                                        "terms": _round_terms(res["terms"]),
                                        "margin": None if res["margin"] is None else round(res["margin"], 4),
                                        "result": dec.get("action") or dec.get("blocked_by")})
                    if dec.get("written"):
                        r = sync_bank(state, k, mem.read_frames())
                        missing_counts[k] += len(r["missing"])
                        logs[k].append({"frame": frame_idx, "action": dec["action"],
                                        "bank": mem.read_frames(), **{kk: r[kk] for kk in ("promoted", "demoted")}})
                _sync()
                overhead += time.perf_counter() - t0
        _sync()
        propagate_s = time.perf_counter() - t_prop
        inference_s = time.perf_counter() - t_start
        peak_a = torch.cuda.max_memory_allocated() / 2**20
        peak_r = torch.cuda.max_memory_reserved() / 2**20
        timing_rows.append({
            "video": video, "n_frames": n_frames,
            "n_objects": sum(v is not None for v in sam_obj_id_of.values()),
            "init_s": round(init_s, 3), "propagate_s": round(propagate_s, 3),
            "inference_s": round(inference_s, 3), "sttm_overhead_s": round(overhead, 3),
            "tcs_overhead_s": round(tcs_overhead, 3),
            "ms_per_frame": round(propagate_s / n_frames * 1000.0, 2),
            "peak_alloc_mb": round(peak_a, 1), "peak_reserved_mb": round(peak_r, 1)})
        print(f"  timing: {inference_s:.1f}s total ({overhead:.1f}s memory+TCS overhead, "
              f"of which TCS {tcs_overhead:.1f}s), {propagate_s / n_frames * 1000.0:.0f} ms/frame, "
              f"peak VRAM {peak_a:.0f} MB")

        # ---- write masks (not timed)
        for exp_id in exp_ids:
            oid = sam_obj_id_of[exp_id]
            for fi, fp in enumerate(frame_paths):
                if oid is None:
                    m = np.zeros((H, W), bool)
                else:
                    m = video_masks.get(oid, {}).get(fi, np.zeros((H, W), bool))
                write_binary_mask(pred_dir / video / exp_id / (fp.stem + ".png"), m)

        # ---- STTM + TCS logs
        vlog, tlog = {}, {}
        for exp_id in exp_ids:
            oid = sam_obj_id_of[exp_id]
            if oid is None or (oid - 1) not in mems:
                continue
            k = oid - 1
            s = mems[k].summary()
            st = s["stats"]
            ev = s["events"]
            margin = s["context"].get("margin")
            vlog[exp_id] = {"expression": vg[exp_id]["expression"], "summary": {
                kk: (vv if kk != "velocity" else list(vv)) for kk, vv in s.items() if kk != "events"},
                "events": ev, "writes": logs[k], "frames": frame_logs[k]}
            sttm_rows.append({
                "video": video, "exp_id": exp_id, "expression": vg[exp_id]["expression"],
                "frames": st["frames"], "writes": st["writes"], "merges": st["merges"],
                "evictions": st["evictions"],
                **{f"blocked_{n}": st["blocked"][n] for n in
                   ("not_visible", "truncated", "low_similarity", "paced", "not_better", "tcs_uncertain")},
                "n_absent_events": sum(e["event"] == "absent" for e in ev),
                "n_reappeared_events": sum(e["event"] == "reappeared" for e in ev),
                "final_bank_frames": " ".join(map(str, s["bank_frames"])),
                "min_margin": "" if margin is None else round(margin, 3),
                "missing_frames": missing_counts[k]})
            tc = tcss[k].counts
            n_t = len(tcs_logs[k])
            tlog[exp_id] = {"expression": vg[exp_id]["expression"], "gate_writes": gate,
                            "tau_low": tcs_params.tau_low, "tau_high": tcs_params.tau_high,
                            "frames": tcs_logs[k]}
            tcs_rows.append({
                "video": video, "exp_id": exp_id, "expression": vg[exp_id]["expression"],
                "frames": n_t, "n_high": tc["HIGH"], "n_medium": tc["MEDIUM"], "n_low": tc["LOW"],
                "n_absent_frames": tc["absent"],
                "pct_high": round(100.0 * tc["HIGH"] / max(n_t, 1), 1),
                "mean_c": round(float(np.mean([f["c"] for f in tcs_logs[k]])) if n_t else 0.0, 4),
                "writes": st["writes"], "blocked_tcs_uncertain": st["blocked"]["tcs_uncertain"]})
        if vlog:
            (results_dir / "sttm_log" / f"{video}.json").write_text(json.dumps(vlog, indent=2, default=float))
        if tlog:
            (results_dir / "tcs_log" / f"{video}.json").write_text(json.dumps(tlog, indent=1, default=float))

        predictor.reset_state(state)

    del predictor
    torch.cuda.empty_cache()

    with open(results_dir / "timing_per_video.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TIMING_FIELDS); w.writeheader(); w.writerows(timing_rows)
    if sttm_rows:
        with open(results_dir / "sttm_per_expression.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=STTM_FIELDS); w.writeheader(); w.writerows(sttm_rows)
    if tcs_rows:
        with open(results_dir / "tcs_per_expression.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TCS_FIELDS); w.writeheader(); w.writerows(tcs_rows)
    frames = sum(r["n_frames"] for r in timing_rows)
    prop = sum(r["propagate_s"] for r in timing_rows)
    lines = [f"sttm: on", f"tcs: on", f"gate_writes: {'on' if gate else 'off'}",
             f"videos: {len(timing_rows)}", f"frames: {frames}",
             f"total_inference_s: {sum(r['inference_s'] for r in timing_rows):.1f}",
             f"total_sttm_and_tcs_overhead_s: {sum(r['sttm_overhead_s'] for r in timing_rows):.1f}",
             f"total_tcs_overhead_s: {sum(r['tcs_overhead_s'] for r in timing_rows):.1f}",
             f"overall_ms_per_frame: {prop / frames * 1000.0:.1f}",
             f"median_video_ms_per_frame: {statistics.median(r['ms_per_frame'] for r in timing_rows):.1f}",
             f"max_peak_alloc_mb: {max(r['peak_alloc_mb'] for r in timing_rows):.0f}",
             f"max_peak_reserved_mb: {max(r['peak_reserved_mb'] for r in timing_rows):.0f}"]
    (results_dir / "timing_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Stage 2 (V2) done. Predictions under {pred_dir}/")


if __name__ == "__main__":
    main()
