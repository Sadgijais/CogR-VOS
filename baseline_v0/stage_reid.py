#!/usr/bin/env python3
"""
Stage V4: memory-grounded re-identification. V2 passive tracking (SAM 2 + STTM + TCS, exactly as stage2_tcs.py) plus one
addition: at a search frame, pick the target among the candidates visible now and restart SAM 2 on it.

  python stage_reid.py --config config_v2_passive.yaml --v3-config config_v3_gemini_events_abstain.yaml \
        --variant off --pred-dir predictions_v4_off --results-dir results_v4_off [--video india]

Variants (V4_PLAN.md):  off     never restarts, only logs the searches (safety test: masks must equal V2 passive)
                        oracle  ground truth picks (a CEILING, never a result)
                        memory  best memory score if >= 0.6
                        vlm     the VLM picks among the 3 best by memory score (confidence >= 0.6)

What changes compared with stage2_tcs.py is marked "V4" in the code. Everything else is the same sequence of calls, on
purpose: a memory write moves SAM 2's conditioning frames, so any difference there would change masks and the safety test
(off == V2 passive) would fail.

Which expressions are live (can search): those with a candidate cache file candidates/<video>/<exp_id>.npz
(stage_candidates.py). Videos without one are not re-run; the other expressions of a live video are re-run too (so SAM 2
sees the same objects as in V2) but never search. The final predictions folder = a copy of the V2 passive masks, with the
live expressions' masks replaced by the V4 ones.

Search = trigger (V3 events with the frozen V3 settings, plus one poll every 10 frames while the mask is empty) ->
candidates (the tracker's current mask + cached Grounding-DINO/SAM 2 masks) -> chooser -> if the pick is another object
than "current": add its mask as a new prompt at this frame and propagate on from it. Frames before it never change.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reid as R
from events import EventParams, EventTrigger

VARIANTS = ("off", "oracle", "memory", "vlm")
VLM_CONF_MIN = 0.6                    # fixed in V4_PLAN.md
POLL_EVERY = 10                       # frames between searches while the tracker's mask is empty
GUARD_PER_ABSENT_EPISODE = 5          # VLM variant only
GUARD_PER_EXPRESSION = 10             # VLM variant only

LOG_FIELDS = ["video", "exp_id", "frame", "trigger", "variant", "n_candidates", "shortlist", "choice", "choice_value",
              "action", "label", "gt_visible", "current_iou", "chosen_iou", "vlm_letter", "vlm_conf", "vlm_latency_s",
              "vlm_cached", "vlm_error", "reason"]
EXP_FIELDS = ["video", "exp_id", "expression", "frames", "searches", "restarts", *R.LABELS, "skipped_uncached",
              "skipped_guard_episode", "skipped_guard_expression", "vlm_calls", "vlm_cached", "vlm_wait_s"]
TIMING_FIELDS = ["video", "n_frames", "n_objects", "n_live", "init_s", "propagate_s", "inference_s", "search_s",
                 "vlm_wait_s", "ms_per_frame", "peak_alloc_mb", "peak_reserved_mb"]


# ------------------------------------------------------------------ when to search (pure, unit-tested)
class SearchPlanner:
    """One per live expression. step() once per frame >= 1, in order. Returns the trigger name or None.
    Triggers: a V3 event (EventTrigger with the frozen V3 settings, which also enforces the 10-frame pause), or
    'absent_poll' = the tracker's mask has been empty and the last search is at least POLL_EVERY frames ago.
    Guards (VLM variant only) cap the cost: at most GUARD_PER_ABSENT_EPISODE searches per run of empty-mask frames and
    GUARD_PER_EXPRESSION per expression; a search that a guard blocks is counted, not run."""

    def __init__(self, event_params: EventParams, guard: bool, poll_every: int = POLL_EVERY):
        self.trig = EventTrigger(event_params)
        self.guard, self.poll_every = guard, poll_every
        self.last_search = None
        self.in_absent, self.episode_searches, self.total_searches = False, 0, 0
        self.skipped_guard_episode = self.skipped_guard_expression = 0

    def step(self, f, c, visible, state, margin, mask_empty):
        ev = self.trig.step(f, c, visible, state, margin)
        if mask_empty:
            if not self.in_absent:
                self.in_absent, self.episode_searches = True, 0
        else:
            self.in_absent = False
        trigger = ev
        if trigger is None and mask_empty and (self.last_search is None or f - self.last_search >= self.poll_every):
            trigger = "absent_poll"
            self.trig.last_call = f                      # share the pause with the event trigger
        if trigger is None:
            return None
        if self.guard:
            if self.total_searches >= GUARD_PER_EXPRESSION:
                self.skipped_guard_expression += 1
                return None
            if mask_empty and self.episode_searches >= GUARD_PER_ABSENT_EPISODE:
                self.skipped_guard_episode += 1
                return None
        self.last_search = f
        self.total_searches += 1
        if mask_empty:
            self.episode_searches += 1
        return trigger


# ------------------------------------------------------------------ who decides (pure, unit-tested)
def decide_search(variant, cands, ranked, gt_mask, p: R.ReIDParams, vlm_pick=None):
    """-> {"choice": candidate id or "none", "value": memory score / oracle IoU / VLM confidence or None, "vlm": dict or None}
    vlm_pick(shortlist) -> the dict returned by vlm.ask_reid (needs "index"); only called for the vlm variant."""
    if variant == "off" or not cands:
        return {"choice": R.NONE, "value": None, "vlm": None}
    if variant == "oracle":
        ch, v = R.choose_oracle(cands, gt_mask, p)
        return {"choice": ch, "value": v, "vlm": None}
    if variant == "memory":
        ch, v = R.choose_memory(ranked, p)
        return {"choice": ch, "value": v, "vlm": None}
    if variant == "vlm":
        sl = R.shortlist(ranked, p)
        if not sl:
            return {"choice": R.NONE, "value": None, "vlm": None}
        r = vlm_pick(sl)
        ch = sl[r["index"]]["id"] if r.get("index") is not None else R.NONE
        return {"choice": ch, "value": r.get("confidence"), "vlm": r}
    raise ValueError(f"variant must be one of {VARIANTS}")


# ------------------------------------------------------------------ the restart driver (pure w.r.t. SAM 2, unit-tested)
def drive(predictor, state, on_frame):
    """Run SAM 2 forward over the video once, restarting from a frame whenever on_frame asks to.
    on_frame(frame_idx, obj_ids, mask_logits) is called exactly once per frame, in order, and returns {sam obj_id: bool mask}
    for objects to restart at THIS frame (empty/None = nothing). For those the mask is added as a new prompt and
    propagation continues from this frame; the repeated yield of this frame is skipped, so no frame is processed twice and
    no earlier frame ever changes. Returns the number of restarts."""
    start, n_restarts = None, 0
    while True:
        kw = {} if start is None else {"start_frame_idx": start}
        restarted = None
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state, **kw):
            if start is not None and frame_idx <= start:
                continue
            restarts = on_frame(frame_idx, obj_ids, mask_logits)
            if restarts:
                for obj_id, m in restarts.items():
                    predictor.add_new_mask(inference_state=state, frame_idx=frame_idx, obj_id=obj_id, mask=m)
                    n_restarts += 1
                restarted = frame_idx
                break
        if restarted is None:
            return n_restarts
        start = restarted


# ------------------------------------------------------------------ small helpers
def _iou(a, b):
    return R.mask_iou(a, b)


def _fmt(x):
    return "" if x is None else (round(float(x), 4) if isinstance(x, (float, np.floating)) else x)


def box_to_prompt(box):
    x1, y1, x2, y2, _ = box
    return [x1, y1, x2, y2]


def _frame_hw(p: Path):
    with Image.open(p) as im:
        w, h = im.size
    return (h, w)


def mask_confidence(logits_np, mask):
    if not mask.any():
        return 0.0
    return float((1.0 / (1.0 + np.exp(-logits_np[mask]))).mean())


@dataclass
class Live:
    """Everything one live expression needs for searching."""
    exp_id: str
    store: object
    planner: SearchPlanner
    obj_id_gt: int
    anchor_mask: np.ndarray | None = None
    last_vis: tuple | None = None                 # (frame, mask) of the last frame with a visible target
    restart_frames: list = field(default_factory=list)
    counts: dict = field(default_factory=lambda: {"searches": 0, "restarts": 0, "skipped_uncached": 0, "vlm_calls": 0,
                                                  "vlm_cached": 0, "vlm_wait_s": 0.0, **{l: 0 for l in R.LABELS}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the V2 PASSIVE config (SAM 2, STTM, TCS settings, dataset)")
    ap.add_argument("--v3-config", required=True, help="the V3 config whose event settings (v3.events) and VLM (v3.vlm) V4 reuses")
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--src-preds", default="predictions_v2_passive")
    ap.add_argument("--cand-dir", default="candidates")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--video", action="append", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    v3 = yaml.safe_load(open(args.v3_config))["v3"]
    v1, v2 = cfg.get("v1", {}), cfg.get("v2", {})
    if bool(v2.get("gate_writes", False)):
        raise SystemExit("V4 builds on V2 PASSIVE: use the passive config (v2.gate_writes: false), e.g. config_v2_passive.yaml")
    from common.dataset import DavisLayout
    from common.io_utils import load_json, read_palette_mask, write_binary_mask
    from sttm import STTMParams, TargetMemory
    from sttm_sam2 import object_score, sync_bank
    from tcs import TCSParams, TrackletCoherence
    import vlm as V
    from stage_candidates import CandidateStore

    params = STTMParams(**v1.get("params", {}))
    tcs_params = TCSParams(**v2.get("tcs", {}))
    evp = EventParams(tau_low=tcs_params.tau_low, tau_high=tcs_params.tau_high, **dict(v3.get("events", {})))
    p = R.ReIDParams()
    variant = args.variant
    layout = DavisLayout(root=Path(cfg["dataset"]["root"]), require_480p=cfg["evaluation"]["require_480p"])
    grounding_dir = Path(cfg["paths"]["grounding_out"])
    emb_dir = Path(v1.get("embeddings_out", "embeddings/"))
    cand_dir, src_pred = Path(args.cand_dir), Path(args.src_preds)
    pred_dir, results_dir = Path(args.pred_dir), Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in sorted(grounding_dir.glob("*.json")) if (cand_dir / f.stem).is_dir()]
    if args.video:
        files = [f for f in files if f.stem in set(args.video)]
    if not files:
        raise FileNotFoundError(f"no live videos: need {cand_dir}/<video>/<exp_id>.npz (run stage_candidates.py) and {grounding_dir}/<video>.json")
    client = V.make_client(v3["vlm"]) if variant == "vlm" else None

    import torch
    from sam2.build_sam import build_sam2_video_predictor
    print(f"V4 variant={variant}; {len(files)} live video(s); events={evp}")
    predictor = build_sam2_video_predictor(cfg["propagation"]["model_cfg"], cfg["propagation"]["checkpoint"],
                                           device=cfg["propagation"]["device"])
    if not hasattr(predictor, "add_new_mask"):
        raise SystemExit("this SAM 2 install has no add_new_mask(); V4 needs it to restart on a candidate mask")
    from common.embed import ClipCropEmbedder, CropEmbedder
    crop_emb = CropEmbedder(device=cfg["propagation"]["device"])
    clip_crop = ClipCropEmbedder(device=cfg["propagation"]["device"])

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    log_rows, exp_rows, timing_rows = [], [], []

    for gfile in files:
        video = gfile.stem
        vg = load_json(gfile)
        frame_paths = layout.frame_paths(video)
        n_frames = len(frame_paths)
        H, W = _frame_hw(frame_paths[0])
        annos = layout.anno_paths(video)
        print(f"V4 {video}: {len(vg)} expression(s), {n_frames} frames")
        cache = np.load(emb_dir / f"{video}.npz")
        shutil.copytree(src_pred / video, pred_dir / video, dirs_exist_ok=True)     # V2 passive masks as the base

        torch.cuda.reset_peak_memory_stats()
        _sync()
        t_start = time.perf_counter()
        state = predictor.init_state(video_path=str(layout.jpeg_dir / video),
                                    offload_video_to_cpu=cfg["propagation"]["offload_video_to_cpu"],
                                    offload_state_to_cpu=cfg["propagation"]["offload_state_to_cpu"])
        _sync()
        init_s = time.perf_counter() - t_start

        exp_ids = list(vg.keys())
        sam_obj_id_of = {}
        for i, exp_id in enumerate(exp_ids, start=1):
            entry = vg[exp_id]
            if not entry["boxes"]:
                sam_obj_id_of[exp_id] = None
                continue
            sam_obj_id_of[exp_id] = i
            predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=i,
                                            box=np.array(box_to_prompt(entry["boxes"][0]), dtype=np.float32))
        mems, tcss, live = {}, {}, {}
        missing_counts, last_bank_sync = {}, {}
        for exp_id in exp_ids:
            oid = sam_obj_id_of[exp_id]
            if oid is None:
                continue
            k = oid - 1
            boxes_emb = cache[f"{exp_id}__boxes"]
            mems[k] = TargetMemory(params, (H, W), cache[f"{exp_id}__text"], boxes_emb[1:] if len(boxes_emb) > 1 else None)
            tcss[k] = TrackletCoherence(tcs_params)
            missing_counts[k] = 0
            cf = cand_dir / video / f"{exp_id}.npz"
            if cf.exists():
                live[k] = Live(exp_id, CandidateStore(cf), SearchPlanner(evp, guard=(variant == "vlm")), int(vg[exp_id]["obj_id"]))

        video_masks: dict = {}
        stats = {"search_s": 0.0, "vlm_wait_s": 0.0}

        def search(k, lv, f, img, mask, trigger):
            """One search for object k at frame f. Returns the candidate mask to restart on, or None."""
            t0 = time.perf_counter()
            lv.counts["searches"] += 1
            gt = read_palette_mask(annos[f], lv.obj_id_gt)
            cands = R.build_candidates(mask, lv.store.get(f), p)
            embeds = {}
            for c in cands:
                embeds[c["id"]] = crop_emb.embed_masked(img, c["mask"])
            state_m = R.memory_state(mems[k], (H, W))
            ranked = R.rank(cands, embeds, state_m, p)

            def vlm_pick(sl):
                lf, lm = lv.last_vis if lv.last_vis else (0, lv.anchor_mask)
                start_img = V.masked_crop(Image.open(frame_paths[0]).convert("RGB"), lv.anchor_mask)
                mem_img = V.masked_crop(Image.open(frame_paths[lf]).convert("RGB"), lm)
                crops = [V.masked_crop(img, c["mask"]) for c in sl]
                r = V.ask_reid(client, vg[lv.exp_id]["expression"], start_img, mem_img, crops, VLM_CONF_MIN)
                lv.counts["vlm_calls"] += 1
                lv.counts["vlm_cached"] += int(r["cached"])
                lv.counts["vlm_wait_s"] += r["latency_s"]
                stats["vlm_wait_s"] += r["latency_s"]
                return r

            d = decide_search(variant, cands, ranked, gt, p, vlm_pick)
            action, chosen = R.decide(d["choice"], cands)
            label, chosen_iou = R.label_attempt(cands, d["choice"], gt, p)
            cur = next((c for c in cands if c["id"] == R.CURRENT), None)
            r = d["vlm"] or {}
            lv.counts[label] += 1
            lv.counts["restarts"] += int(action == "restart")
            log_rows.append({
                "video": video, "exp_id": lv.exp_id, "frame": f, "trigger": trigger, "variant": variant,
                "n_candidates": len(cands), "shortlist": " ".join(c["id"] for c in R.shortlist(ranked, p)),
                "choice": d["choice"], "choice_value": _fmt(d["value"]), "action": action, "label": label,
                "gt_visible": int(bool(gt.any())), "current_iou": _fmt(_iou(cur["mask"], gt) if cur is not None else None),
                "chosen_iou": _fmt(chosen_iou if chosen_iou is not None else (_iou(chosen["mask"], gt) if chosen else None)),
                "vlm_letter": r.get("letter", ""), "vlm_conf": _fmt(r.get("confidence")),
                "vlm_latency_s": _fmt(r.get("latency_s")), "vlm_cached": r.get("cached", ""),
                "vlm_error": r.get("error", ""), "reason": r.get("reason", "")})
            stats["search_s"] += time.perf_counter() - t0 - r.get("latency_s", 0.0)
            return chosen["mask"] if action == "restart" else None

        def on_frame(frame_idx, obj_ids, mask_logits):
            img = None
            restarts = {}
            for k, (obj_id, logits) in enumerate(zip(obj_ids, mask_logits)):
                lg = logits.squeeze()
                mask = (lg > 0.0).cpu().numpy()
                mem = mems.get(k)
                lv = live.get(k)
                if mem is None:
                    video_masks.setdefault(obj_id, {})[frame_idx] = mask
                    continue
                _sync()
                if img is None:
                    img = Image.open(frame_paths[frame_idx]).convert("RGB")
                emb = crop_emb.embed_masked(img, mask) if mask.any() else None
                if frame_idx == 0:
                    video_masks.setdefault(obj_id, {})[frame_idx] = mask
                    if emb is None:
                        del mems[k]
                        tcss.pop(k, None)
                        live.pop(k, None)
                    else:
                        mem.set_anchor(0, mask, emb)
                        cemb0 = clip_crop.embed_masked(img, mask)
                        tcss[k].set_anchor(mask, float(cemb0 @ mem.semantic))
                        if lv is not None:
                            lv.anchor_mask = mask
                    continue
                od = state["output_dict_per_obj"][k]
                out = od["non_cond_frame_outputs"].get(frame_idx) or od["cond_frame_outputs"].get(frame_idx)
                osc = object_score(out)
                conf = mask_confidence(lg.float().cpu().numpy(), mask)
                cemb = clip_crop.embed_masked(img, mask) if mask.any() else None
                sem = float(cemb @ mem.semantic) if cemb is not None else None
                res = tcss[k].score(frame_idx, mem, mask, osc, conf, emb, sem)
                # ---- V4: search. Uses only information available at this frame (causal). The event trigger sees the same
                # rounded values the V2 log holds, so it fires exactly where the V3 replay fired.
                if lv is not None:
                    margin = None if res["margin"] is None else round(float(res["margin"]), 4)
                    trig = lv.planner.step(frame_idx, round(float(res["c"]), 4), res["visible"], res["state"], margin,
                                           not mask.any())
                    if trig is not None:
                        if not lv.store.has(frame_idx):
                            lv.counts["skipped_uncached"] += 1
                        else:
                            new_mask = search(k, lv, frame_idx, img, mask, trig)
                            if new_mask is not None:
                                mask = new_mask                                  # the corrected mask is this frame's output
                                emb = crop_emb.embed_masked(img, mask)
                                osc, conf = 10.0, 1.0                            # a mask prompt is a certain mask (as set_anchor)
                                restarts[obj_id] = torch.from_numpy(mask)
                                lv.restart_frames.append(frame_idx)
                    if res["visible"] and mask.any():
                        lv.last_vis = (frame_idx, mask)
                video_masks.setdefault(obj_id, {})[frame_idx] = mask
                dec = mem.update(frame_idx, mask, emb, osc, conf, allow_write=True)      # passive: gate_writes is false
                # A frame that is being restarted is NOT synced now: SAM 2 still holds the old output for it, and promoting
                # that would make the old mask a conditioning frame. The next write promotes the new prompt instead.
                if dec.get("written") and obj_id not in restarts:
                    pinned = [rf for rf in lv.restart_frames if rf < frame_idx] if lv is not None else []
                    r = sync_bank(state, k, list(mem.read_frames()) + pinned)       # earlier restart prompts stay conditioning
                    missing_counts[k] += len(r["missing"])
            return restarts

        _sync()
        t_prop = time.perf_counter()
        n_restarts = drive(predictor, state, on_frame)
        _sync()
        propagate_s = time.perf_counter() - t_prop
        inference_s = time.perf_counter() - t_start
        peak_a = torch.cuda.max_memory_allocated() / 2**20
        peak_r = torch.cuda.max_memory_reserved() / 2**20
        timing_rows.append({"video": video, "n_frames": n_frames, "n_objects": sum(v is not None for v in sam_obj_id_of.values()),
                            "n_live": len(live), "init_s": round(init_s, 3), "propagate_s": round(propagate_s, 3),
                            "inference_s": round(inference_s, 3), "search_s": round(stats["search_s"], 3),
                            "vlm_wait_s": round(stats["vlm_wait_s"], 3), "ms_per_frame": round(propagate_s / n_frames * 1000.0, 2),
                            "peak_alloc_mb": round(peak_a, 1), "peak_reserved_mb": round(peak_r, 1)})
        print(f"  {inference_s:.1f}s total ({stats['search_s']:.1f}s searching, {stats['vlm_wait_s']:.1f}s waiting for the VLM), "
              f"{n_restarts} restart(s), peak VRAM {peak_a:.0f} MB")

        # ---- write the live expressions' masks (the others keep the copied V2 passive masks)
        for k, lv in live.items():
            oid = k + 1
            for fi, fp in enumerate(frame_paths):
                m = video_masks.get(oid, {}).get(fi, np.zeros((H, W), bool))
                write_binary_mask(pred_dir / video / lv.exp_id / (fp.stem + ".png"), m)
            c = lv.counts
            exp_rows.append({"video": video, "exp_id": lv.exp_id, "expression": vg[lv.exp_id]["expression"], "frames": n_frames,
                             "searches": c["searches"], "restarts": c["restarts"], **{l: c[l] for l in R.LABELS},
                             "skipped_uncached": c["skipped_uncached"],
                             "skipped_guard_episode": lv.planner.skipped_guard_episode,
                             "skipped_guard_expression": lv.planner.skipped_guard_expression,
                             "vlm_calls": c["vlm_calls"], "vlm_cached": c["vlm_cached"], "vlm_wait_s": round(c["vlm_wait_s"], 3)})
        predictor.reset_state(state)

    del predictor
    torch.cuda.empty_cache()

    def dump(name, fields, rows):
        with open(results_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    dump("reid_log.csv", LOG_FIELDS, log_rows)
    dump("reid_per_expression.csv", EXP_FIELDS, exp_rows)
    dump("timing_per_video.csv", TIMING_FIELDS, timing_rows)
    n_s = sum(r["searches"] for r in exp_rows)
    lab = {l: sum(r[l] for r in exp_rows) for l in R.LABELS}
    lines = [f"variant: {variant}", f"live_videos: {len(timing_rows)}", f"live_expressions: {len(exp_rows)}",
             f"searches: {n_s}", f"restarts: {sum(r['restarts'] for r in exp_rows)}",
             *[f"label_{l}: {lab[l]}" for l in R.LABELS],
             f"skipped_uncached: {sum(r['skipped_uncached'] for r in exp_rows)}",
             f"skipped_guard_episode: {sum(r['skipped_guard_episode'] for r in exp_rows)}",
             f"skipped_guard_expression: {sum(r['skipped_guard_expression'] for r in exp_rows)}",
             f"vlm_calls: {sum(r['vlm_calls'] for r in exp_rows)} (cached answers: {sum(r['vlm_cached'] for r in exp_rows)})",
             f"live_frames: {sum(r['n_frames'] for r in timing_rows)}",
             f"total_inference_s (live reruns only): {sum(r['inference_s'] for r in timing_rows):.1f}",
             f"total_search_s: {sum(r['search_s'] for r in timing_rows):.1f}",
             f"total_vlm_wait_s: {sum(r['vlm_wait_s'] for r in timing_rows):.1f}",
             f"max_peak_alloc_mb: {max((r['peak_alloc_mb'] for r in timing_rows), default=0):.0f}"]
    (results_dir / "reid_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Stage V4 ({variant}) done. Predictions under {pred_dir}/ -> next: python stage3_evaluate.py --config <config> with this predictions folder")


if __name__ == "__main__":
    main()
