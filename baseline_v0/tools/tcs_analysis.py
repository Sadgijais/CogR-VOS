#!/usr/bin/env python3
"""
Does the Tracklet Coherence Score actually mean anything?   (V2 evidence for H2)

Joins every frame's TCS record (results/tcs_log/<video>.json, written by stage2_tcs.py) to the
ground truth and the predictions, then reports:

  1. Per band (HIGH / MEDIUM / LOW): share of frames, mean IoU, share of lost / degraded frames.
  2. AUROC of c_t and of each of its four terms for spotting bad frames
        lost      IoU on the target < 0.10 (target present in the ground truth)
        degraded  IoU on the target < 0.50
     0.5 = no information, 1.0 = perfect.
  3. Failure cases: every run of >= 3 lost frames, whether it was a drift (IoU > 0.5 on ANOTHER
     annotated object, the stage-4 definition), and whether/when TCS flagged it.
  4. --suggest: band thresholds by a rule fixed in advance (V2_TUNING_PLAN.md). It REFUSES to run
     on any video outside the ablation manifest, so thresholds can never be tuned on the val split.

Expressions whose FRAME-0 mask already misses the target (IoU < 0.10: grounding failures) are left
out of the AUROC and the thresholds: TCS compares a track with its own anchor, so a track that is
consistently on the wrong object cannot look incoherent. They are listed separately.

No GPU. Usage (from baseline_v0/):
    python tools/tcs_analysis.py --config config_v2.yaml
    python tools/tcs_analysis.py --config config_v2_tune.yaml --manifest subsets/ref_davis17_ablation.json --suggest \
        --write-config config_v2_final.yaml
    python tools/tcs_analysis.py --config config_v2.yaml --tau-low 0.4 --tau-high 0.6     # re-band offline, no rerun
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.dataset import DavisLayout
from common.io_utils import load_json, read_binary_mask

LOST_IOU = 0.10
DEGRADED_IOU = 0.50
DRIFT_OTHER_IOU = 0.50
MIN_RUN = 3                  # frames in a row to count as a failure case
LEAD_WINDOW_BEFORE = 10      # look this many frames before onset for the first non-HIGH frame
LEAD_WINDOW_AFTER = 3        # ... and this many after (late flags count, with negative lead)
MIN_POSITIVES = 10           # fewer bad frames than this: AUROC / thresholds are not trustworthy
TERMS = ("mask", "motion", "appearance", "semantic")


# ------------------------------------------------------------------ pure helpers (unit-tested)
def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks, ties get the average rank (numpy only)."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), float)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc_low_is_bad(scores, bad) -> float:
    """P(score of a random bad frame < score of a random good frame); ties count half.
    0.5 = no information. nan if either class is empty."""
    s = np.asarray(scores, float)
    y = np.asarray(bad, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = _avg_ranks(-s)                    # high rank = LOW score = looks bad
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def youden_threshold(scores, bad, grid=None):
    """Cut t (flag 'bad' when c < t) maximising TPR - FPR. Returns (t, tpr, fpr)."""
    s = np.asarray(scores, float)
    y = np.asarray(bad, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    best = None
    for t in grid:
        flag = s < t
        tpr, fpr = float(flag[y].sum() / n1), float(flag[~y].sum() / n0)
        if best is None or (tpr - fpr) > (best[1] - best[2]) + 1e-12:
            best = (float(t), tpr, fpr)
    return best


def band_array(c, tau_low, tau_high):
    c = np.asarray(c, float)
    return np.where(c >= tau_high, "HIGH", np.where(c < tau_low, "LOW", "MEDIUM"))


def lost_runs(frames, lost, min_run=MIN_RUN):
    """Contiguous runs of consecutive frame indices where lost is True, length >= min_run."""
    runs, cur = [], []
    for f, l in zip(frames, lost):
        if l and (not cur or f == cur[-1] + 1):
            cur.append(f)
        else:
            if len(cur) >= min_run:
                runs.append(cur)
            cur = [f] if l else []
    if len(cur) >= min_run:
        runs.append(cur)
    return runs


def first_flag(frames, bands, onset, before=LEAD_WINDOW_BEFORE, after=LEAD_WINDOW_AFTER):
    """Earliest frame in [onset-before, onset+after] whose band is not HIGH, else None."""
    for f, b in sorted(zip(frames, bands)):
        if onset - before <= f <= onset + after and b != "HIGH":
            return f
    return None


# ------------------------------------------------------------------ data join
def _palette_array(p: Path) -> np.ndarray:
    im = Image.open(p)
    if im.mode != "P":
        im = im.convert("P")
    return np.array(im)


def build_rows(cfg, results_dir: Path, pred_dir: Path, tau_low=None, tau_high=None):
    layout = DavisLayout(root=Path(cfg["dataset"]["root"]), require_480p=False)
    gdir = Path(cfg["paths"]["grounding_out"])
    g_iou = cfg.get("failure_analysis", {}).get("grounding_failure_iou", 0.10)
    rows, exps = [], []
    for lf in sorted((results_dir / "tcs_log").glob("*.json")):
        video = lf.stem
        vlog, vg = load_json(lf), load_json(gdir / f"{video}.json")
        annos = layout.anno_paths(video)
        n = len(annos)
        gt_arrays = [_palette_array(p) for p in annos]
        for exp_id, e in vlog.items():
            obj_id = int(vg[exp_id]["obj_id"])
            tl = tau_low if tau_low is not None else e["tau_low"]
            th = tau_high if tau_high is not None else e["tau_high"]
            pdir = pred_dir / video / exp_id
            pred0 = read_binary_mask(pdir / f"{annos[0].stem}.png")
            iou0 = iou(pred0, gt_arrays[0] == obj_id)
            grounding_ok = iou0 >= g_iou
            exps.append({"video": video, "exp_id": exp_id, "expression": e["expression"],
                         "obj_id": obj_id, "grounding_ok": grounding_ok, "iou_frame0": iou0})
            for r in e["frames"]:
                f = r["f"]
                if f >= n - 1:                                  # DAVIS convention: drop the last frame too
                    continue
                pred = read_binary_mask(pdir / f"{annos[f].stem}.png")
                arr = gt_arrays[f]
                gt = arr == obj_id
                other = [i for i in np.unique(arr).tolist() if i not in (0, obj_id)]
                iou_other = max([iou(pred, arr == i) for i in other], default=0.0)
                gt_present = bool(gt.any())
                iou_t = iou(pred, gt)
                rows.append({
                    "video": video, "exp_id": exp_id, "frame": f, "c": r["c"],
                    "band": str(band_array([r["c"]], tl, th)[0]), "state": r["state"],
                    "visible": r["visible"], **{f"t_{t}": r["terms"].get(t) for t in TERMS},
                    "margin": r.get("margin"), "gt_present": gt_present, "iou": iou_t,
                    "iou_other": iou_other, "grounding_ok": grounding_ok,
                    "lost": gt_present and iou_t < LOST_IOU,
                    "degraded": gt_present and iou_t < DEGRADED_IOU,
                    "drift": gt_present and iou_t < LOST_IOU and iou_other > DRIFT_OTHER_IOU,
                    "false_absence": gt_present and not r["visible"],
                    "write_result": r.get("result")})
    return rows, exps


# ------------------------------------------------------------------ analyses
def band_table(rows):
    out = []
    n = len(rows)
    for b in ("HIGH", "MEDIUM", "LOW"):
        sub = [r for r in rows if r["band"] == b]
        out.append({"band": b, "frames": len(sub), "share_pct": 100 * len(sub) / max(n, 1),
                    "mean_iou": float(np.mean([r["iou"] for r in sub])) if sub else float("nan"),
                    "lost_pct": 100 * float(np.mean([r["lost"] for r in sub])) if sub else float("nan"),
                    "degraded_pct": 100 * float(np.mean([r["degraded"] for r in sub])) if sub else float("nan")})
    return out


def auroc_table(rows):
    out = []
    for name, key in [("c_t (combined)", "c")] + [(t, f"t_{t}") for t in TERMS]:
        sub = [r for r in rows if r[key] is not None]
        res = {"signal": name, "frames": len(sub)}
        for label in ("lost", "degraded"):
            y = [r[label] for r in sub]
            res[f"auroc_{label}"] = auroc_low_is_bad([r[key] for r in sub], y)
            res[f"n_{label}"] = int(sum(y))
        out.append(res)
    return out


def failure_cases(rows, exps):
    cases = []
    ok = {(e["video"], e["exp_id"]): e["grounding_ok"] for e in exps}
    by = {}
    for r in rows:
        by.setdefault((r["video"], r["exp_id"]), []).append(r)
    for (video, exp_id), rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: r["frame"])
        if not ok[(video, exp_id)]:
            cases.append({"video": video, "exp_id": exp_id, "kind": "grounding_failure (not detectable by TCS)",
                          "onset": 0, "length": len(rs), "drift": False, "c_at_onset": "", "band_at_onset": "",
                          "first_flag_frame": "", "lead_frames": "", "flagged": ""})
            continue
        frames = [r["frame"] for r in rs]
        runs = lost_runs([r["frame"] for r in rs if r["gt_present"]], [r["lost"] for r in rs if r["gt_present"]])
        for run in runs:
            onset = run[0]
            ro = next(r for r in rs if r["frame"] == onset)
            ff = first_flag(frames, [r["band"] for r in rs], onset)
            drift = any(next(r for r in rs if r["frame"] == f)["drift"] for f in run)
            cases.append({"video": video, "exp_id": exp_id,
                          "kind": "identity_drift" if drift else "track_lost",
                          "onset": onset, "length": len(run), "drift": drift,
                          "c_at_onset": round(ro["c"], 3), "band_at_onset": ro["band"],
                          "first_flag_frame": "" if ff is None else ff,
                          "lead_frames": "" if ff is None else onset - ff, "flagged": ff is not None})
    return cases


def suggest_thresholds(rows, placeholder_low, placeholder_high):
    """Pre-registered rule (V2_TUNING_PLAN.md): tau_low = Youden cut for LOST frames,
    tau_high = Youden cut for DEGRADED frames, tau_high >= tau_low. Needs >= MIN_POSITIVES bad frames."""
    out = {"rule": "Youden J; tau_low on lost (IoU<0.10), tau_high on degraded (IoU<0.50); grounding failures excluded"}
    c = [r["c"] for r in rows]
    for key, label, ph in (("tau_low", "lost", placeholder_low), ("tau_high", "degraded", placeholder_high)):
        y = [r[label] for r in rows]
        n_pos = int(sum(y))
        res = youden_threshold(c, y) if n_pos >= MIN_POSITIVES else None
        if res is None:
            out[key] = ph
            out[f"{key}_note"] = f"FALLBACK to placeholder: only {n_pos} {label} frames (< {MIN_POSITIVES})"
        else:
            t, tpr, fpr = res
            out[key] = t
            out[f"{key}_note"] = f"{label}: n_pos={n_pos}, TPR={tpr:.2f}, FPR={fpr:.2f}, Youden J={tpr - fpr:.2f}"
    if out["tau_high"] < out["tau_low"]:
        out["tau_high"] = out["tau_low"]
        out["tau_high_note"] += " (raised to tau_low)"
    return out


def write_config(template: Path, out: Path, tau_low: float, tau_high: float, gate: bool):
    s = template.read_text()
    s = re.sub(r"(tau_low:\s*)[0-9.]+", rf"\g<1>{tau_low:.2f}", s)
    s = re.sub(r"(tau_high:\s*)[0-9.]+", rf"\g<1>{tau_high:.2f}", s)
    s = re.sub(r"(gate_writes:\s*)(true|false)", rf"\g<1>{'true' if gate else 'false'}", s)
    s = re.sub(r"PLACEHOLDER - [^\n]*", "TUNED on the ablation tier by tools/tcs_analysis.py --suggest, then FROZEN", s)
    out.write_text(s)


def _fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v2.yaml")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--tau-low", type=float, default=None, help="re-band offline with this threshold")
    ap.add_argument("--tau-high", type=float, default=None)
    ap.add_argument("--manifest", default=None, help="subset manifest; required with --suggest")
    ap.add_argument("--suggest", action="store_true", help="suggest band thresholds (ablation tier only)")
    ap.add_argument("--write-config", default=None, help="with --suggest: write the frozen config here")
    ap.add_argument("--template", default="config_v2.yaml")
    ap.add_argument("--include-grounding-failures", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    results_dir = Path(a.results_dir or cfg["paths"]["results_out"])
    pred_dir = Path(a.pred_dir or cfg["paths"]["predictions_out"])

    rows, exps = build_rows(cfg, results_dir, pred_dir, a.tau_low, a.tau_high)
    if not rows:
        raise SystemExit(f"no TCS logs under {results_dir}/tcs_log - run stage2_tcs.py first")
    videos = sorted({r["video"] for r in rows})

    if a.suggest:
        if not a.manifest:
            raise SystemExit("--suggest needs --manifest subsets/ref_davis17_ablation.json")
        allowed = set(json.load(open(a.manifest))["videos"])
        outside = [v for v in videos if v not in allowed]
        if outside:
            raise SystemExit(f"REFUSING to suggest thresholds: {outside} are not in the ablation manifest. "
                             "Thresholds may be tuned on the ablation tier only.")

    ev = [r for r in rows if r["gt_present"] and (r["grounding_ok"] or a.include_grounding_failures)]
    n_gf = sum(not e["grounding_ok"] for e in exps)

    bt, at, cases = band_table(ev), auroc_table(ev), failure_cases(rows, exps)
    flagged = [c for c in cases if c["kind"] in ("identity_drift", "track_lost")]
    bad = [r for r in ev if r["lost"]]
    good = [r for r in ev if not r["degraded"]]
    uncertain = lambda r: r["band"] != "HIGH"

    lines = [f"TCS analysis  -  {len(videos)} videos, {len(exps)} expressions ({n_gf} grounding failures left out), "
             f"{len(ev)} frames evaluated", f"results: {results_dir}   predictions: {pred_dir}",
             f"thresholds used: tau_low={a.tau_low if a.tau_low is not None else 'from log'}, "
             f"tau_high={a.tau_high if a.tau_high is not None else 'from log'}", "",
             "1. BANDS (stable = HIGH, uncertain = MEDIUM + LOW)",
             f"   {'band':<8}{'frames':>8}{'share%':>9}{'meanIoU':>9}{'lost%':>8}{'degraded%':>11}"]
    lines += [f"   {b['band']:<8}{b['frames']:>8}{b['share_pct']:>9.1f}{_fmt(b['mean_iou'], 2):>9}"
              f"{_fmt(b['lost_pct'], 1):>8}{_fmt(b['degraded_pct'], 1):>11}" for b in bt]
    if bad:
        lines.append(f"   lost frames caught as uncertain (recall): {100 * np.mean([uncertain(r) for r in bad]):.1f}%  "
                     f"(n={len(bad)})")
    if good:
        lines.append(f"   good frames (IoU>=0.5) wrongly flagged uncertain (false alarms): "
                     f"{100 * np.mean([uncertain(r) for r in good]):.1f}%  (n={len(good)})")
    lines += ["", "2. AUROC  (0.5 = no information, 1.0 = perfect; low score should mean bad frame)",
              f"   {'signal':<16}{'frames':>8}{'lost':>8}{'(n)':>6}{'degraded':>10}{'(n)':>7}"]
    for t in at:
        lines.append(f"   {t['signal']:<16}{t['frames']:>8}{_fmt(t['auroc_lost'], 3):>8}{t['n_lost']:>6}"
                     f"{_fmt(t['auroc_degraded'], 3):>10}{t['n_degraded']:>7}")
    if any(t["n_lost"] < MIN_POSITIVES for t in at):
        lines.append(f"   WARNING: fewer than {MIN_POSITIVES} lost frames -> the 'lost' AUROC is not trustworthy "
                     "(Ref-DAVIS17 has almost no drift; this is decided on Long-RVOS / MeViS).")
    lines += ["", f"3. FAILURE CASES  ({len(flagged)} runs of >= {MIN_RUN} lost frames in correctly grounded tracks)"]
    for c in cases:
        if c["kind"].startswith("grounding"):
            lines.append(f"   {c['video']}/{c['exp_id']}: grounding failure, whole track off-target - TCS cannot see this")
        else:
            lead = "NOT flagged" if c["flagged"] is False else f"flagged at frame {c['first_flag_frame']} (lead {c['lead_frames']:+d})"
            lines.append(f"   {c['video']}/{c['exp_id']}: {c['kind']} from frame {c['onset']} for {c['length']} frames; "
                         f"c at onset {c['c_at_onset']} ({c['band_at_onset']}); {lead}")
    if flagged:
        det = [c for c in flagged if c["flagged"]]
        lines.append(f"   detected: {len(det)}/{len(flagged)}")

    sug = None
    if a.suggest:
        sug = suggest_thresholds(ev, 0.50, 0.70)
        lines += ["", "4. SUGGESTED THRESHOLDS (ablation tier only; freeze these)",
                  f"   tau_low  = {sug['tau_low']:.2f}   ({sug['tau_low_note']})",
                  f"   tau_high = {sug['tau_high']:.2f}   ({sug['tau_high_note']})"]
        (results_dir / "tcs_thresholds.json").write_text(json.dumps(sug, indent=2))
        if a.write_config:
            write_config(Path(a.template), Path(a.write_config), sug["tau_low"], sug["tau_high"], gate=True)
            lines.append(f"   wrote {a.write_config} (gate_writes: true). Read it, then commit it BEFORE the full run.")

    text = "\n".join(lines) + "\n"
    (results_dir / "TCS_ANALYSIS.txt").write_text(text)
    print(text)

    def dump(name, items):
        if items:
            with open(results_dir / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(items[0].keys())); w.writeheader(); w.writerows(items)
    dump("tcs_frames.csv", [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
    dump("tcs_failure_cases.csv", cases)
    dump("tcs_bands.csv", bt)
    dump("tcs_auroc.csv", at)


if __name__ == "__main__":
    main()
