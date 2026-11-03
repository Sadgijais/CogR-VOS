#!/usr/bin/env python3
"""
Amendment 9 item 5 (Task 4): can the Tracklet Coherence Score (TCS) predict failures? CPU only, reads only, no re-tuning (the band thresholds are the
FROZEN values stored in the TCS log: tau_low 0.81, tau_high 0.92). Uses the existing results_v2_passive/tcs_log, predictions_v2_passive and the
ground truth of the subset.

Definitions (fixed before any number of this tool was produced):
  frame t of a sentence = a frame of its TCS log (f = 1, 2, ...; f = 0 is the initial frame and is not scored; prediction file <f:05d>.png, ground truth
  file frames[f] of meta_expressions.json). Target mask G_t = palette index obj_id of the ground truth; P_t = predicted mask.
  gt visible = G_t non-empty. FAILED = gt visible and IoU(P_t, G_t) < 0.50 (an empty prediction on a visible target is failed, IoU 0).
  TCS low-is-bad: a low coherence c_t = a predicted problem. Flags: LOW = band LOW; LOW+MEDIUM = band not HIGH.
  DETECTION (k = 0): population = gt visible and prediction non-empty; positive = FAILED at t.
  FORECAST (k = 5, 10): population = gt visible, prediction non-empty and NOT failed at t, with at least k later scored frames; positive = at least one
  FAILED frame among t+1 .. t+k. (Frames with the target absent are not failures; they are the disappearance events of Amendment 9 item 4.)
  Empty-mask frames (prediction empty, gt visible) are excluded from the AUROC (their TCS is 0 by definition) and counted separately.
  Failure run = at least 3 consecutive scored frames with gt visible and FAILED. For each run: onset at the first scored frame (cannot be forecast; it is
  an initial failure) or, otherwise, whether a flag (LOW+MEDIUM) occurred in the 10 scored frames before the onset, and the lead.
  Sentences whose target is absent at frame 0 are excluded and counted. Populations: ALL (visible at frame 0) and GROUNDING_OK (frame-0 mask IoU >= 0.10
  with the target). The GROUNDING_OK split is added by the author before the numbers, because grounding failures are invisible to a score that compares a
  track with its own anchor (earlier note, Ref-DAVIS17); both are reported.
  AUROC: probability that a positive frame has a LOWER c than a negative frame (ties 0.5). Interval: 10,000 resamples of the VIDEOS (seed 0); draws with a
  single class are skipped and counted. Term-wise AUROCs (mask, motion, appearance, semantic) are point estimates.

  python -I tools/tcs_failure_prediction.py --root /home/sadgi/data/MeViS-subset --run ~/cogrvos/task6_mevis/baseline_v0 \
         --out ~/cogrvos/analysis/tcs_fp_mevis.csv --label "MeViS valid_u subset (12 videos, single-target sentences) - not the full benchmark"
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io_utils import read_binary_mask, read_palette_mask  # noqa: E402

TERMS = ["mask", "motion", "appearance", "semantic"]
BAD_IOU, GROUND_IOU, MIN_RUN, LEAD_WIN = 0.50, 0.10, 3, 10


def iou(a, b):
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def avg_ranks(x):
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    return ((cum - cnt + cum + 1) / 2.0)[inv]


def auroc_low_bad(c, bad):
    nb, ng = int(bad.sum()), int((~bad).sum())
    if nb == 0 or ng == 0:
        return float("nan")
    r = avg_ranks(c)
    u_bad = r[bad].sum() - nb * (nb + 1) / 2.0
    return float(1.0 - u_bad / (nb * ng))


def boot_auroc(per_video, reps, seed):
    vids = sorted(per_video)
    rng = np.random.default_rng(seed)
    out, skipped = [], 0
    for _ in range(reps):
        pick = rng.integers(0, len(vids), len(vids))
        c = np.concatenate([per_video[vids[i]][0] for i in pick])
        b = np.concatenate([per_video[vids[i]][1] for i in pick])
        v = auroc_low_bad(c, b)
        if np.isnan(v):
            skipped += 1
        else:
            out.append(v)
    if not out:
        return float("nan"), float("nan"), skipped
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root with Annotations/ and meta_expressions/")
    ap.add_argument("--run", required=True, help="run folder with grounding/, results_v2_passive/, predictions_v2_passive/")
    ap.add_argument("--results", default="results_v2_passive")
    ap.add_argument("--pred", default="predictions_v2_passive")
    ap.add_argument("--tier", default="480p")
    ap.add_argument("--out", required=True, help="per-frame CSV to write (must not exist)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root, run, outp = Path(a.root).expanduser(), Path(a.run).expanduser(), Path(a.out).expanduser()
    if outp.exists():
        sys.exit(f"refusing to overwrite: {outp}")
    outp.parent.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(root / "meta_expressions" / "meta_expressions.json"))["videos"]

    sent = []      # one dict per sentence: video, exp, ground_ok, frames[list of dict]
    n_absent_f0 = n_short = n_missing = 0
    for lf in sorted((run / a.results / "tcs_log").glob("*.json")):
        video = lf.stem
        tcs = json.load(open(lf))
        gro = json.load(open(run / "grounding" / f"{video}.json"))
        obj_of = {e: int(v["obj_id"]) for e, v in gro.items()}
        names = meta[video]["frames"]
        gt_cache = {}

        def gt_arr(i):
            if i not in gt_cache:
                from PIL import Image
                p = root / "Annotations" / a.tier / video / f"{names[i]}.png"
                im = Image.open(p)
                im = im if im.mode == "P" else im.convert("P")
                gt_cache[i] = np.array(im)
            return gt_cache[i]

        for exp_id, ent in tcs.items():
            oid = obj_of[exp_id]
            others = sorted({o for e, o in obj_of.items() if o != oid})
            g0 = gt_arr(0) == oid
            if not g0.any():
                n_absent_f0 += 1
                continue
            p0f = run / a.pred / video / exp_id / "00000.png"
            ground_ok = p0f.exists() and iou(read_binary_mask(p0f), g0) >= GROUND_IOU
            frs = []
            for r in ent["frames"]:
                f = int(r["f"])
                pf = run / a.pred / video / exp_id / f"{f:05d}.png"
                if not pf.exists() or f >= len(names):
                    n_missing += 1
                    continue
                P, arr = read_binary_mask(pf), gt_arr(f)
                G = arr == oid
                vis = bool(G.any())
                io = iou(P, G) if vis else float("nan")
                frs.append({"f": f, "c": float(r["c"]), "band": r["band"], "terms": r.get("terms", {}), "gt_visible": vis,
                            "pred_empty": not P.any(), "iou": io, "iou_other": max([iou(P, arr == o) for o in others], default=0.0),
                            "failed": bool(vis and io < BAD_IOU)})
            if sum(1 for x in frs if x["gt_visible"]) < MIN_RUN:
                n_short += 1
            sent.append({"video": video, "exp": exp_id, "ground_ok": bool(ground_ok), "frames": frs})

    with open(outp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "exp_id", "grounding_ok", "f", "c", "band", "mask", "motion", "appearance", "semantic", "gt_visible", "pred_empty", "iou", "iou_other", "failed"])
        for s in sent:
            for x in s["frames"]:
                w.writerow([s["video"], s["exp"], int(s["ground_ok"]), x["f"], x["c"], x["band"]] + [x["terms"].get(t, "") for t in TERMS] +
                           [int(x["gt_visible"]), int(x["pred_empty"]), "" if np.isnan(x["iou"]) else f"{x['iou']:.4f}", f"{x['iou_other']:.4f}", int(x["failed"])])

    print(a.label)
    print(f"TCS = {a.results} (frozen thresholds from the log), predictions = {a.pred}. Sentences used {len(sent)}; excluded: target absent at frame 0 {n_absent_f0}; "
          f"frames without a prediction file {n_missing}; sentences with fewer than {MIN_RUN} visible scored frames {n_short} (kept in the frame tables).")
    for popname, keep in (("ALL (target visible at frame 0)", lambda s: True), ("GROUNDING_OK (frame-0 mask IoU >= 0.10 with the target)", lambda s: s["ground_ok"])):
        S = [s for s in sent if keep(s)]
        print(f"\n=== {popname}: {len(S)} sentences, {len({s['video'] for s in S})} videos ===")
        vis_f = [x for s in S for x in s["frames"] if x["gt_visible"]]
        emp = [x for x in vis_f if x["pred_empty"]]
        print(f"frames with the target visible: {len(vis_f)}; failed: {sum(x['failed'] for x in vis_f)}; of the failed, empty prediction: "
              f"{sum(x['failed'] for x in emp)} (flagged LOW+MEDIUM: {sum(x['band'] != 'HIGH' for x in emp if x['failed'])}; excluded from the AUROC below)")
        print(f"\n{'task':22s} {'frames':>7} {'positive':>8} {'AUROC':>7} {'95% interval':>16} {'skipped':>8}  | LOW: recall / false alarm | LOW+MED: recall / false alarm")
        for k in [0] + list(a.horizons):
            rows = []
            for s in S:
                fr = s["frames"]
                for i, x in enumerate(fr):
                    if not x["gt_visible"] or x["pred_empty"]:
                        continue
                    if k == 0:
                        rows.append((s["video"], x, x["failed"]))
                    elif not x["failed"] and i + k < len(fr):
                        rows.append((s["video"], x, any(y["failed"] for y in fr[i + 1:i + k + 1])))
            name = "detect (k=0)" if k == 0 else f"forecast (k={k})"
            if not rows:
                print(f"{name:22s} no frames")
                continue
            c = np.array([r[1]["c"] for r in rows]); b = np.array([r[2] for r in rows]); v = [r[0] for r in rows]
            pv = {}
            for vid in sorted(set(v)):
                m = np.array([x == vid for x in v])
                pv[vid] = (c[m], b[m])
            au = auroc_low_bad(c, b)
            lo, hi, sk = boot_auroc(pv, a.reps, a.seed)
            low = np.array([r[1]["band"] == "LOW" for r in rows]); lm = np.array([r[1]["band"] != "HIGH" for r in rows])
            f = lambda m: (f"{m[b].mean():.2f} / {m[~b].mean():.2f}" if b.any() and (~b).any() else "n/a")
            print(f"{name:22s} {len(rows):>7} {int(b.sum()):>8} {au:>7.3f} {f'[{lo:.3f}, {hi:.3f}]':>16} {sk:>8}  | {f(low):>25} | {f(lm):>25}")
            if k == 0:
                tr = "   terms alone (point AUROC): " + ", ".join(
                    f"{t} {auroc_low_bad(np.array([r[1]['terms'].get(t, np.nan) for r in rows], float), b):.3f}" for t in TERMS)
                print(tr)
        runs = []
        for s in S:
            fr, i = s["frames"], 0
            while i < len(fr):
                if fr[i]["gt_visible"] and fr[i]["failed"]:
                    j = i
                    while j + 1 < len(fr) and fr[j + 1]["gt_visible"] and fr[j + 1]["failed"]:
                        j += 1
                    if j - i + 1 >= MIN_RUN:
                        runs.append((s, i, j))
                    i = j + 1
                else:
                    i += 1
        first = [r for r in runs if r[1] == 0]
        later = [r for r in runs if r[1] > 0]
        flagged, leads = 0, []
        for s, i, j in later:
            win = [(s["frames"][i]["f"] - s["frames"][q]["f"]) for q in range(max(0, i - LEAD_WIN), i) if s["frames"][q]["band"] != "HIGH"]
            if win:
                flagged += 1
                leads.append(max(win))
        print(f"\nfailure runs (>= {MIN_RUN} consecutive failed frames): {len(runs)} in {len({r[0]['video'] + r[0]['exp'] for r in runs})} sentences; "
              f"onset at the first scored frame (initial failure, cannot be forecast): {len(first)}; later onset: {len(later)}")
        if later:
            print(f"  later onsets with a LOW/MEDIUM flag in the {LEAD_WIN} scored frames before: {flagged} of {len(later)}; lead (earliest flag to onset): "
                  f"median {np.median(leads) if leads else float('nan'):.1f}, max {max(leads) if leads else float('nan')} frames")
            hi_pre = sum(1 for s, i, j in later if all(s["frames"][q]["band"] == "HIGH" for q in range(max(0, i - LEAD_WIN), i)))
            print(f"  later onsets preceded only by HIGH frames: {hi_pre}")
    print("\nSingle run, subset. Sentences of one video are not independent; intervals cover video sampling only. TCS thresholds not re-tuned.")


if __name__ == "__main__":
    sys.exit(main())
