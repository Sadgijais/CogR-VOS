#!/usr/bin/env python3
"""
Task 3 (Sir, 3 Nov 2026): frame-wise temporal failure analysis. Amendment 9 item 4. CPU only; reads saved files only (no tracking, no GPU, no Gemini).

Unit = one sentence (target object) under one arm. Per annotated frame t of the video (names from meta_expressions.json "frames"):
  G_t = ground-truth mask of the target (absent when the object id is not in the palette PNG or the PNG is missing)
  P_t = predicted mask (a missing prediction file counts as empty and is counted in the printout)
  IoU_t = IoU(P_t, G_t); other_t = max IoU of P_t with the GT mask of any OTHER annotated object of the video
          (annotated = has a sentence in grounding/<video>.json; unannotated look-alikes are not known)
Frame states: absent (G_t empty) | good (IoU_t >= 0.50) | failed (visible and IoU_t < 0.50) | lost (visible and IoU_t < 0.10)
Failure run: >= --min-run (default 3) CONSECUTIVE annotated frames that are all visible and failed; an absent frame ends a run.
Onset = first frame of the FIRST failure run. No failure run = "no_failure". Category of that run, first true rule wins:
  1 reid_failure               the onset lies in [r, r + window) for a return frame r (a frame where the target is visible again after an absent
                               run that has a visible frame on BOTH sides, as in common/failure_taxonomy.find_reappearance_events), and no frame
                               of that window has IoU >= 0.50 (the recovery rule of the existing taxonomy, window = 10 annotated frames)
  2 initial_grounding_failure  the onset is the first visible frame
  3 identity_drift             later onset and other_t >= 0.50 at the onset frame
  4 mask_propagation_failure   later onset and other_t < 0.50 (target lost, shrunk, split or drifted onto background)
similar_object flag = other_t >= 0.50 at the onset frame (also set for initial_grounding_failure). Occlusion / disappearance is reported as EVENTS
(absent runs, reappearance events, false-presence frames = P non-empty while G empty, missed-presence frames = P empty while G non-empty), not as a category.
Groups (kept apart in every table): visible_f0 (target visible at frame 0), absent_f0 (first visible later), never_visible.
Propagation measures per failing sentence: onset position and fraction (onset / (n_frames - 1)), run length, recovers (a later frame after the run has
IoU >= 0.50), share of visible frames from the onset on (onset included) that are failed.
Start box (for the largest-box question): start_hit = IoU of boxes[0] of grounding_<arm>/<video>.json with the tight GT box at frame 0 >= 0.50.

Run length rule, fixed BEFORE any failure number: --min-run 3. Use 2 for a dataset only if more than 10% of its sentences that are visible at least once
have only 1 or 2 visible frames. Decide with --length-check (GT only, no predictions), then pass --min-run explicitly.
The existing sentence-level labels (failure_analysis*.csv) are shown next to the new categories and are not merged with them.
Descriptive only: single run, subsets, sentences of one video are not independent, categories come from the fixed rules and not from looking at pictures.

  python -I tools/frame_failure_analysis.py --selftest
  python -I tools/frame_failure_analysis.py --root ~/data/MeViS-subset --run ~/cogrvos/task6_mevis/baseline_v0 --length-check --label "..."
  python -I tools/frame_failure_analysis.py --root ~/data/MeViS-subset --run ~/cogrvos/task6_mevis/baseline_v0 --min-run 3 \
         --out ~/cogrvos/analysis/frame_failure_mevis.csv --label "MeViS valid_u subset (12 videos, single-target sentences) - not the full benchmark"
"""
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CATS = ["reid_failure", "initial_grounding_failure", "identity_drift", "mask_propagation_failure"]
# arm: (predictions folder, grounding folder, results folder, old failure_analysis csv)
ARMS = {
    "A0": ("predictions", "grounding", "results", "failure_analysis.csv"),
    "A1": ("predictions_a1_clip", "grounding_a1_clip", "results_a1_clip", "failure_analysis_a1_clip.csv"),
    "A3-5": ("predictions_a3_n5", "grounding_a3_n5", "results_a3_n5", "failure_analysis_a3_n5.csv"),
    "A3-10": ("predictions_a3_n10", "grounding_a3_n10", "results_a3_n10", "failure_analysis_a3_n10.csv"),
    "largest": ("predictions_largest", "grounding_largest", "results_largest", "failure_analysis_largest.csv"),
}
OLD_LABEL_COLS = ("failure_mode", "mode", "label", "category", "failure")
FALLBACK_SHARE = 0.10


def bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def box_iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def mask_iou(p, g):
    u = int(np.logical_or(p, g).sum())
    return float(np.logical_and(p, g).sum()) / u if u else 0.0


def runs_of(flags):
    """maximal runs of True as (start, end_inclusive)"""
    out, i, n = [], 0, len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def analyse(g_empty, p_empty, iou_t, other_t, min_run=3, win=10, hit=0.5, lost=0.1):
    n = len(g_empty)
    vis = [not e for e in g_empty]
    failed = [vis[t] and iou_t[t] < hit for t in range(n)]
    lostf = [vis[t] and iou_t[t] < lost for t in range(n)]
    first_vis = vis.index(True) if any(vis) else None
    group = "never_visible" if first_vis is None else ("visible_f0" if first_vis == 0 else "absent_f0")
    absent_runs = runs_of(g_empty)
    returns = [e + 1 for (s, e) in absent_runs if s > 0 and e + 1 < n]  # visible on both sides of the absent run
    r = {"group": group, "n_frames": n, "n_visible": sum(vis), "first_visible": first_vis, "n_failed": sum(failed), "n_lost": sum(lostf),
         "absent_runs": len(absent_runs), "reappear_events": len(returns),
         "false_presence": sum(1 for t in range(n) if g_empty[t] and not p_empty[t]),
         "missed_presence": sum(1 for t in range(n) if vis[t] and p_empty[t]),
         "category": "no_failure", "onset": "", "onset_frac": "", "run_len": "", "recovers": "", "frac_failed_after": "",
         "similar_flag": "", "other_at_onset": ""}
    runs = [(s, e) for (s, e) in runs_of(failed) if e - s + 1 >= min_run]
    if not runs:
        return r
    s, e = runs[0]
    reid = any(rt <= s < rt + win and not any(iou_t[t] >= hit for t in range(rt, min(rt + win, n))) for rt in returns)
    if reid:
        cat = "reid_failure"
    elif s == first_vis:
        cat = "initial_grounding_failure"
    elif other_t[s] >= hit:
        cat = "identity_drift"
    else:
        cat = "mask_propagation_failure"
    vis_after = sum(1 for t in range(s, n) if vis[t])
    failed_after = sum(1 for t in range(s, n) if failed[t])
    r.update({"category": cat, "onset": s, "onset_frac": round(s / (n - 1), 4) if n > 1 else 0.0, "run_len": e - s + 1,
              "recovers": int(any(vis[t] and iou_t[t] >= hit for t in range(e + 1, n))),
              "frac_failed_after": round(failed_after / vis_after, 4) if vis_after else "",
              "similar_flag": int(other_t[s] >= hit), "other_at_onset": round(float(other_t[s]), 4)})
    return r


def selftest():
    def case(vis, iou, other=None, **kw):
        n = len(vis)
        return analyse([not v for v in vis], [False] * n, [iou[t] if vis[t] else 0.0 for t in range(n)], other or [0.0] * n, **kw)

    T, F = True, False
    c = case([T] * 8, [0.9] * 8)
    assert c["category"] == "no_failure" and c["group"] == "visible_f0", c
    c = case([T] * 10, [0.1] * 10)
    assert c["category"] == "initial_grounding_failure" and c["onset"] == 0 and c["run_len"] == 10 and c["recovers"] == 0, c
    c = case([T] * 10, [0.9] * 4 + [0.0] * 6, other=[0.0] * 4 + [0.8] * 6)
    assert c["category"] == "identity_drift" and c["onset"] == 4 and c["similar_flag"] == 1 and c["n_lost"] == 6, c
    c = case([T] * 10, [0.9] * 4 + [0.0] * 6)
    assert c["category"] == "mask_propagation_failure" and c["similar_flag"] == 0 and c["onset_frac"] == round(4 / 9, 4), c
    v = [T, T, T, F, F, F, T, T, T, T]
    c = case(v, [0.9, 0.9, 0.9, 0, 0, 0, 0.0, 0.0, 0.0, 0.0])
    assert c["category"] == "reid_failure" and c["onset"] == 6 and c["reappear_events"] == 1 and c["absent_runs"] == 1, c
    c = case(v, [0.9, 0.9, 0.9, 0, 0, 0, 0.7, 0.7, 0.7, 0.7])
    assert c["category"] == "no_failure", c
    c = case(v, [0.9, 0.9, 0.9, 0, 0, 0, 0.7, 0.0, 0.0, 0.0])  # recovered at the return frame, fails later: not re-id
    assert c["category"] == "mask_propagation_failure" and c["onset"] == 7, c
    two = [0.9] * 4 + [0.0, 0.0] + [0.9] * 2
    assert case([T] * 8, two)["category"] == "no_failure"
    c = case([T] * 8, two, min_run=2)
    assert c["category"] == "mask_propagation_failure" and c["run_len"] == 2 and c["recovers"] == 1 and c["frac_failed_after"] == round(2 / 4, 4), c
    c = case([F, F, T, T, T, T], [0.0] * 6)
    assert c["group"] == "absent_f0" and c["category"] == "initial_grounding_failure" and c["onset"] == 2 and c["first_visible"] == 2, c
    c = case([F] * 5, [0.0] * 5)
    assert c["group"] == "never_visible" and c["category"] == "no_failure", c
    c = analyse([T, F, F, T], [F, F, T, T], [0.0, 0.0, 0.0, 0.0], [0.0] * 4)  # g_empty/p_empty: frame 0 G empty, P non-empty ...
    assert c["false_presence"] == 1 and c["missed_presence"] == 1, c
    print("selftest: all checks passed")


def read_pred(path, shape):
    if not path.exists():
        return None, True
    arr = np.asarray(Image.open(path))
    m = arr.any(axis=2) if arr.ndim == 3 else arr > 0
    if shape is not None and m.shape != shape:
        sys.exit(f"shape mismatch (prediction {m.shape}, ground truth {shape}): {path}")
    return (m if m.any() else None), False


def med(v):
    return float(np.median(v)) if len(v) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--run", help="run folder with grounding*/, predictions*/, results*/")
    ap.add_argument("--out", help="CSV to write (must not exist); a .summary.txt is written next to it")
    ap.add_argument("--label")
    ap.add_argument("--tier", default="480p")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--min-run", type=int, default=3)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--hit", type=float, default=0.5)
    ap.add_argument("--lost", type=float, default=0.1)
    ap.add_argument("--length-check", action="store_true", help="ground truth only: sentence lengths, no predictions, no CSV")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if not (a.root and a.run and a.label) or not (a.out or a.length_check):
        sys.exit("need --root --run --label and (--out or --length-check)")
    from common.io_utils import read_palette_mask

    root, run = Path(a.root).expanduser(), Path(a.run).expanduser()
    outp = Path(a.out).expanduser() if a.out else None
    summ = Path(str(outp) + ".summary.txt") if outp else None
    if outp and (outp.exists() or summ.exists()):
        sys.exit(f"refusing to overwrite: {outp} or {summ}")
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("command: " + " ".join(sys.argv))
    emit(a.label)
    meta = json.load(open(root / "meta_expressions" / "meta_expressions.json"))["videos"]
    jf = {}
    for arm in a.arms:
        pth = run / ARMS[arm][2] / "per_video_results.csv"
        jf[arm] = {(r["video"], r["exp_id"]): r["JF"] for r in csv.DictReader(open(pth))} if pth.exists() else {}
    rows, info, warn = [], [], Counter()
    for gfile in sorted((run / "grounding").glob("*.json")):
        video = gfile.stem
        data = json.load(open(gfile))
        names = meta[video]["frames"]
        oids = sorted({int(e["obj_id"]) for e in data.values()})
        gt, shape = {o: [] for o in oids}, None
        for nm in names:
            p = root / "Annotations" / a.tier / video / f"{nm}.png"
            if p.exists():
                for o in oids:
                    m = np.asarray(read_palette_mask(p, o)).astype(bool)
                    shape = m.shape
                    gt[o].append(m if m.any() else None)
            else:
                warn["gt_file_missing"] += 1
                for o in oids:
                    gt[o].append(None)
        for exp_id, e in data.items():
            tg = gt[int(e["obj_id"])]
            nv = sum(1 for m in tg if m is not None)
            info.append({"video": video, "exp_id": exp_id, "n_frames": len(names), "n_visible": nv,
                         "first_visible": next((t for t, m in enumerate(tg) if m is not None), None)})
        if a.length_check:
            continue
        for arm in a.arms:
            pdir, gdir = ARMS[arm][0], ARMS[arm][1]
            gj = run / gdir / f"{video}.json"
            adata = json.load(open(gj)) if gj.exists() else {}
            for exp_id, e in data.items():
                tid = int(e["obj_id"])
                tgt = gt[tid]
                others = [gt[o] for o in oids if o != tid]
                p_empty, iou_t, other_t = [], [], []
                for t, nm in enumerate(names):
                    pm, miss = read_pred(run / pdir / video / exp_id / f"{nm}.png", shape)
                    warn[f"prediction_file_missing:{arm}"] += int(miss)
                    p_empty.append(pm is None)
                    iou_t.append(mask_iou(pm, tgt[t]) if pm is not None and tgt[t] is not None else 0.0)
                    ot = 0.0
                    if pm is not None:
                        for om in others:
                            if om[t] is not None:
                                ot = max(ot, mask_iou(pm, om[t]))
                    other_t.append(ot)
                r = analyse([m is None for m in tgt], p_empty, iou_t, other_t, a.min_run, a.window, a.hit, a.lost)
                ae = adata.get(exp_id)
                tb = bbox(tgt[0]) if tgt[0] is not None else None
                start_hit = ""
                if tb is not None:
                    if ae and ae.get("boxes"):
                        start_hit = int(box_iou([float(x) for x in ae["boxes"][0][:4]], tb) >= a.hit)
                    else:
                        warn[f"start_box_missing:{arm}"] += 1
                r.update({"video": video, "exp_id": exp_id, "arm": arm, "expression": e.get("expression", ""), "obj_id": tid,
                          "start_hit": start_hit, "JF": jf[arm].get((video, exp_id), "")})
                rows.append(r)

    # ---- A. sentence lengths (ground truth only) ----
    N = len(info)
    vis0 = sum(1 for i in info if i["first_visible"] == 0)
    abs0 = sum(1 for i in info if i["first_visible"] not in (None, 0))
    nev = sum(1 for i in info if i["first_visible"] is None)
    seen = [i for i in info if i["n_visible"] > 0]
    short = sum(1 for i in seen if i["n_visible"] < 3)
    share = short / len(seen) if seen else float("nan")
    emit(f"\nA. Sentence lengths (ground truth only). Sentences {N}: visible at frame 0 {vis0}, absent at frame 0 but visible later {abs0}, never visible {nev}")
    emit(f"   annotated frames per video (median over sentences) {med([i['n_frames'] for i in info]):.0f}; visible frames per sentence (median over sentences seen at least once) {med([i['n_visible'] for i in seen]):.0f}")
    emit(f"   sentences with only 1 or 2 visible frames: {short} of {len(seen)} ({100 * share:.1f}%). Fixed rule: more than {100 * FALLBACK_SHARE:.0f}% -> use --min-run 2, else 3. "
         f"Rule says: {'2' if share > FALLBACK_SHARE else '3'}. This run uses --min-run {a.min_run}.")
    for k, v in sorted(warn.items()):
        emit(f"   NOTE {k}: {v}")
    if a.length_check:
        emit("\nLength check only: no predictions were read.")
        if summ:
            summ.write_text("\n".join(lines) + "\n")
        return 0

    arms = a.arms
    # ---- B. categories per arm and group ----
    for group in ("visible_f0", "absent_f0"):
        emit(f"\nB. Failure category of the first failure run, group {group} (min run {a.min_run}, window {a.window}); similar = other object IoU >= {a.hit} at the onset")
        emit(f"   {'arm':8s} {'n':>4} {'none':>5} {'reid':>5} {'init':>5} {'drift':>5} {'prop':>5} {'similar':>7} {'mean J&F':>9}")
        for arm in arms:
            rs = [r for r in rows if r["arm"] == arm and r["group"] == group]
            c = Counter(r["category"] for r in rs)
            jfs = [float(r["JF"]) for r in rs if r["JF"] != ""]
            emit(f"   {arm:8s} {len(rs):>4} {c['no_failure']:>5} {c['reid_failure']:>5} {c['initial_grounding_failure']:>5} {c['identity_drift']:>5} "
                 f"{c['mask_propagation_failure']:>5} {sum(1 for r in rs if r['similar_flag'] == 1):>7} {np.mean(jfs) if jfs else float('nan'):>9.2f}")
    emit(f"   (never-visible sentences: {nev}; they cannot fail and are not in the tables)")
    # ---- C. propagation measures ----
    emit("\nC. Propagation of the first failure run, group visible_f0 (median unless stated)")
    emit(f"   {'arm':8s} {'failing':>7} {'onset pos':>9} {'onset frac':>10} {'run len':>8} {'recovers %':>10} {'failed after (mean)':>20}")
    for arm in arms:
        fl = [r for r in rows if r["arm"] == arm and r["group"] == "visible_f0" and r["category"] != "no_failure"]
        emit(f"   {arm:8s} {len(fl):>7} {med([r['onset'] for r in fl]):>9.1f} {med([r['onset_frac'] for r in fl]):>10.2f} {med([r['run_len'] for r in fl]):>8.1f} "
             f"{100 * np.mean([r['recovers'] for r in fl]) if fl else float('nan'):>10.1f} {np.mean([r['frac_failed_after'] for r in fl if r['frac_failed_after'] != '']) if fl else float('nan'):>20.2f}")
    emit("   later-onset failures only (reid, drift, propagation: the target was found first and then lost):")
    for arm in arms:
        lt = [r for r in rows if r["arm"] == arm and r["group"] == "visible_f0" and r["category"] in ("reid_failure", "identity_drift", "mask_propagation_failure")]
        emit(f"   {arm:8s} n {len(lt):>3}  median onset fraction {med([r['onset_frac'] for r in lt]):.2f}  recovers {100 * np.mean([r['recovers'] for r in lt]) if lt else float('nan'):.1f}%")
    # ---- D. events ----
    emit("\nD. Occlusion / disappearance EVENTS (all groups, summed over sentences)")
    emit(f"   {'arm':8s} {'absent runs':>11} {'reappearances':>13} {'sentences with >=1':>19} {'false-presence frames':>22} {'missed-presence frames':>23}")
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        emit(f"   {arm:8s} {sum(r['absent_runs'] for r in rs):>11} {sum(r['reappear_events'] for r in rs):>13} {sum(1 for r in rs if r['reappear_events'] > 0):>19} "
             f"{sum(r['false_presence'] for r in rs):>22} {sum(r['missed_presence'] for r in rs):>23}")
    # ---- E. start box versus outcome ----
    emit("\nE. Start box (box IoU >= 0.5 with the ground-truth box at frame 0) versus the later outcome, group visible_f0")
    emit(f"   {'arm':8s} {'start':>6} {'n':>4} {'no failure':>10} {'initial':>8} {'later':>6} {'mean J&F':>9}")
    for arm in arms:
        for sh, name in ((1, "right"), (0, "wrong")):
            rs = [r for r in rows if r["arm"] == arm and r["group"] == "visible_f0" and r["start_hit"] == sh]
            c = Counter(r["category"] for r in rs)
            jfs = [float(r["JF"]) for r in rs if r["JF"] != ""]
            emit(f"   {arm:8s} {name:>6} {len(rs):>4} {c['no_failure']:>10} {c['initial_grounding_failure']:>8} {len(rs) - c['no_failure'] - c['initial_grounding_failure']:>6} "
                 f"{np.mean(jfs) if jfs else float('nan'):>9.2f}")
    # ---- F. existing taxonomy next to the new one ----
    emit("\nF. Existing sentence-level labels (failure_analysis*.csv, unchanged), counted per arm; not merged with B")
    for arm in arms:
        pth = run / ARMS[arm][3]
        if not pth.exists():
            emit(f"   {arm:8s} file not found: {pth.name}")
            continue
        rd = list(csv.DictReader(open(pth)))
        col = next((c for c in OLD_LABEL_COLS if rd and c in rd[0]), None)
        if col is None:
            emit(f"   {arm:8s} no label column found; columns: {list(rd[0]) if rd else 'empty file'}")
        else:
            emit(f"   {arm:8s} " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r[col] for r in rd).items())))
    emit("\nSingle run on a subset; descriptive. Sentences of one video are not independent. Categories follow the fixed rules above, not pictures.")

    cols = ["video", "exp_id", "arm", "expression", "obj_id", "group", "n_frames", "n_visible", "first_visible", "start_hit", "JF", "category", "onset", "onset_frac",
            "run_len", "recovers", "frac_failed_after", "similar_flag", "other_at_onset", "n_failed", "n_lost", "absent_runs", "reappear_events",
            "false_presence", "missed_presence"]
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summ.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {outp} ({len(rows)} rows) and {summ}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
