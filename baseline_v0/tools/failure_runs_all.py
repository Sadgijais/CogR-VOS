#!/usr/bin/env python3
"""Task 3 extension (Sir, 4 Nov): ALL failure runs of a sentence, not only the first. CPU only; reads saved files only (no tracking, no GPU, no Gemini).

The committed tools/frame_failure_analysis.py classifies only the FIRST failure run of a sentence. This tool reuses its functions unchanged (imported, not copied)
and applies the SAME fixed rules to every failure run (3 or more consecutive visible failed frames, IoU < 0.5):
  reid_failure | initial_grounding_failure (only a first run can be this) | identity_drift (onset after the start and other_t >= 0.5 at the onset) |
  mask_propagation_failure.
In addition, for every run it records whether another ANNOTATED object reaches IoU >= 0.5 with the prediction in ANY frame of the run (switch_in_run). This shows
switches that start after the onset (target lost first, another object taken later), which the onset-only rule cannot see.
First runs and later runs ("initial vs subsequent") are counted separately. Consistency check: the category of run 1 must equal the category from the committed
tool's analyse(); any mismatch is printed as a NOTE (expected: none).
Groups are kept apart: visible_f0 (target visible at frame 0), absent_f0 (first visible later). never_visible sentences cannot fail.
Descriptive only: single run, subsets, sentences of one video are not independent. --others annotated (default): other_t knows only objects with a sentence in grounding/.
--others all: other_t uses EVERY non-zero object id in the annotation PNGs of the video, as the old drift rule (common/failure_taxonomy, stage4) does.
Extra switch cut: 'switch>=min' = runs in which another object has IoU >= hit in at least min-run frames of the run (the loose flag needs only one frame).

  python -I tools/failure_runs_all.py --selftest
  python -I tools/failure_runs_all.py --root ~/data/MeViS-subset --run ~/cogrvos/task6_mevis/baseline_v0 --min-run 3 \
         --others all --out ~/cogrvos/analysis/failure_runs_all_allobj_mevis.csv --label "MeViS valid_u subset (12 videos, single-target sentences) - not the full benchmark"
Writes OUT (one row per failure run), OUT.sentences.csv (one row per sentence and arm) and OUT.summary.txt. Refuses to overwrite.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frame_failure_analysis as ffa  # committed tool, unchanged; its import also puts the repo root on sys.path

CATS = ["reid_failure", "initial_grounding_failure", "identity_drift", "mask_propagation_failure"]
RUN_FIELDS = ["video", "exp_id", "arm", "group", "obj_id", "run_idx", "onset", "run_len", "category", "other_at_onset", "other_max_in_run",
              "switch_in_run", "n_other_frames", "first_switch_pos", "recovers_after"]
SENT_FIELDS = ["video", "exp_id", "arm", "group", "obj_id", "n_runs", "first_category", "n_reid", "n_init", "n_drift", "n_prop", "n_runs_with_switch", "n_runs_switch_min"]


def classify_runs(g_empty, iou_t, other_t, min_run=3, win=10, hit=0.5):
    n = len(g_empty)
    vis = [not e for e in g_empty]
    failed = [vis[t] and iou_t[t] < hit for t in range(n)]
    first_vis = vis.index(True) if any(vis) else None
    returns = [e + 1 for (s, e) in ffa.runs_of(g_empty) if s > 0 and e + 1 < n]
    out = []
    runs = [(s, e) for (s, e) in ffa.runs_of(failed) if e - s + 1 >= min_run]
    for k, (s, e) in enumerate(runs, 1):
        if any(rt <= s < rt + win and not any(iou_t[t] >= hit for t in range(rt, min(rt + win, n))) for rt in returns):
            cat = "reid_failure"
        elif s == first_vis:
            cat = "initial_grounding_failure"
        elif other_t[s] >= hit:
            cat = "identity_drift"
        else:
            cat = "mask_propagation_failure"
        sw = [t for t in range(s, e + 1) if other_t[t] >= hit]
        out.append({"run_idx": k, "onset": s, "run_len": e - s + 1, "category": cat, "other_at_onset": round(float(other_t[s]), 4),
                    "other_max_in_run": round(float(max(other_t[s:e + 1])), 4), "switch_in_run": int(bool(sw)), "n_other_frames": len(sw),
                    "first_switch_pos": (sw[0] - s) if sw else "",
                    "recovers_after": int(any(vis[t] and iou_t[t] >= hit for t in range(e + 1, n)))})
    return out


def selftest():
    T, F = True, False

    def case(vis, iou, other=None, **kw):
        n = len(vis)
        g = [not v for v in vis]
        i = [iou[t] if vis[t] else 0.0 for t in range(n)]
        o = other or [0.0] * n
        runs = classify_runs(g, i, o, **kw)
        base = ffa.analyse(g, [False] * n, i, o, **kw)
        first = runs[0]["category"] if runs else "no_failure"
        assert first == base["category"], (first, base["category"])
        return runs

    assert case([T] * 8, [0.9] * 8) == []
    r = case([T] * 10, [0.1] * 10, other=[0.8] * 10)
    assert len(r) == 1 and r[0]["category"] == "initial_grounding_failure" and r[0]["switch_in_run"] == 1 and r[0]["first_switch_pos"] == 0, r
    r = case([T] * 10, [0.9] * 4 + [0.0] * 6, other=[0.0] * 6 + [0.8] * 4)  # lost first, another object only from the third failed frame on
    assert len(r) == 1 and r[0]["category"] == "mask_propagation_failure" and r[0]["other_at_onset"] == 0.0 and r[0]["switch_in_run"] == 1 \
        and r[0]["first_switch_pos"] == 2 and r[0]["n_other_frames"] == 4, r
    r = case([T] * 11, [0.9] * 2 + [0.0] * 3 + [0.9] * 3 + [0.0] * 3, other=[0.0] * 8 + [0.8] * 3)  # a second run that is identity drift
    assert [x["category"] for x in r] == ["mask_propagation_failure", "identity_drift"] and [x["run_idx"] for x in r] == [1, 2], r
    assert r[0]["switch_in_run"] == 0 and r[1]["switch_in_run"] == 1 and r[0]["recovers_after"] == 1 and r[1]["recovers_after"] == 0, r
    v = [T, T, T, F, F, F, T, T, T, T]
    r = case(v, [0.9, 0.9, 0.9, 0, 0, 0, 0.0, 0.0, 0.0, 0.0])
    assert len(r) == 1 and r[0]["category"] == "reid_failure" and r[0]["onset"] == 6, r
    r = case([T] * 8, [0.9] * 4 + [0.0, 0.0] + [0.9] * 2)  # run of 2 is below min_run 3
    assert r == [], r
    r = case([T] * 8, [0.9] * 4 + [0.0, 0.0] + [0.9] * 2, min_run=2)
    assert len(r) == 1 and r[0]["run_len"] == 2, r
    print("selftest: all checks passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--run")
    ap.add_argument("--out")
    ap.add_argument("--label")
    ap.add_argument("--tier", default="480p")
    ap.add_argument("--arms", nargs="+", default=list(ffa.ARMS), choices=list(ffa.ARMS))
    ap.add_argument("--min-run", type=int, default=3)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--hit", type=float, default=0.5)
    ap.add_argument("--others", choices=["annotated", "all"], default="annotated")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if not (a.root and a.run and a.out and a.label):
        sys.exit("need --root --run --out --label")
    from common.io_utils import read_palette_mask

    root, run = Path(a.root).expanduser(), Path(a.run).expanduser()
    outp = Path(a.out).expanduser()
    sentp, summ = Path(str(outp) + ".sentences.csv"), Path(str(outp) + ".summary.txt")
    if any(p.exists() for p in (outp, sentp, summ)):
        sys.exit(f"refusing to overwrite: {outp}, {sentp} or {summ}")
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("command: " + " ".join(sys.argv))
    emit(a.label)
    emit(f"other-object pool: {a.others}")
    meta = json.load(open(root / "meta_expressions" / "meta_expressions.json"))["videos"]
    run_rows, sent_rows, warn = [], [], Counter()
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
        pal, area, nid = None, None, 0
        if a.others == "all":
            pal = []
            for nm in names:
                p = root / "Annotations" / a.tier / video / f"{nm}.png"
                if p.exists():
                    img = Image.open(p)
                    if img.mode != "P":
                        img = img.convert("P")
                    pal.append(np.array(img))
                    nid = max(nid, int(pal[-1].max()))
                else:
                    pal.append(None)
            area = [None if q is None else np.bincount(q.ravel(), minlength=nid + 1) for q in pal]
        for arm in a.arms:
            pdir = ffa.ARMS[arm][0]
            for exp_id, e in data.items():
                tid = int(e["obj_id"])
                tgt = gt[tid]
                others = [gt[o] for o in oids if o != tid]
                p_empty, iou_t, other_t = [], [], []
                for t, nm in enumerate(names):
                    pm, miss = ffa.read_pred(run / pdir / video / exp_id / f"{nm}.png", shape)
                    warn[f"prediction_file_missing:{arm}"] += int(miss)
                    p_empty.append(pm is None)
                    iou_t.append(ffa.mask_iou(pm, tgt[t]) if pm is not None and tgt[t] is not None else 0.0)
                    ot = 0.0
                    if pm is not None and a.others == "all":
                        if pal[t] is not None:
                            inter = np.bincount(pal[t][pm], minlength=nid + 1)
                            union = int(pm.sum()) + area[t] - inter
                            ious = inter / np.maximum(union, 1)
                            ious[0] = 0.0
                            if tid <= nid:
                                ious[tid] = 0.0
                            ot = float(ious.max())
                    elif pm is not None:
                        for om in others:
                            if om[t] is not None:
                                ot = max(ot, ffa.mask_iou(pm, om[t]))
                    other_t.append(ot)
                g_empty = [m is None for m in tgt]
                runs = classify_runs(g_empty, iou_t, other_t, a.min_run, a.window, a.hit)
                base = ffa.analyse(g_empty, p_empty, iou_t, other_t, a.min_run, a.window, a.hit)
                first = runs[0]["category"] if runs else "no_failure"
                if first != base["category"]:
                    warn["first_run_mismatch_with_committed_tool"] += 1
                c = Counter(r["category"] for r in runs)
                key = dict(video=video, exp_id=exp_id, arm=arm, group=base["group"], obj_id=tid)
                for r in runs:
                    run_rows.append({**key, **r})
                sent_rows.append({**key, "n_runs": len(runs), "first_category": first, "n_reid": c["reid_failure"], "n_init": c["initial_grounding_failure"],
                                  "n_drift": c["identity_drift"], "n_prop": c["mask_propagation_failure"],
                                  "n_runs_with_switch": sum(r["switch_in_run"] for r in runs),
                                  "n_runs_switch_min": sum(1 for r in runs if r["n_other_frames"] >= a.min_run)})
    for k, v in sorted(warn.items()):
        emit(f"NOTE {k}: {v}")
    emit(f"min run {a.min_run}, window {a.window}, hit {a.hit}. Single run on a subset; descriptive. Sentences of one video are not independent.")
    emit("switch = another object (pool: " + a.others + ") has IoU >= hit with the prediction in at least one frame of the failure run; switch>=min = in at least min-run frames.")
    for group in ("visible_f0", "absent_f0"):
        emit(f"\n=== group {group} ===")
        emit("C1. failure runs per arm")
        emit(f"   {'arm':8s} {'sentences':>9} {'with>=1 run':>11} {'with>=2 runs':>12} {'runs':>5} {'first runs':>10} {'later runs':>10}")
        for arm in a.arms:
            ss = [r for r in sent_rows if r["arm"] == arm and r["group"] == group]
            emit(f"   {arm:8s} {len(ss):>9} {sum(1 for r in ss if r['n_runs'] >= 1):>11} {sum(1 for r in ss if r['n_runs'] >= 2):>12} "
                 f"{sum(r['n_runs'] for r in ss):>5} {sum(1 for r in ss if r['n_runs'] >= 1):>10} {sum(max(r['n_runs'] - 1, 0) for r in ss):>10}")
        emit("C2. category of first runs and of later (2nd, 3rd, ...) runs; 'switch' = runs with another object (pool above) >= hit in some frame of the run")
        emit(f"   {'arm':8s} {'which':>6} {'runs':>5} {'reid':>5} {'init':>5} {'drift':>5} {'prop':>5} {'switch':>7} {'switch, not drift':>18} {'switch>=min':>12}")
        for arm in a.arms:
            rr = [r for r in run_rows if r["arm"] == arm and r["group"] == group]
            for which, sel in (("first", [r for r in rr if r["run_idx"] == 1]), ("later", [r for r in rr if r["run_idx"] > 1])):
                c = Counter(r["category"] for r in sel)
                emit(f"   {arm:8s} {which:>6} {len(sel):>5} {c['reid_failure']:>5} {c['initial_grounding_failure']:>5} {c['identity_drift']:>5} "
                     f"{c['mask_propagation_failure']:>5} {sum(r['switch_in_run'] for r in sel):>7} "
                     f"{sum(1 for r in sel if r['switch_in_run'] and r['category'] != 'identity_drift'):>18} "
                     f"{sum(1 for r in sel if r['n_other_frames'] >= a.min_run):>12}")
        emit("C3. sentences: drift in ANY run, and a switch in ANY run")
        emit(f"   {'arm':8s} {'sentences':>9} {'drift run (any)':>15} {'switch (any run)':>16} {'switch, no drift run':>20} {'switch only after loss':>22} {'switch>=min (any)':>18}")
        for arm in a.arms:
            ss = [r for r in sent_rows if r["arm"] == arm and r["group"] == group]
            after = 0
            for r in ss:
                mine = [x for x in run_rows if x["arm"] == arm and x["video"] == r["video"] and x["exp_id"] == r["exp_id"]]
                if any(x["switch_in_run"] and x["category"] == "mask_propagation_failure" and x["first_switch_pos"] != 0 for x in mine):
                    after += 1
            emit(f"   {arm:8s} {len(ss):>9} {sum(1 for r in ss if r['n_drift'] > 0):>15} {sum(1 for r in ss if r['n_runs_with_switch'] > 0):>16} "
                 f"{sum(1 for r in ss if r['n_runs_with_switch'] > 0 and r['n_drift'] == 0):>20} {after:>22} "
                 f"{sum(1 for r in ss if r['n_runs_switch_min'] > 0):>18}")
    emit("\n'switch only after loss' = a propagation-failure run in which another object is reached only after the first failed frame (first_switch_pos > 0).")
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RUN_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(run_rows)
    with open(sentp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SENT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sent_rows)
    summ.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {outp} ({len(run_rows)} runs), {sentp} ({len(sent_rows)} sentence rows) and {summ}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
