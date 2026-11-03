#!/usr/bin/env python3
"""
Task 6 failure analysis of arm A3 (Amendment 8 item 5; CPU only, reads only, no tracking, no Gemini). Per sentence, from saved files only: the sentence
text, whether the cached pool contains the target (a box with IoU >= --hit with the tight box of the ground-truth mask at frame 0), whether A0 / A1 / A3 chose a
hit, the J&F of the three arms, and for every wrong pick when the pool does contain the target one reason:
  (a) similar_object = the chosen box has IoU >= --hit with the ground-truth box of ANOTHER annotated object of the same video (annotated = has a sentence in the
      cached folder);
  (b) mask_quality (A3 only) = the target candidate scored 0 on at least one of its N tracked frames (mask under 200 pixels or empty) and its mean over its
      non-zero frames is higher than the score of the candidate that won, i.e. it would have won without those empty frames;
  (c) other = neither of these.
If (a) and (b) both hold the primary reason is (a); both flags are written to the CSV and counted. Sentences whose target is absent at frame 0 and sentences whose
pool does not contain the target are not selection failures and are labelled separately. The ground truth is used for analysis only.

  python -I tools/a3_failure_analysis.py --root ~/data/MeViS-subset --run ~/cogrvos/task6_mevis/baseline_v0 --audit ~/cogrvos/audit/audit_grounding_mevis.csv \
         --n 5 --out ~/cogrvos/analysis/a3_n5_failure_mevis.csv --label "MeViS valid_u subset (12 videos, single-target sentences) - not the full benchmark"
"""
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io_utils import read_palette_mask  # noqa: E402


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


def load_jf(path):
    return {(r["video"], r["exp_id"]): float(r["JF"]) for r in csv.DictReader(open(path))}


def load_csv(path):
    return {(r["video"], r["exp_id"]): r for r in csv.DictReader(open(path))}


def frame_scores(cell, n):
    """'a,b,c|d,e,f' (candidates separated by |, frames by ,) -> list of lists, first n frames of each candidate; [] for an empty cell."""
    if not cell:
        return []
    return [[float(x) for x in cand.split(",")][:n] for cand in cell.split("|")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", required=True, help="run folder with grounding/, grounding_a1_clip/, grounding_a3_n<N>/, results*/")
    ap.add_argument("--audit", required=True, help="only used to check that the sentence set is the same as in the audit")
    ap.add_argument("--n", type=int, required=True, choices=[5, 10], help="A3 arm N")
    ap.add_argument("--out", required=True, help="CSV to write (must not exist)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--tier", default="480p")
    ap.add_argument("--hit", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=6, help="how many biggest gains and losses to list")
    a = ap.parse_args()

    root, run, outp = Path(a.root).expanduser(), Path(a.run).expanduser(), Path(a.out).expanduser()
    if outp.exists():
        sys.exit(f"refusing to overwrite: {outp}")
    outp.parent.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(root / "meta_expressions" / "meta_expressions.json"))["videos"]
    sel3 = load_csv(run / f"grounding_a3_n5" / "selection_a3.csv")
    sel1 = load_csv(run / "grounding_a1_clip" / "selection_a1.csv")
    jf0, jf1, jf3 = (load_jf(run / d / "per_video_results.csv") for d in ("results", "results_a1_clip", f"results_a3_n{a.n}"))
    audit_keys = {(r["video"], r["exp_id"]) for r in csv.DictReader(open(Path(a.audit).expanduser()))}

    rows, mismatch = [], 0
    for gfile in sorted((run / "grounding").glob("*.json")):
        video = gfile.stem
        data = json.load(open(gfile))
        f0 = root / "Annotations" / a.tier / video / f"{meta[video]['frames'][0]}.png"
        gt = {oid: bbox(np.asarray(read_palette_mask(f0, oid)).astype(bool)) for oid in sorted({int(e["obj_id"]) for e in data.values()})}
        for exp_id, e in data.items():
            k = (video, exp_id)
            boxes = [list(map(float, b[:4])) for b in e["boxes"]]
            tid = int(e["obj_id"])
            tb = gt[tid]
            r = {"video": video, "exp_id": exp_id, "expression": e["expression"], "n_candidates": len(boxes), "visible_f0": int(tb is not None)}
            p1, p3 = int(sel1[k]["chosen_pos"]), int(sel3[k][f"chosen_pos_n{a.n}"])
            fs = frame_scores(sel3[k]["frame_scores"], a.n)
            means = [float(np.mean(f)) for f in fs]
            if means:  # self-check: the stored choice must be the highest mean, a tie going to the lower position
                best = 0
                for i, m in enumerate(means):
                    if m > means[best] + 1e-3:
                        best = i
                mismatch += int(best != p3)
            r.update({"a1_pos": p1, "a3_pos": p3, "JF_A0": jf0[k], "JF_A1": jf1[k], "JF_A3": jf3[k], "dJF_A3_minus_A0": jf3[k] - jf0[k], "dJF_A3_minus_A1": jf3[k] - jf1[k]})
            if tb is None:
                r.update({"pool_has_target": "", "a0_hit": "", "a1_hit": "", "a3_hit": "", "a3_reason": "absent_f0", "a1_reason": "absent_f0", "a0_reason": "absent_f0"})
                rows.append(r)
                continue
            ious = [box_iou(b, tb) for b in boxes]
            tpos = int(np.argmax(ious)) if ious and max(ious) >= a.hit else None
            r["pool_has_target"] = int(tpos is not None)
            others = [g for oid, g in gt.items() if oid != tid and g is not None]
            sim = lambda pos: any(box_iou(boxes[pos], g) >= a.hit for g in others)  # noqa: E731
            for name, pos in (("a0", 0), ("a1", p1), ("a3", p3)):
                hit = bool(ious) and ious[pos] >= a.hit
                r[f"{name}_hit"] = int(hit)
                if hit:
                    r[f"{name}_reason"] = "hit"
                elif tpos is None:
                    r[f"{name}_reason"] = "target_not_in_pool"
                else:
                    r[f"{name}_reason"] = "similar_object" if sim(pos) else "other"
            # A3 only: the mask-quality flag
            r["a3_sim_flag"] = r["a3_mask_flag"] = ""
            if tpos is not None and not r["a3_hit"]:
                r["a3_sim_flag"] = int(sim(p3))
                mq = 0
                if fs and tpos < len(fs):
                    zeros = sum(1 for v in fs[tpos] if v == 0.0)
                    nz = [v for v in fs[tpos] if v != 0.0]
                    mq = int(zeros > 0 and len(nz) > 0 and float(np.mean(nz)) > means[p3])
                    r["target_zero_frames"], r["target_nonzero_mean"] = zeros, (round(float(np.mean(nz)), 4) if nz else "")
                r["a3_mask_flag"] = mq
                if not r["a3_sim_flag"] and mq:
                    r["a3_reason"] = "mask_quality"
            r["target_pos"] = tpos
            r["a3_score_chosen"] = round(means[p3], 4) if means else ""
            r["a3_score_target"] = round(means[tpos], 4) if means and tpos is not None and tpos < len(means) else ""
            rows.append(r)

    cols = ["video", "exp_id", "expression", "n_candidates", "visible_f0", "pool_has_target", "target_pos", "a0_hit", "a1_pos", "a1_hit", "a3_pos", "a3_hit",
            "a0_reason", "a1_reason", "a3_reason", "a3_sim_flag", "a3_mask_flag", "target_zero_frames", "target_nonzero_mean", "a3_score_chosen", "a3_score_target",
            "JF_A0", "JF_A1", "JF_A3", "dJF_A3_minus_A0", "dJF_A3_minus_A1"]
    with open(outp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    vis = [r for r in rows if r["visible_f0"]]
    print(a.label)
    print(f"A3 N={a.n}: sentences {len(rows)} (audit {len(audit_keys)}), target visible at frame 0: {len(vis)}, pool contains the target: {sum(r['pool_has_target'] for r in vis)}; "
          f"self-check: stored A3 choice differs from the highest mean in {mismatch} sentences")
    print(f"\n{'outcome of the pick (visible sentences)':42s} {'A0':>5} {'A1':>5} {'A3':>5}")
    for key in ("hit", "similar_object", "other", "target_not_in_pool"):
        c = [sum(1 for r in vis if r[f"{n}_reason"] == key) for n in ("a0", "a1")]
        c3 = sum(1 for r in vis if r["a3_reason"] == key)
        if key == "other":  # A3 splits 'other' into mask_quality and other
            c3 = sum(1 for r in vis if r["a3_reason"] in ("other",))
        print(f"{key:42s} {c[0]:>5} {c[1]:>5} {c3:>5}")
    print(f"{'mask_quality (A3 only; A0 and A1 not defined)':42s} {'':>5} {'':>5} {sum(1 for r in vis if r['a3_reason'] == 'mask_quality'):>5}")
    wrong = [r for r in vis if r["pool_has_target"] and not r["a3_hit"]]
    both = sum(1 for r in wrong if r["a3_sim_flag"] == 1 and r["a3_mask_flag"] == 1)
    print(f"A3 wrong picks with the target in the pool: {len(wrong)}; similar-object flag {sum(1 for r in wrong if r['a3_sim_flag'] == 1)}, "
          f"mask-quality flag {sum(1 for r in wrong if r['a3_mask_flag'] == 1)}, both flags {both}")

    print(f"\nA3 hit versus A0 hit (visible sentences) and the J&F change A3 - A0")
    g = defaultdict(list)
    for r in vis:
        g[(r["a0_hit"], r["a3_hit"])].append(r["dJF_A3_minus_A0"])
    for (h0, h3), name in (((0, 1), "A3 gained the hit"), ((1, 0), "A3 lost the hit"), ((1, 1), "both hit"), ((0, 0), "both miss")):
        v = g.get((h0, h3), [])
        print(f"  {name:20s} n {len(v):>3}   mean J&F change {np.mean(v) if v else float('nan'):>+7.2f}")

    ordered = sorted(rows, key=lambda r: r["dJF_A3_minus_A0"], reverse=True)
    for title, sel in (("biggest J&F gains of A3 over A0", ordered[:a.top]), ("biggest J&F losses of A3 against A0", ordered[::-1][:a.top])):
        print(f"\n{title} (picked after seeing the results):")
        for r in sel:
            print(f"  {r['video']} exp {r['exp_id']:>3}  dJ&F {r['dJF_A3_minus_A0']:>+7.2f}  A0 hit {r['a0_hit']!s:>2} A3 hit {r['a3_hit']!s:>2}  A3 reason {r['a3_reason']:<18s} {r['expression'][:80]}")
    print("\nSingle run on a subset; descriptive. Reasons are assigned by the fixed rules above from the saved files, not by looking at the pictures.")


if __name__ == "__main__":
    sys.exit(main())
