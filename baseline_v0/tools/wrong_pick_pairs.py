#!/usr/bin/env python3
"""Task 1 (Sir, 4 Nov): which object does a wrong frame-0 pick land on? CPU only; reads saved files only (no model, no GPU, no Gemini).

For every sentence whose target is visible at frame 0 and whose A0 top-1 box is NOT correct (box IoU with the tight ground-truth box < 0.5), the tool finds
the other ANNOTATED object (an object that has a sentence in grounding/) whose ground-truth box at frame 0 overlaps the top-1 box most (box IoU >= --hit),
and writes one row with: the target sentence, the picked object's own sentences, the words the two texts share, the audit class, and whether each other arm
(A1, A3-5, A3-10, largest) fixes the pick. The column "code" is left EMPTY on purpose: the reading "same-kind distractor / reference object named in the
sentence / unclear" is a manual coding from the sentence text, done after the coding rules are written down. No object names or categories exist in the data.
Consistency checks (printed as NOTE): A0 top-1 IoU recomputed here versus the audit CSV (tolerance 0.001).
Reuses box_iou() and bbox() of the committed tools/frame_failure_analysis.py unchanged.

  python -I tools/wrong_pick_pairs.py --root ~/data/MeViS-subset --run ~/cogrvos/task6_mevis/baseline_v0 --audit ~/cogrvos/audit/audit_grounding_mevis.csv \
         --out ~/cogrvos/analysis/wrong_pick_pairs_mevis.csv --label "MeViS valid_u subset (12 videos, single-target sentences) - not the full benchmark"
Writes OUT and OUT.summary.txt. Refuses to overwrite.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frame_failure_analysis as ffa  # committed tool, unchanged; its import also puts the repo root on sys.path

STOP = set("a an the of on in at to and or with its it is are was were that this those these his her their them he she they from by for as be being been has have "
           "had while who which what up down out over under into onto off than then there here one two another other".split())
ARM_DIRS = [("a1", "grounding_a1_clip"), ("a3n5", "grounding_a3_n5"), ("a3n10", "grounding_a3_n10"), ("largest", "grounding_largest")]
FIELDS = ["video", "exp_id", "obj_id", "expression", "audit_class", "top1_iou_target", "n_candidates", "best_cand_rank", "picked_obj", "picked_box_iou",
          "picked_expressions", "shared_words", "a1_hit", "a3n5_hit", "a3n10_hit", "largest_hit", "code"]


def words(s):
    return [w for w in re.findall(r"[a-z]+", s.lower()) if len(w) >= 3 and w not in STOP]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tier", default="480p")
    ap.add_argument("--hit", type=float, default=0.5)
    a = ap.parse_args()
    from common.io_utils import read_palette_mask

    root, run = Path(a.root).expanduser(), Path(a.run).expanduser()
    outp = Path(a.out).expanduser()
    summ = Path(str(outp) + ".summary.txt")
    if outp.exists() or summ.exists():
        sys.exit(f"refusing to overwrite: {outp} or {summ}")
    meta = json.load(open(root / "meta_expressions" / "meta_expressions.json"))["videos"]
    audit = list(csv.DictReader(open(Path(a.audit).expanduser())))
    by_video = {}
    for r in audit:
        by_video.setdefault(r["video"], []).append(r)
    rows, notes, visible = [], Counter(), 0
    for video, ar in sorted(by_video.items()):
        names = meta[video]["frames"]
        g0 = json.load(open(run / "grounding" / f"{video}.json"))
        oids = sorted({int(e["obj_id"]) for e in g0.values()})
        p = root / "Annotations" / a.tier / video / f"{names[0]}.png"
        gtb = {}
        for o in oids:
            m = np.asarray(read_palette_mask(p, o)).astype(bool)
            gtb[o] = ffa.bbox(m) if m.any() else None
        sents = {}
        for e in g0.values():
            sents.setdefault(int(e["obj_id"]), []).append(e["expression"].strip())
        arm_g = {n: (json.load(open(run / d / f"{video}.json")) if (run / d / f"{video}.json").exists() else {}) for n, d in ARM_DIRS}

        def top(g, eid):
            e = g.get(eid)
            return [float(x) for x in e["boxes"][0][:4]] if e and e.get("boxes") else None

        for r in ar:
            if r["visible_at_f0"] != "1":
                continue
            visible += 1
            eid, tid = r["exp_id"], int(r["obj_id"])
            tb = gtb[tid]

            def iou_t(b):
                return ffa.box_iou(b, tb) if b is not None and tb is not None else 0.0

            b0 = top(g0, eid)
            i0 = iou_t(b0)
            if abs(i0 - float(r["top1_iou_target"])) > 0.001:
                notes["a0_top1_iou_differs_from_audit"] += 1
            if i0 >= a.hit:
                continue
            po, pv = None, 0.0
            for o, gb in gtb.items():
                if o != tid and gb is not None and b0 is not None:
                    v = ffa.box_iou(b0, gb)
                    if v > pv:
                        po, pv = o, v
            found = po is not None and pv >= a.hit
            pex = sents.get(po, []) if found else []
            shared = sorted(set(words(r["expression"])) & {w for s in pex for w in words(s)}) if found else []
            hits = {n: int(iou_t(top(g, eid)) >= a.hit) for n, g in arm_g.items()}
            notes[(r["top1_class"], "picked object found" if found else "no annotated object >= hit")] += 1
            rows.append({"video": video, "exp_id": eid, "obj_id": tid, "expression": r["expression"].strip(), "audit_class": r["top1_class"],
                         "top1_iou_target": round(i0, 4), "n_candidates": r["n_candidates"], "best_cand_rank": r["best_cand_rank"],
                         "picked_obj": po if found else "", "picked_box_iou": round(pv, 4) if found else "", "picked_expressions": " | ".join(pex),
                         "shared_words": " ".join(shared), "a1_hit": hits["a1"], "a3n5_hit": hits["a3n5"], "a3n10_hit": hits["a3n10"],
                         "largest_hit": hits["largest"], "code": ""})
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("command: " + " ".join(sys.argv))
    emit(a.label)
    emit(f"sentences visible at frame 0: {visible}; A0 top-1 not correct (box IoU < {a.hit}): {len(rows)}. Single run on a subset; descriptive.")
    for k, v in sorted(notes.items(), key=lambda kv: str(kv[0])):
        emit(f"   {k}: {v}")
    pk = [r for r in rows if r["picked_obj"] != ""]
    emit(f"wrong picks that land on another annotated object: {len(pk)}")
    emit(f"   with at least one shared content word between the target sentence and the picked object's sentences: {sum(1 for r in pk if r['shared_words'])}")
    emit(f"   fixed by (arm top-1 box becomes correct): A1 {sum(r['a1_hit'] for r in pk)}, A3-5 {sum(r['a3n5_hit'] for r in pk)}, "
         f"A3-10 {sum(r['a3n10_hit'] for r in pk)}, largest {sum(r['largest_hit'] for r in pk)}")
    emit(f"   target not in the pool at all (best_cand_rank empty or negative): {sum(1 for r in pk if r['best_cand_rank'] in ('', '-1'))}")
    emit("The column 'code' is empty: coding is manual, from sentence text, after the coding rules are written down.")
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    summ.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {outp} ({len(rows)} rows) and {summ}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
