#!/usr/bin/env python3
"""
Task 6: choose failure cases BY RULE (no hand-picking) from files a finished run already wrote. CPU only, reads files,
writes two small files into --out-dir. Rule (fixed here, before looking at the result):
  1. the 3 best and the 3 worst sentences by J&F (ties broken by video, exp_id),
  2. every sentence that has a disappearance event in recovery_events.csv (any 'kind' other than 'loss_episode'),
     tagged 'failed_recovery' when that event has recovered == False,
  3. for each failure_mode other than 'ok' in the failure-analysis csv, the 3 sentences with the lowest J&F of that mode.
Also writes counts over ALL sentences (failure modes, distractor confusion = drifted onto another object, loss episodes,
reappearance events). A sentence can have several tags.
"""
import argparse, csv, os
from collections import Counter, defaultdict


def load(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results_x/per_video_results.csv")
    ap.add_argument("--failure", required=True, help="failure_analysis_x.csv")
    ap.add_argument("--recovery", required=True, help="recovery_x/recovery_events.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()

    pv = load(a.results)
    fa = {(r["video"], r["exp_id"]): r for r in load(a.failure)}
    rec = load(a.recovery)
    by_rec = defaultdict(list)
    for r in rec:
        by_rec[(r["video"], r["exp_id"])].append(r)

    def key(r):
        return (r["video"], r["exp_id"])

    ranked = sorted(pv, key=lambda r: (-num(r["JF"]), r["video"], num(r["exp_id"])))
    tags = defaultdict(set)
    for r in ranked[: a.n]:
        tags[key(r)].add("best")
    for r in ranked[-a.n:]:
        tags[key(r)].add("worst")
    for k, rows in by_rec.items():
        for r in rows:
            if r["kind"] != "loss_episode":
                tags[k].add("disappearance")
                if str(r.get("recovered", "")).strip() == "False":
                    tags[k].add("failed_recovery")
    modes = Counter(fa[key(r)]["failure_mode"] for r in pv if key(r) in fa)
    for mode in sorted(modes):
        if mode == "ok":
            continue
        cand = [r for r in sorted(pv, key=lambda r: (num(r["JF"]), r["video"], num(r["exp_id"]))) if key(r) in fa and fa[key(r)]["failure_mode"] == mode]
        for r in cand[: a.n]:
            tags[key(r)].add("example_" + mode)

    os.makedirs(a.out_dir, exist_ok=True)
    byk = {key(r): r for r in pv}
    cols = ["tags", "video", "exp_id", "expression", "J", "F", "JF", "failure_mode", "n_drift_events", "drift_onto_obj",
            "n_reappearance_events", "recovery_rate", "n_false_absence_frames", "events"]
    out_rows = []
    for k in sorted(tags, key=lambda k: (k[0], num(k[1]))):
        r = byk.get(k)
        if r is None:
            continue
        f = fa.get(k, {})
        ev = "; ".join(f"{e['kind']}:{e['start_frame']}-{e['end_frame']}:rec={e.get('recovered','')}" for e in by_rec.get(k, []))
        out_rows.append({"tags": ";".join(sorted(tags[k])), "video": k[0], "exp_id": k[1], "expression": r["expression"],
                         "J": r["J"], "F": r["F"], "JF": r["JF"], "failure_mode": f.get("failure_mode", ""),
                         "n_drift_events": f.get("n_drift_events", ""), "drift_onto_obj": f.get("drift_onto_obj", ""),
                         "n_reappearance_events": f.get("n_reappearance_events", ""), "recovery_rate": f.get("recovery_rate", ""),
                         "n_false_absence_frames": f.get("n_false_absence_frames", ""), "events": ev})
    with open(os.path.join(a.out_dir, "selected_cases.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)

    kinds = Counter(r["kind"] for r in rec)
    reapp = [r for r in rec if r["kind"] != "loss_episode"]
    lines = [f"sentences: {len(pv)}", f"selected: {len(out_rows)} (tags: best/worst {a.n} each, disappearance, failed_recovery, example_<mode> {a.n} each)",
             "failure_mode counts (all sentences): " + ", ".join(f"{m}={c}" for m, c in sorted(modes.items())),
             "distractor confusion (drifted onto another object): " + str(sum(1 for r in pv if fa.get(key(r), {}).get("drift_onto_obj", "").strip())),
             "sentences with identity drift events: " + str(sum(1 for r in pv if num(fa.get(key(r), {}).get("n_drift_events", 0)) > 0)),
             "recovery_events kinds: " + ", ".join(f"{k}={c}" for k, c in sorted(kinds.items())),
             f"disappearance events (non-loss_episode rows): {len(reapp)}; recovered==False: {sum(1 for r in reapp if str(r.get('recovered','')).strip()=='False')}",
             "false absence frames (all sentences): " + str(int(sum(num(fa.get(key(r), {}).get('n_false_absence_frames', 0)) for r in pv)))]
    with open(os.path.join(a.out_dir, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print()
    for r in out_rows:
        print(f"{r['tags']:44s} {r['video']:11s} {r['exp_id']:>3} JF={r['JF']:>6} {r['failure_mode']:18s} {r['expression'][:50]}")


if __name__ == "__main__":
    main()
