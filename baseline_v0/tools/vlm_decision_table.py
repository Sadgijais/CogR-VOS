#!/usr/bin/env python3
"""Task 6 / Amendment 9 item 3: decision-wise table of the LIVE VLM calls (V3 judge log and V4-vlm re-ID log).

CPU only. Reads two existing CSV logs. Makes NO Gemini call, imports no API client, writes only the files named by
--out-csv / --out-txt. Counts of calls on the live video(s) only: they are examples, not rates, and not method-level
estimates.

Groups (one per decision, first true rule wins):
  failed             the call errored (V3: reason starts with "error:"; V4: vlm_error is not empty)
  not_called         V3 only: empty crop, no call was made
  invalid            V3: verdict outside {match, no_match, unsure, not_visible}; V4: no error but no answer letter
  unsure             V3 only: the model answered "unsure" (a real answer, not graded by the existing code)
  correct            V3: the log's own "correct" is True; V4: label correct_restart or correct_none
  incorrect          V3: "correct" is False; V4: label wrong_restart, weak_restart or missed
  no_good_candidate  V4 only: label candidate_miss (nothing changed and no candidate was good: not a VLM error)
  ungraded           anything else

True state of the tracker mask on the call frame: absent (target not in the label), good (IoU >= 0.50),
poor (0.10 <= IoU < 0.50), lost (IoU < 0.10), unknown. Thresholds as in reid.py (oracle_iou 0.5, wrong_iou 0.10).
"""
import argparse
import csv
from collections import Counter, defaultdict

GOOD, LOST = 0.50, 0.10
VALID_V3 = {"match", "no_match", "unsure", "not_visible"}
V4_CORRECT = {"correct_restart", "correct_none"}
V4_INCORRECT = {"wrong_restart", "weak_restart", "missed"}
GROUPS = ["correct", "incorrect", "unsure", "no_good_candidate", "invalid", "failed", "not_called", "ungraded"]
STATES = ["good", "poor", "lost", "absent", "unknown"]
FIELDS = ["dataset", "source", "video", "exp_id", "frame", "trigger", "kind", "state", "answer", "confidence",
          "effect", "group", "label", "cached", "latency_s", "iou_current", "iou_chosen"]


def tf(x):
    s = str(x).strip().lower()
    if s in ("true", "1", "1.0"):
        return True
    if s in ("false", "0", "0.0"):
        return False
    return None


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def state(present, iou):
    if present is None:
        return "unknown"
    if present is False:
        return "absent"
    if iou is None:
        return "unknown"
    if iou >= GOOD:
        return "good"
    if iou >= LOST:
        return "poor"
    return "lost"


def v3_rows(path, dataset, blank_conf):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            verdict = (r["verdict"] or "").strip()
            reason = (r.get("reason") or "").strip()
            conf = fl(r["confidence"])
            ok = tf(r["correct"])
            if reason.startswith("error:"):
                group = "failed"
            elif reason == "empty crop":
                group = "not_called"
            elif verdict not in VALID_V3:
                group = "invalid"
            elif verdict == "unsure":
                group = "unsure"
            elif ok is True:
                group = "correct"
            elif ok is False:
                group = "incorrect"
            else:
                group = "ungraded"
            blank = verdict == "no_match" and conf is not None and conf >= blank_conf
            out.append(dict(dataset=dataset, source="V3", video=r["video"], exp_id=r["exp_id"], frame=r["frame"],
                            trigger=r["event"], kind=r["kind"], state=state(tf(r["gt_present"]), fl(r["gt_iou"])),
                            answer=verdict, confidence=r["confidence"], effect="blank_mask" if blank else "none",
                            group=group, label="", cached=r["cached"], latency_s=r["latency_s"],
                            iou_current=r["gt_iou"], iou_chosen=""))
    return out


def v4_rows(path, dataset):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            err = (r.get("vlm_error") or "").strip()
            letter = (r.get("vlm_letter") or "").strip()
            label = (r["label"] or "").strip()
            if err:
                group = "failed"
            elif not letter:
                group = "invalid"
            elif label in V4_CORRECT:
                group = "correct"
            elif label in V4_INCORRECT:
                group = "incorrect"
            elif label == "candidate_miss":
                group = "no_good_candidate"
            else:
                group = "ungraded"
            out.append(dict(dataset=dataset, source="V4vlm", video=r["video"], exp_id=r["exp_id"], frame=r["frame"],
                            trigger=r["trigger"], kind="reid", state=state(tf(r["gt_visible"]), fl(r["current_iou"])),
                            answer=r["choice"], confidence=r["vlm_conf"], effect=r["action"], group=group,
                            label=label, cached=r["vlm_cached"], latency_s=r["vlm_latency_s"],
                            iou_current=r["current_iou"], iou_chosen=r["chosen_iou"]))
    return out


def matrix(rows, key, order, title):
    cnt = Counter((r[key], r["state"]) for r in rows)
    vals = [v for v in order if any(cnt[(v, s)] for s in STATES)]
    vals += sorted({r[key] for r in rows} - set(order))
    lines = [title, "  %-22s" % key + "".join("%8s" % s for s in STATES) + "%8s" % "total"]
    col = Counter()
    for v in vals:
        n = [cnt[(v, s)] for s in STATES]
        for s, x in zip(STATES, n):
            col[s] += x
        lines.append("  %-22s" % (v or "(empty)") + "".join("%8d" % x for x in n) + "%8d" % sum(n))
    lines.append("  %-22s" % "total" + "".join("%8d" % col[s] for s in STATES) + "%8d" % sum(col.values()))
    return lines


def summarize(rows, name, dataset):
    L = ["", "=" * 78, "%s  (%s): %d decisions" % (name, dataset, len(rows))]
    if not rows:
        return L
    L.append("live video(s): " + ", ".join(sorted({r["video"] for r in rows})))
    L.append("")
    L += matrix(rows, "group", GROUPS, "Group by true state of the tracker mask on the call frame")
    if name == "V3":
        L.append("")
        for kind in ("identity", "presence"):
            sub = [r for r in rows if r["kind"] == kind]
            if sub:
                L += matrix(sub, "answer", ["match", "no_match", "unsure", "not_visible"], "V3 %s calls: answer by true state" % kind)
                L.append("")
        blank = [r for r in rows if r["effect"] == "blank_mask"]
        L += matrix(blank, "effect", ["blank_mask"], "V3 answers that would blank the mask (no_match, confidence at or above the blank threshold), by true state")
    else:
        L.append("")
        L += matrix(rows, "label", ["correct_restart", "correct_none", "wrong_restart", "weak_restart", "missed", "candidate_miss"],
                    "V4-vlm label by true state of the current mask")
        L.append("")
        L += matrix(rows, "effect", ["keep", "restart"], "V4-vlm action by true state")
        restarts = [r for r in rows if r["effect"] == "restart"]
        L.append("")
        L.append("  restarts (current IoU -> chosen IoU):")
        for r in restarts:
            L.append("    video %s sentence %s frame %s  label %s  %s -> %s" % (
                r["video"], r["exp_id"], r["frame"], r["label"], r["iou_current"], r["iou_chosen"] or "n/a"))
        if not restarts:
            L.append("    none")
    per = defaultdict(Counter)
    for r in rows:
        per[(r["video"], r["exp_id"])][r["group"]] += 1
    L += ["", "  per sentence (video, sentence): calls | correct incorrect unsure no_good_cand invalid failed other"]
    for (v, e), c in sorted(per.items(), key=lambda kv: (kv[0][0], int(kv[0][1]) if kv[0][1].isdigit() else 0)):
        other = c["not_called"] + c["ungraded"]
        L.append("    %s %3s: %3d | %3d %3d %3d %3d %3d %3d %3d" % (
            v, e, sum(c.values()), c["correct"], c["incorrect"], c["unsure"], c["no_good_candidate"], c["invalid"], c["failed"], other))
    cached = sum(1 for r in rows if tf(r["cached"]) is True)
    live = [r for r in rows if tf(r["cached"]) is False and r["group"] != "failed" and fl(r["latency_s"]) is not None]
    mean = sum(fl(r["latency_s"]) for r in live) / len(live) if live else float("nan")
    L += ["", "  cached answers: %d of %d | uncached answered calls: %d, mean latency %.2f s (cached rows keep their stored latency and are left out)" % (
        cached, len(rows), len(live), mean)]
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="label printed in the output, e.g. 'MeViS valid_u subset'")
    ap.add_argument("--v3-log", required=True)
    ap.add_argument("--v4-log", required=True)
    ap.add_argument("--blank-conf", type=float, default=0.6, help="V3 design: no_match at or above this confidence blanks the mask")
    ap.add_argument("--out-csv")
    ap.add_argument("--out-txt")
    a = ap.parse_args()
    v3 = v3_rows(a.v3_log, a.dataset, a.blank_conf)
    v4 = v4_rows(a.v4_log, a.dataset)
    L = ["Decision-wise table of the live VLM calls. Dataset: %s" % a.dataset,
         "V3 log: %s" % a.v3_log, "V4 log: %s" % a.v4_log,
         "Reading rules: these are the calls made on the live video(s) only. Counts, not rates. One run. No new Gemini call was made.",
         "State thresholds: good IoU >= %.2f, lost IoU < %.2f, poor in between, absent = target not in the label." % (GOOD, LOST),
         "V3 correct = the existing log column; unsure and not-visible-for-identity are not graded by it."]
    L += summarize(v3, "V3", a.dataset)
    L += summarize(v4, "V4vlm", a.dataset)
    text = "\n".join(L)
    print(text)
    if a.out_txt:
        with open(a.out_txt, "w") as f:
            f.write(text + "\n")
    if a.out_csv:
        with open(a.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(v3 + v4)


if __name__ == "__main__":
    main()
