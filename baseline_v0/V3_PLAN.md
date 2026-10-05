# V3 plan: event-driven VLM verification (written and committed BEFORE any V3 run)

V3 = V2 (memory + Tracklet Coherence Score) + a VLM that is called only at important events.
Code: `events.py`, `vlm.py`, `stage_events.py`. Runner: `tools/run_v3.sh`. No re-identification (that is V4).

## What the VLM does and does not do

It never produces pixels and never searches the frame for the target. At an event it answers ONE small question
about crops the tracker already produced:

| situation | question | verdicts |
|---|---|---|
| target visible (mask non-empty) | identity: is the crop the same object as the frame-0 crop? | match / no_match / unsure |
| target absent (mask empty) | presence: is the described object visible in the frame? | match (visible) / not_visible / unsure |

## Events (computed from the V2 log only, no extra cost)

`reappear`, `disappear`, `coherence_drop` (c fell by 0.30 or entered the LOW band), `distractor` (crop closer to a
cached distractor than to the identity prototype), `drift_suspect` (LOW for 5 visible frames in a row).
Pacing: a refractory window of 10 frames after each call, per target. Keep-alive is OFF and there is no per-target
call cap in the primary run. Band thresholds are the FROZEN V2 values (`config_v2_final.yaml`), not re-tuned.

## Action

Primary: **abstain**. An identity verdict `no_match` (confidence >= 0.6) blanks the current output mask and the following
ones until a `match` verdict arrives, TCS has been HIGH for 5 consecutive frames, or 40 frames have passed.
A presence verdict `match` on an empty mask is logged as a false absence (a V4 trigger) and changes nothing.
Control: **log_only** (same calls, no mask change) isolates how accurate the VLM is from what abstention costs.

## Replay, not a live loop (stated limit)

The VLM does not feed back into SAM 2, memory writes or TCS here, so the V2 run already contains every input the
trigger needs. V3 is therefore computed as an offline replay of the V2 log and masks. Frames up to each event are
processed causally (a decision at frame f uses nothing after f). What this does NOT test: letting a verdict change
which frames enter memory. That needs a live SAM 2 rerun and is left for later.

## Fixed in advance (not tuned on the validation split)

Every number above: 0.30 / 3 / 5 / 0.0 / 10 / off / unlimited, conf_min 0.6, recover_high 5, max_abstain 40.
Reported on the full 30-video split (61 expressions). If any of these is ever tuned it is on the ablation tier only.

## Controls and what is reported

* **Oracle** (`vlm: oracle`): verdicts computed from ground truth. A CEILING for this action and trigger, not a result
  and never described as one. It separates "is the trigger and action useful" from "is the VLM good enough".
* **Periodic, matched budget**: the same number of calls, evenly spaced, same VLM and action. This is the test of
  "event-driven beats a fixed schedule at the same cost" (H3).
* **Every frame**: one call per tracked frame, the cost upper bound (V5).
* For every run: J, F, J&F; VLM calls, calls per tracked frame and per video frame; verdict accuracy against ground
  truth (an identity verdict is correct if it agrees with IoU >= 0.5); sequential VLM latency and estimated inference
  time (V2 time + latency); events fired and suppressed by pacing; frames blanked; every call listed in `vlm_log.csv`.

## Limits stated before the results

* Ref-DAVIS17 has almost no drift or occlusion, so few events will fire; expect few calls and a small J&F change.
  H3 is decided on Long-RVOS / MeViS, not here.
* TCS cannot see a grounding failure (the anchor itself is wrong), so no event fires there and the VLM is never asked.
* Abstention can only turn a wrong mask into an empty one. It helps when the target is genuinely gone or the tracker is
  on the wrong object, and hurts when the VLM wrongly says no_match on a good frame. It cannot recover the target (V4).
* Real-VLM latency in a live system is sequential per target. The replay runs calls in parallel for speed; the summary
  reports both the parallel wall time and the sequential sum.

## Amendment 1 (written AFTER the first oracle runs; post-hoc, labelled as such)

What the pre-registered runs showed (oracle verdicts, V2 passive as source, 61 expressions): the abstain action lowered
J&F even with perfect verdicts (events 48.64, matched periodic 50.72, every frame 46.29, log_only 50.97 = V2).
Cause found: the oracle answered `no_match` whenever IoU < 0.5. 1,515 of 3,923 tracked frames (38.6%) fall below 0.5,
mostly sloppy masks on the right object, not wrong-object tracking. A real VLM asked "is this the same object?" would not
say no_match for those, so the 0.5 oracle is a pessimistic stand-in and abstaining on it throws away partial credit.
Event calls also hit such frames no more often than random (41% no_match for events vs 39% for every-frame).

Added (not a replacement): an **identity oracle**, `V3_ORACLE_IOU=0.10 bash tools/run_v3.sh oracle ...`, which says
`no_match` only when IoU on the target < 0.10 (the project's own "lost / wrong object" definition). The 0.5 results stay
the primary pre-registered ones and are reported first. Every identity-oracle number is labelled post-hoc, with the
threshold chosen after seeing the 0.5 results. The periodic control's call budget was also fixed after the first run
(it matched 34 calls instead of 76 because short tracks never reached the period); both periodic runs are reported.
