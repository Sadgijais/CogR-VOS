# V4 plan: memory-grounded re-identification (written and committed BEFORE any V4 run)

Date written: 2026-10-06. Branch: `v4-reid`, started from `main` at 4c3a9ef (PR #3 merged).
V4 = V3 (memory + coherence score + event-driven VLM) + a live loop that, after a loss event, finds the target
again among the objects visible now and restarts SAM 2 on it.
Planned code (names may change; any change is added under "Amendments" below, never silently):
`stage_candidates.py`, `reid.py`, `stage_reid.py`, new functions in `vlm.py` (re-ID question and its parser),
`tools/recovery_metrics.py`, `tools/make_ablation_table.py`, `tools/run_v4.sh`.

## One-screen summary

| Question | Answer fixed in advance |
|---|---|
| Data | Ref-DAVIS17 val, 30 videos, 61 expressions, 480p, annotator 0 (same as V0 to V3). Long-RVOS is a later, separate plan |
| What triggers a search? | The V3 events (frozen params) plus, while the tracker's mask is empty, one search every 10 frames |
| Where do candidates come from? | Grounding-DINO (frozen, same settings as stage 1), top 5 boxes on the search frame, turned into masks by SAM 2 |
| Who chooses? | Four variants: off, oracle (ceiling), memory-only, VLM |
| What does a restart do? | Gives SAM 2 the chosen mask at frame t as a new prompt for the same object and propagates forward from t. Frames before t never change |
| Does V4 blank masks? | No. An answer of "none" leaves the mask as it is (an empty mask stays empty). This isolates recovery from abstention, because V3 abstention lowered J&F |
| Which expressions are re-run live? | Only those that trigger at least one search; the others keep the V2 passive masks exactly |
| Which numbers are results? | V4-off (safety test), V4-memory, V4-VLM. V4-oracle is a ceiling, never a result |

## What V4 does and does not do

It does: at a search frame it builds a small list of candidates (the tracker's own current mask, if any, plus
Grounding-DINO candidates), ranks them with the target memory, lets one of four choosers pick, and if the pick is a
different object than the tracker's current one, restarts SAM 2 on it.

It does not: change frame 0 grounding (a wrong frame-0 box, 16 of 61 expressions, is invisible to the coherence score
and is not repaired); search on frames with no trigger; use the VLM to draw pixels; tune any parameter on the full split.

## When it searches (triggers)

1. Any V3 event: `reappear`, `disappear`, `coherence_drop`, `distractor`, `drift_suspect`, with the frozen V2 thresholds
   (`config_v2_final.yaml`) and the frozen event parameters used in V3 (`config_v3_*.yaml`).
2. While the tracker's mask is empty: one search every 10 frames.
3. Pause: after any search, no new search for the same expression for 10 frames (as in V3).
4. VLM budget guards (only for the VLM variant, a priori, not tuned): at most 5 searches per absent episode and at most
   10 per expression. If a guard is hit it is counted and reported.

## Candidates

- Grounding-DINO (frozen, same settings as stage 1) on the search frame with the expression: top 5 boxes.
- Each box becomes a mask with SAM 2 (box prompt), on the search frame only.
- The tracker's own current mask (when non-empty) is always candidate "current". A DINO candidate whose mask overlaps
  "current" with IoU above 0.7 is dropped as a duplicate.
- Candidate boxes are cached on disk, so every rerun and every variant reuses them. They are computed only for
  expressions that trigger (see "Fast path").

## Ranking with memory (a shortlist of at most 3)

score = geometric mean of three terms in [0, 1], equal weights, untuned (same style as the V2 coherence score):
1. appearance: DINOv2 cosine similarity of the candidate's masked crop to the target memory prototype (mapped to [0, 1]);
2. position: closeness of the candidate to the last known box of the target (relative to the image diagonal);
3. size: ratio of the candidate's area to the last known area (smaller over larger).
The three best candidates form the shortlist. If the memory has no usable entry, the appearance term uses the frame-0 anchor.

## Who chooses (the four variants)

| Variant | Chooser | VLM calls | Role |
|---|---|---|---|
| V4-off | Nobody (searches are logged, nothing is restarted) | 0 | Safety test: masks must equal V2 passive exactly |
| V4-oracle | Ground truth: the candidate with the highest IoU to the true target mask if that IoU is at least 0.5, else "none" | 0 | Ceiling for these candidates. Never described as a result |
| V4-memory | Top-ranked candidate if its score is at least 0.6, else "none" (0.6 is an a priori value, not tuned) | 0 | Is the VLM needed at all? |
| V4-VLM | The VLM, shown the start crop, the best memory crop and the shortlist labelled A, B, C, plus the expression | counted | The V4 result |

VLM answer format: `{"choice": "A" | "B" | "C" | "none", "confidence": 0 to 1, "reason": "..."}`. A choice with confidence
below 0.6, an unparseable answer, an error, or "unsure" all count as "none". Same provider and model as V3
(`gemini-3.1-flash-lite`), temperature 0, answers cached on disk.

## Restart

- Choice is a candidate other than "current": the chosen mask is given to SAM 2 at the search frame t as a new prompt
  for the same object. Propagation then continues from t. Masks for frames before t are never changed (causal).
- Choice is "current" or "none": nothing changes.
- The restart frame enters target memory only if it passes the normal V1 write gate (no forced writes).

## Fast path and its safety test

Until its first search, V4 behaves exactly like V2 passive. So:
1. Replay the triggers offline from the saved V2 passive logs (CPU, free). Expressions with no trigger keep their V2
   passive masks and need no new computation.
2. Expressions with at least one trigger are re-run live for their whole video. Frames before the first search keep the
   V2 passive masks; from the first search on, the live masks are used. Masks of non-triggering expressions in the same
   video are never replaced.
3. Safety test, run before any full run: V4-off on `libby` and on one video with events must reproduce the V2 passive
   masks byte for byte. If it does not, the fast path is dropped and every video is re-run live, and this is reported.
4. Reported inference time for V4 = V2 passive time + the measured extra time of the live reruns and the candidate
   cache. The way it was measured is written next to the number.

## Metrics (definitions fixed here)

- J, F, J&F: `stage3_evaluate.py`, 480p, first and last frame of each object excluded.
- Reappearance event: the true target is absent for at least 2 consecutive frames, then visible again.
- Recovery Rate: share of reappearance events where the prediction reaches IoU 0.5 or more within 10 frames of the
  target's first visible frame. Counts are always printed next to the rate.
- Recovery Latency: frames from the target's first visible frame to the first frame with IoU 0.5 or more
  (0 = instant). Mean and median over recovered events, plus the number that never recovered.
- Decision latency: seconds of VLM waiting between a search and its restart (VLM variant only).
- Loss episode: 5 or more consecutive visible frames with IoU below 0.10. Recovered if IoU 0.5 or more returns later.
- Every search attempt gets exactly one label: correct restart, wrong restart (restart on IoU below 0.10), missed (a good
  candidate existed, answer "none"), correct none, candidate miss (no candidate with IoU 0.5 or more).
- VLM calls, calls per tracked frame, failure modes (`stage4_failure_analysis.py`), NRE (empty mask while visible).
- The same recovery tool is run on V0, V1, V2 and V3 predictions, so every row of the ablation table has the same columns.

## Fixed in advance (not tuned on the validation split)

Top-5 DINO boxes; shortlist 3; duplicate IoU 0.7; search pause 10 frames; empty-mask search every 10 frames;
VLM confidence minimum 0.6; memory-only score minimum 0.6; guards 5 per absent episode and 10 per expression;
oracle IoU 0.5; recovery window 10 frames; loss episode 5 frames at IoU below 0.10. All thresholds come from
`config_v2_final.yaml` and the V3 event settings. Anything tuned later is tuned on the 15-video ablation tier only
and labelled post-hoc.

## Limits stated before the results

- Ref-DAVIS17 has 4 of 61 targets that vanish and 3 that reappear. Recovery rates on 3 events are anecdotes, not statistics.
  H4 is decided on Long-RVOS, which is not part of this plan.
- V3 found that the VLM misses many lost frames (recall 8 of 21), and a likely (unchecked) reason is a wrong frame-0
  reference crop when grounding failed. The V4 question adds a memory crop and the expression, but it cannot fix a wrong
  anchor.
- One run of one VLM model, no repeats, no significance test. Differences of a few tenths of J&F are "no clear change".
- The oracle uses ground truth and is a ceiling only.
- Free-tier request limits can force the VLM variant to run over several days; cached answers make this safe.
- Candidates come from Grounding-DINO. If it does not propose the target, no chooser can recover it ("candidate miss").

## Amendments

(none yet. Anything changed after the first run is added here with its date and the reason.)
