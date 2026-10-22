# TASK 6 PLAN: evaluating CogR-VOS on more datasets

Written 2026-10-15, BEFORE any model has been run on the new data.
Mentor's instruction (Arpit Sir): evaluate the final CogR-VOS system on (1) Ref-YouTube-VOS (primary),
then (2) MeViS and (3) Long-RVOS. Keep the model configuration and the evaluation protocol the same.
This file covers Ref-YouTube-VOS. MeViS and Long-RVOS get their own sections, added and committed
before their runs.

## 1. What is frozen (nothing here is tuned on the new data)

- Models: Grounding-DINO-tiny (frame 0 only, top-5 boxes cached), SAM 2.1 Hiera-tiny (propagation),
  DINOv2-small and CLIP ViT-B/32 (embeddings), Gemini `gemini-3.1-flash-lite` (VLM judge only).
- Settings: the exact values of `config_v2_final.yaml` and the frozen V3 / V4 settings
  (`tools/run_v3.sh`, `tools/run_v4.sh`). Only PATHS are changed (dataset root, output folders).
  STTM parameters, tau_low = 0.81, tau_high = 0.92, V3 event settings, V4 chooser settings: all unchanged.
- Code: the frozen V0-V4 code runs UNCHANGED. New data is converted into the Ref-DAVIS17 folder shape
  by `tools/convert_ytvos_slice.py` (convert the data, never edit the pipeline).
- Same laptop (RTX 2050, 4 GB), one run per version (single run, no seeds to average; stated as a limit).

## 2. Dataset: Ref-YouTube-VOS

Problem: the official `valid` split has NO public ground-truth masks. On this laptop `valid/Annotations`
holds the predictions layout `<video>/<sentence_id>/<frame>.png`. J and F cannot be scored there.
Decision: score on a FIXED SLICE OF `train` (which has masks) and call it
"held-out slice of Ref-YouTube-VOS train, NOT the official valid split". Numbers are therefore not
comparable to the leaderboard. They are comparable to our own Ref-DAVIS17 numbers.

Slice rule (no randomness):
1. candidates = train videos with at least 10 annotated frames (3455 of 3471),
2. order by sha256("cogrvos-task6-ytvos-v1" + video_name),
3. take the first 50, then sort the list.
Result: 50 videos, 176 sentences, 88 target objects, 4740 scored frame-observations.
List: `subsets/ref_ytvos_train_slice.json`; content sha256 = 6a20616e... (full value is inside that file).

Facts about the slice (measured on the ground truth, not on any model output):
8 of 176 sentences have a target that is not visible in frame 0; 6 of 88 objects vanish for 2+ frames
and come back; 0 objects are never visible. So disappearance/reappearance is rare here.
The recovery numbers will rest on very few events and will be reported with the event count.

Caveat to state in every table: SAM 2.1's own training data may include Ref-YouTube-VOS-related
video sources. For SAM 2 (paper) the released model used SA-V + Internal + SA-1B and the optional YouTube-VOS
mix is not part of it; this is NOT verified for SAM 2.1 tiny. So the slice is "held-out" for our system
(nothing was tuned on it) but not guaranteed unseen by the backbone.

Resolution protocol: frames and masks are resized to 480 px height (1280x720 -> 854x480) so that J, F and the
4 px boundary tolerance mean the same as on Ref-DAVIS17. Images bilinear; masks nearest neighbour; object ids
checked not to change. Evaluation conventions identical to Ref-DAVIS17: first and last frame of a video
excluded, both-empty = 1, one-empty = 0.

## 3. Versions that will be run, and how

| Version | How | Gemini calls |
|---|---|---|
| V0 baseline | full slice | 0 |
| V1 STTM | full slice | 0 |
| V2 TCS passive (the shipped V2) | full slice; also writes the log V3 replays | 0 |
| V3 event-driven | offline replay of the V2 passive log. PLANNED calls/frame are counted on the full slice from the event schedule (no API). Real Gemini verdicts only on the VLM subset below | subset only |
| V4-memory | full slice, memory chooser | 0 |
| V4-vlm | VLM subset only | subset only |

Not run: V2 gated (already a reported negative result), V4-memory_margin (it was POST HOC on DAVIS).

VLM subset (why: the free Gemini key allows only about 20 requests per day per model):
the first K videos of the slice in the same hash order, where K is the LARGEST number such that the planned
V3 events schedule needs at most 40 calls on those K videos. K is computed from the V2 passive log with no API
and no look at accuracy. The same K videos are used for V3 and V4-vlm, and V0/V1/V2/V4-memory are also
reported on that same subset so the rows are comparable. Cached verdicts are reused, so reruns cost nothing.
The VLM rows are labelled "subset, K videos" and never mixed into full-slice rows.

## 4. Metrics (reported for every version)

J, F, J&F (mean over sentences), same as Ref-DAVIS17.
Inference time: total seconds per version (V3/V4: computed by the same rule as the ablation table).
FPS = scored frame-observations (sentence x tracked frames) / inference seconds. Scope stated next to the
number: SAM 2 propagation + memory + scoring stages on the GPU, grounding done once per sentence included,
video loading included; GPU model load excluded. For V3 the time is the estimated time from v3_summary.
VLM calls per frame = VLM calls / scored frame-observations (planned count for full slice, real count for subset).
Recovery rate and recovery latency: from `tools/recovery_metrics.py` with the frozen settings
(recovery_iou 0.5, window 10 frames), reported together with the number of events they rest on.
All tables built by scripts from the files the stages wrote, nothing typed by hand.

## 5. Per-video results, visualizations, failure cases

- Per-video and per-sentence CSVs kept for every version under `results/task6/ref_ytvos_slice/<version>/`.
- Qualitative visualizations: a fixed set chosen by rule (the 3 best and 3 worst sentences by V2 J&F,
  plus every sentence with a disappearance), not hand-picked.
- Failure cases: same definitions and thresholds as the Ref-DAVIS17 failure analysis
  (`failure_analysis` block of the config): drift, occlusion, distractor confusion, failed recovery,
  grounding failure. Counts reported per type, with example frames.

## 6. Order of work and stop rules

1. Commit this plan + slice list + the two scripts. (Now.)
2. Pilot: 3 videos (the first 3 in hash order) through V0-V2, only to check paths, time per frame and memory.
   Pilot numbers are NOT reported. Only crashes and path mistakes may be fixed afterwards, each logged in the
   progress note. No setting that affects accuracy is changed after the pilot.
3. Full slice: V0, V1, V2 passive, then V3 replay (planned calls), V4-memory.
4. Compute K, run the real VLM rows on the subset.
5. Tables, failure analysis, visualizations, one progress note.
Stop and report (do not push on) if: a GPU out-of-memory error repeats, GPU temperature stays above 85 C,
or the free Gemini quota runs out (then the VLM rows stay partial and say so).

## 7. Honest limits (to be repeated in the write-up)

- Held-out slice of train, not the official valid split; 50 videos / 88 distinct targets, so differences
  smaller than about 1 J&F point are not meaningful. Single run, no confidence intervals.
- Only 6 vanishing objects: recovery results are anecdotal.
- On Ref-DAVIS17 the ladder gave no accuracy gain (J&F 50.48 -> about 51); the contribution is V3's lower cost.
  We do not expect that to change and will not describe any gain smaller than the noise as an improvement.
- VLM rows use a subset because of the free-tier quota.

## AMENDMENT 1 (2026-10-17): written after V0, V1, V2 and V4-memory were run, BEFORE any real Gemini call

Mentor reply (Arpit Sir, 17 Oct 2026). Rules adopted from now on:
1. Label for the Ref-YouTube-VOS slice in every table and in the report: "Reserved subset from the training split -
   not official validation." No claim of official-validation or independent-test results.
2. Real-VLM versions (V3, V4-vlm) run only on the pre-declared 10-video subset (subsets/ytvos_vlm_subset.json,
   commit 12b5e1b, rule in tools/pick_vlm_subset.py), the same videos for both. V0, V1, V2 and V4-memory are also
   reported on that subset, and on all 50 videos.
3. For every version on the same videos: J, F, J&F and inference time (tools/subset_table.py). For the VLM versions:
   ACTUAL executed VLM calls and calls per tracked frame.
4. The 171 calls counted on all 50 videos come from the event schedule (oracle replay, no API). They are called
   "estimated/projected" and kept apart from actual calls.
5. Frozen: video ids, prompts, thresholds, configs, protocol. Nothing changes after results are seen. The pipeline
   code (V0-V4, prompts in vlm.py) is the repository version at commit 2d50ba3 and was not edited for Task 6; the run
   configs differ only in dataset path and tier name.
Order of work: (a) fair comparison of all versions on the 10-video subset, then (b) full-50-video results and
failure cases, reported separately.

Definitions fixed here (they replace the wording in sections 4 and 5 where different):
- FPS = tracked target frames / stage-2 inference seconds. It excludes the one-off Grounding-DINO stage (175 s)
  and the embeddings cache.
- Calls per frame = VLM calls / tracked target frames (4916 on 50 videos, 976 on the subset), not the 4740 scored.
- V4-memory time = V2 time + measured search time (11.9 s on 50 videos).

Results seen before this amendment (no Gemini involved), J&F on all 50 videos: V0 44.04, V1 43.96, V2 passive
43.96, V4-memory 43.80. V4-memory made 26 restarts: 3 correct, 21 wrong, 2 weak. No setting was changed after seeing them.

## AMENDMENT 2 (2026-10-20): MeViS and Long-RVOS, written BEFORE any MeViS or Long-RVOS data is converted and BEFORE any model is run on it

Mentor's Task 6 order: after Ref-YouTube-VOS (done, Amendment 1), run the whole ladder (V0-V4) on MeViS, then Long-RVOS, with
the same frozen models, configs, code, thresholds and prompts. Both datasets were already on the laptop (no download).
Facts below were read from the meta files only (no model, no frames).

1. Splits and labels (use exactly in every table and in the report):
   - MeViS: "MeViS valid_u subset (official MeViS v2 split with public masks; fixed subset; offline evaluation, not the
     online leaderboard)". valid_u = 50 videos, 907 sentences, frames 25-216 per video (mean 68.8), every frame annotated.
   - Long-RVOS: "Long-RVOS valid subset (official valid split; fixed subset)". valid = 112 videos, 1326 sentences
     (460 static, 427 dynamic, 439 hybrid), 188-675 frames per video (mean 378.9), frames already every 5th.
   Neither is comparable to a leaderboard (subset, training-free single-box method).

2. Which sentences are run:
   - MeViS valid_u has 656 single-target, 213 multi-target (several objects) and 38 no-target (object absent) sentences. The method
     follows ONE frame-0 box, so only the single-target sentences are run. The other 251 are NOT RUN, are counted and reported
     as "not supported by this method", and are never dropped silently.
   - Long-RVOS sentences have one object each, so none are excluded by this rule.

3. Fixed subsets (rules in the two tools, no randomness; lists and content hashes in the manifest files):
   - tools/build_mevis_subset.py -> subsets/mevis_valid_u_subset.json. Videos ordered by sha256("cogrvos-task6-mevis-v1"+video),
     kept while the running total of tracked frames (single-target sentences x (frames-1)) is <= 12000; stop at the first video
     that does not fit.
   - tools/build_longrvos_subset.py -> subsets/longrvos_valid_subset.json. Per video one sentence of each type (static, dynamic,
     hybrid; the hash-smallest of that type); videos lacking a type are skipped; videos ordered by
     sha256("cogrvos-task6-longrvos-v1"+video); kept while the running total of tracked frames (3 x (frames-1)) is <= 13500;
     stop at the first video that does not fit.
   - Budgets are set from time, not from accuracy: at the YouTube-VOS speed (V0+V1+V2+V4-memory about 1.98 s per tracked frame)
     the chosen subsets need about 5.9 h (MeViS) and 6.9 h (Long-RVOS). These are ESTIMATES. MeViS frames are 1080p and Long-RVOS
     videos are long, so the real time can differ; actual times will be reported.
   - Result of the two tools (counted on the meta files only, no model): MeViS subset = 12 videos, 213 sentences, 10744 tracked
     frames (47 of 50 videos had a single-target sentence; in the 12 chosen videos 8 no-target and 48 multi-target sentences are NOT
     RUN). Long-RVOS subset = 11 videos, 33 sentences, 12474 tracked frames, 235-675 frames per video (88 of 112 videos have all 3
     sentence types; the 24 without are skipped by rule, so the subset is drawn from the 88 only).

4. Versions: V0, V1, V2 passive and V4-memory on the whole subset; V3 planned calls counted from the event schedule (no API,
   labelled "estimated/projected"). Real Gemini rows (V3, V4-vlm) only on a smaller VLM subset: the first K videos of the subset in
   the same hash order, K the largest value with at most 40 planned V3 calls (computed from the V2 passive log with
   tools/pick_vlm_subset.py before any API call, then committed). The other versions are also reported on that VLM subset. Actual VLM
   calls and calls per tracked frame are reported separately from projected ones. Same definitions as Amendment 1.

5. Conversion (the pipeline is not edited): frames and masks are converted into the Ref-DAVIS17 480p folder shape, 480 px height
   (bilinear for images, nearest for masks). MeViS masks come from RLE in mask_dict.json; Long-RVOS masks from the per-object PNGs
   (a frame without a PNG is an empty mask). The converters are committed before the pilot. If two objects overlap on a pixel the
   single label PNG can keep only one, so the converter counts overlapping pixels and the count is reported.

6. Pilot: the first 3 videos in hash order of each subset, through V0-V2, only to check paths, time per frame and memory. Pilot
   numbers are NOT reported. Only crashes and path mistakes may be fixed afterwards, each logged. No setting that affects accuracy
   changes after the pilot.

7. Stop rules: as in section 6 of this plan. In addition, if a long Long-RVOS video does not fit in 7.4 GB RAM or the run is much
   slower than estimated, stop and report; any change (smaller subset, video cap) is written as Amendment 3 before the rerun and is
   never decided from accuracy.

8. Honest limits: single runs; fixed subsets, so differences under about 1 J&F are noise; the frame-0 grounding step may miss targets
   that are not visible or not distinct in frame 0 (likely for MeViS motion sentences), which will show up as grounding failures; time
   estimates come from another dataset; whether SAM 2.1 saw these videos in training is not verified.

## AMENDMENT 3 (2026-10-20): pilot outcome, memory finding, stress test and a fallback rule. Written BEFORE any full run on MeViS or Long-RVOS

1. Pilot (Amendment 2, section 6: first 3 videos in hash order of each subset, V0, V1, V2 passive) finished without a crash on MeViS
   and, at the time of writing, V0 and V1 finished on Long-RVOS (V2 still running). No code or setting was changed. The pilot log
   prints recovery counts; they were seen incidentally, are NOT reported, and no setting was changed because of them.
2. Measured speed in the pilot (tracked frames = sum over sentences of (frames-1): 2919 for MeViS, 3252 for Long-RVOS):
   MeViS V0 about 2.40 and V1 about 2.02 frames/s; Long-RVOS V0 about 2.00 and V1 about 1.65 frames/s. Projected GPU time for
   V0 + V1 + V2 + V4-memory on the full subsets (ESTIMATE, V2 and V4-memory assumed about 3 % above V1): about 5.8 h (MeViS) and
   about 8.2 h (Long-RVOS). Actual times will be reported.
3. Memory finding: RAM used reached 11.9 GB (WSL limit 11.7 GiB) at the switch between two Long-RVOS videos of 420 and 432 frames, and
   was about 7.3 GB while one 420-frame video ran. Estimate: about 16 MB per frame, so the longest subset video (675 frames) needs about
   11.4 GB while it runs. WSL memory was raised from 7.4 GB to 12 GB (.wslconfig, laptop has 15.2 GB) and swap to 24 GB. This changes the
   laptop's settings only; the code and the configs are untouched. MeViS needs no change (at most 216 frames).
4. Stress test before the full Long-RVOS run: V0 stage 1 and stage 2 only, on the 675-frame video alone (a one-video tier manifest in a
   separate run folder). Its outputs are thrown away and are not reported; it only shows whether that video fits in memory.
5. Fallback rule, fixed now: if the stress test or the full run is killed for lack of memory although WSL has 12 GB RAM and 24 GB swap,
   the Long-RVOS subset is rebuilt with tools/build_longrvos_subset.py plus a new option --max-frames 500 (videos longer than 500 frames
   are skipped; same hash order and same budget), the new list is committed before the rerun, and every table says "videos up to 500
   frames". No other fallback. Nothing here is ever decided from accuracy.

## AMENDMENT 4 (2026-10-21): minimum VLM subset size. Written BEFORE any Gemini call on either dataset

1. Amendment 2 says the VLM subset is the first K videos in hash order with K as large as possible while the planned V3 calls stay at
   40 or fewer. On Long-RVOS the first video in hash order (39db436191, 3 sentences) already needs 59 planned calls, so the rule gives
   K = 0, an empty subset. The planned calls come from the oracle replay (no API, no accuracy).
2. New rule, for Long-RVOS only: K = 1, the first video in hash order (39db436191, 3 sentences, 59 planned calls). The cap of 40 is
   exceeded and every table and the final report say so. The list is made with tools/pick_vlm_subset.py --max-calls 59, which gives
   K = 1; nothing else in the tool changes.
3. MeViS is unchanged: 1 video (7ca2e40c5af3), 21 sentences, 40 planned calls (committed as b6609a6; label-only fix 0fdfb2a).
4. Nothing here uses accuracy. The V4-vlm runs on the same video(s) as V3 on each dataset, as in Amendment 2.

## AMENDMENT 5 (2026-10-22): VLM subset rule changed to "skip videos that do not fit"; replaces Amendment 4 item 2. Written BEFORE any Long-RVOS Gemini call

1. Reason. Arpit Sir's reply (WhatsApp, 22 Oct 06:35 and 06:49): do not force the 59-call video; check whether suitable videos fit within
   40 calls, several videos are better, otherwise a single-video pilot clearly labelled; report actual VLM calls, J, F, J&F, recovery and
   failure cases; state the API limitation; do not claim overall performance from one video.
2. New rule (both datasets): take the videos in hash order; add a video if the planned V3 events calls on the chosen videos stay <= 40,
   otherwise skip it and continue. Planned calls come from the oracle replay (events_per_expression.csv); accuracy is never used.
   Tool: tools/pick_vlm_subset_skip.py (61 lines, md5 9287815d3d7083ba7c35ac2ea3eda49b).
3. MeViS: no change. 7ca2e40c5af3 (21 sentences, 40 planned calls); every other video would exceed 40. The new tool reproduces the committed
   list exactly. Real Gemini V3 and V4-vlm on this list were run on 22 Oct, before this amendment: V3 = 40 answers (3 of them from an earlier
   attempt that was stopped after HTTP 503 errors), V4-vlm = 60 calls.
4. Long-RVOS: f6f82de62a (28 planned calls) and bd4aefda10 (0 planned calls) = 2 videos, 6 sentences, 28 planned calls. Skipped in hash
   order: 39db436191 (59), 1a5dead3e3 (75), 092ba1d6df (30), 235ff86d37 (39), 20eaede0fb (20), 2683deffcb (112), 24b7b95ca4 (34),
   8b0e20e373 (13, would make 41), e6f1a061ea (32). Amendment 4 item 2 (K = 1 on 39db436191) is withdrawn.
5. Nothing here uses accuracy. V4-vlm runs on the same videos as V3. Every table says "VLM subset, K videos"; the report says that the VLM
   rows are a small subset because of the API limit and that single-video rows are pilots, not a Long-RVOS evaluation.
