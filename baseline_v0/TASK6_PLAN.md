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

## AMENDMENT 6 (2026-11-02): candidate-selection experiment with A0, A1 and A3 (no VLM). Written BEFORE any candidate-selection run and BEFORE any Gemini call for it

1. Approval. Arpit Sir approved the revised candidate-selection protocol in his WhatsApp message of 2 Nov 10:58 am: proceed with A0, A1
   and A3, do not include a VLM arm now, update the plan amendment and start the runs; his seven conditions are covered in items 3 to 12 below. The
   protocol text is the Claude Doc "CogR-VOS Candidate-Selection Experiment: Protocol, Revised after Sir's Review" (revision 27 of 27 Oct);
   this amendment repeats everything that is frozen so the repository does not depend on that Doc.
2. Question. Can the frame-0 box be chosen better than top-1, using only the boxes Grounding-DINO already gave and evidence a real system
   could collect, and does a better box give better V0 segmentation? Nothing is run yet; the audit numbers behind it (top-1 correct in
   37.7% and 27.6% of the visible group, correct box in the cached top-5 in 82.7% and 86.2%) come from subsets and single runs.
3. Data, pool and tracker, identical for every arm (Sir's condition 1 and 2). Data: MeViS valid_u subset (12 videos, 213 single-target
   sentences) and Long-RVOS valid subset (11 videos, 33 sentences), always labelled as subsets. Pool: the cached top-5 Grounding-DINO boxes
   of frame 0 (grounding/<video>.json), no new detector run; the list is sorted by detector score in all 246 sentences, so top-1 = best
   detector-score box = A0 (counted once). Tracker after the choice: V0 exactly (SAM 2.1 Hiera-tiny, the chosen box as the prompt at
   frame 0, no memory, no trust score, no VLM, no re-ID). Scoring: the same stage 3 code (J, F, J&F). Every other setting is identical
   between arms; only the way the box is chosen differs.
4. Arms (fixed now).
   A0: top-1 (highest detector score).
   A1: crop each pool box from frame 0 (the box plus 8 pixels of padding, no masking), CLIP ViT-B/32 image embedding (openai/clip-vit-base-patch32,
       frozen, as in common/embed.py), cosine with the CLIP text embedding of the sentence text exactly as in the data (no prompt template);
       the highest cosine wins; a tie goes to the higher detector score.
   A3 (multi-frame, no VLM): for each pool box, start SAM 2.1 from that box at frame 0 and follow it for the first N frames (frame 0 plus
       N - 1 more, counted in the dataset's own annotated frame list; on Long-RVOS one listed frame is 5 video frames). On each frame take
       the masked crop (8 pixels of padding, grey background outside the mask, as in common/embed.py), its CLIP image embedding and the
       cosine with the sentence embedding. A frame whose mask is empty or under 200 pixels scores 0. A candidate's score is the mean over its
       N frames; the highest wins; a tie goes to the higher detector score. The N-frame tracks are thrown away; V0 is run again from frame 0
       over the whole video with the chosen box, and all J, F and J&F values come only from this run.
   N = 5 and N = 10 are two pre-declared arms (A3-5 and A3-10). Both are reported with equal status; neither is chosen after the results
   (Sir's condition 3).
   Frame-0-only sanity rows (no tracking run, seconds on the CPU, not arms): largest box, smallest box, and the expected accuracy of a
   random pick from the pool.
5. Groups and denominators (MeViS / Long-RVOS). All sentences 213 / 33 (J, F, J&F). Target visible in frame 0: 191 / 29 (choice accuracy,
   mean IoU of the chosen box). Target absent in frame 0: 22 / 4 (own group; choice accuracy not defined; they are not recovered here).
   Original top-1 correct / partial overlap / another annotated object / elsewhere: 72 / 26 / 80 / 13 and 8 / 8 / 6 / 7. Target in the
   cached pool: 158 / 25. Choice accuracy = share of the visible group whose chosen box has IoU >= 0.5 with the ground-truth box.
6. Oracle references, diagnostic only and never results of any arm. Normal start (= A0): 26.88 / 30.38. Ground-truth-box start for the
   visible group (the absent group keeps its normal start): 64.29 / 61.15. Best-cached-candidate start (IoU >= 0.5, otherwise top-1): 52.80 /
   61.22, slightly optimistic because it also swaps in a better box when top-1 already hit (15 sentences in MeViS, 1 in Long-RVOS). The
   denominators are all 213 and all 33 sentences. They are reported in their own lines and tables.
7. Measures, for every arm and both datasets (Sir's condition 4): choice accuracy and mean IoU of the chosen box; J, F and J&F; the same split
   by group of item 5; per-video and per-sentence results (how many sentences got better, stayed the same (|difference| < 1 J&F) or got
   worse); cost (selection time per sentence, total time, extra SAM 2 frames, CLIP passes, frames observed N, selection delay, all
   reported separately from the V0 tracking time). Differences: each arm minus A0, and A3 minus A1, with a 95% interval from 10,000
   resamples of the videos (seed 0; the interval covers the choice of videos only, not run-to-run variation). Limitation stated next to
   every table: single runs on 12 and 11 videos, so one or two videos can dominate a result. If time allows, A0 is repeated once on
   Long-RVOS to show how much two identical runs differ.
8. Sensitivity (Sir's condition 5). For every paired comparison, on Long-RVOS and also on MeViS, the result is reported with and without
   the 3 sentences that have the largest gain (as printed by tools/compare_pair.py), and we say whether the direction and size remain. The
   same is done for the V1-from-the-ground-truth-box comparison (V1gt minus V0gt: MeViS -1.64, Long-RVOS +7.51, where the 3 largest
   sentences give 151.8 of 247.7 points). compare_pair.py has no option for this yet; the option (or a separate script) is committed before
   the numbers are produced.
9. Ground-truth-initialised and end-to-end results are reported separately (Sir's condition 6). Any table that starts from the ground-truth
   box (V0gt, V1gt, later versions) is labelled as a diagnostic ceiling and is never listed in the same ranking as the arms A0, A1, A3.
10. Claim rules, fixed now. A3 is a candidate core contribution only if it beats A0 and A1 on both datasets and on both measures (choice
   accuracy and J&F) in the point estimates, and the 95% intervals are shown; a strong claim is written only if the intervals exclude zero,
   otherwise the result is described, not claimed. With wide intervals we say so. The VLM: no VLM arm and no Gemini call under this
   amendment (Sir's condition 7). After A0, A1 and A3 are complete and reviewed, a separate protocol proposes a pre-declared subset within
   40 calls or a local VLM; a VLM becomes part of the core method only if it measurably beats the best non-VLM arm and the extra cost is
   justified. The same separate protocol covers V3 and V4 from the ground-truth start.
11. Order of work. (a) Frame-0-only sanity rows; (b) A1; (c) A3 selection pass for N = 5 and N = 10 (GPU); (d) a downstream V0 run for each
   arm. To save time, only the sentences whose chosen box differs from A0 are re-run, after a one-time mask-identity check (tools/
   audit_mask_identity.py) that sentences with an unchanged box give identical masks; if the check finds any difference, every sentence
   of that arm is re-run. (e) Paired tables, per-video results, failure frames and pictures.
12. Reproducibility (Sir's closing instruction). Every config, selection score (per sentence and candidate), chosen box, log, per_video_results.csv,
   timing and the commit hash of this amendment are kept in the run folders; nothing is overwritten or deleted. Any change after the first
   run is a dated amendment with its reason, and results already seen are reported as they are.
13. Honest limits. Single runs on small subsets; A3 is an offline look-ahead of N frames, not an online method; targets absent in frame 0
   are not recovered; motion evidence is not used (CLIP sees one frame at a time); all time estimates are estimates, not measurements.

## AMENDMENT 7 (2026-11-02): every arm is re-run on ALL sentences; what was seen before the A1 downstream run. Written BEFORE any downstream V0 run of A1 and BEFORE any A3 run

1. Change to Amendment 6 item 11. The downstream V0 run of every arm covers ALL sentences of the dataset (213 MeViS, 33 Long-RVOS), not only
   the sentences whose chosen box differs from A0, and the one-time mask-identity check is not used. Reason: it is simpler, needs no merging of
   results from two runs, and is conservative (a superset of the planned runs). Cost, an ESTIMATE from the earlier oracle V0 runs: about 1.2 h on
   MeViS and 1.8 h on Long-RVOS per arm. Scoring, metrics, groups, intervals and claim rules of Amendment 6 are unchanged.
2. Disclosure of what was seen before this amendment, no setting was changed because of it. Frame-0 rows (commit 1ec8918) and the choice accuracy of
   A1 (tools committed as 16982b9) were computed before the A1 downstream run. Choice accuracy on the visible group (MeViS 191, Long-RVOS 29):
   A0 37.7% and 27.6%; largest box 38.7% and 41.4%; smallest box 35.1% and 17.2%; random pick (expected) 38.6% and 28.2%; A1 38.7% and 44.8%.
   On Long-RVOS the largest box (12 of 29) is already above A0 (8 of 29). Amendment 6 item 10 is NOT changed (A3 must beat A0 and A1); the
   largest-box row stays a frame-0 reference row and gets no downstream V0 run unless Sir asks for one.
3. Run files, same pattern as the oracle runs: per arm a grounding folder written by the selection tool (A1: grounding_a1_clip), config_<arm>.yaml
   (only the five path lines differ from config.yaml, checked with diff), and run_arm.sh (a copy of run_oracle.sh with the names changed, same GPU
   temperature guard). The V0 code, SAM 2.1 Hiera-tiny and stage 3 are unchanged. No Gemini call.
4. Order: A1 downstream (MeViS, then Long-RVOS); then the A3 selection tool is written and tested on the CPU and run on the GPU for N = 5 and
   N = 10; then the downstream V0 run of each A3 arm; then the paired tables, per-video results, sensitivity and pictures.

## AMENDMENT 8 (2026-11-03): Sir's review of the A0, A1 and A3-5 results; A3 N=10, a largest-box reference arm, failure and cost analysis, consistency view. Written BEFORE any A3 N=10 result and BEFORE any largest-box downstream run. No Gemini call
1. Direction. Arpit Sir's WhatsApp reply of 3 Nov 9:37 am (to the update sent 3 Nov 1:36 am), in summary: report the results honestly; A3 looks
   promising but its results are not consistent, so it is not claimed as a successful method; run A3 N=10 with the same frozen settings and no tuning
   or change after seeing results; run the largest-box baseline downstream on both datasets with the same V0 tracker as an additional reference /
   exploratory experiment and update the protocol; inspect the videos and sentences where A3 helped and where it hurt and find why the CLIP ranking
   was wrong (similar-object confusion or mask quality); analyse the overhead of the A3 selection pass against its accuracy gain; no Gemini calls now,
   the VLM question is decided only after the N=10 and largest-box results are complete; the focus is consistent improvement, not a high gain on a few
   videos; all results, negative ones included, are documented.
2. Seen before this amendment (disclosure). J&F of A0, A1 and A3-5 on both datasets with their paired comparisons (A3 tool committed as a56072f);
   choice accuracy of A0, A1, A3-5 and A3-10 (MeViS 37.7, 38.7, 45.0, 43.5%; Long-RVOS 27.6, 44.8, 31.0, 31.0%); the frame-0 choice accuracy of the
   largest box (38.7% and 41.4%, commit 1ec8918, and the self-check of item 4). NOT seen: any J&F of A3-10 or of the largest-box arm. The A3-10 MeViS
   run started on 3 Nov at 11:23 IST, as already declared in Amendments 6 and 7, before this text was committed.
3. A3 N=10 downstream. Exactly as declared in Amendment 6 items 4 and 11 and in Amendment 7: arm a3_n10, config_a3_n10.yaml = config.yaml with the
   five output-path lines changed, the boxes of grounding_a3_n10 (same selection pass as N=5), every sentence re-run, same V0 tracker. No setting is
   changed after seeing results. Reported with equal status to A3-5.
4. Largest-box reference arm ("largest"). Change to Amendment 7 item 2 (earlier: no downstream run unless Sir asks; Sir asked). Definition: among the
   cached top-5 boxes of frame 0 the box with the largest area (x2 - x1) * (y2 - y1); an exact tie goes to the higher detector score; no image, no
   text, no ground truth. Tool tools/select_largest_box.py (68 lines, md5 b4a9a34825d2edbb5f77084611bbadad) writes <run>/grounding_largest (box moved
   to the front in 111 of 213 MeViS and 22 of 33 Long-RVOS sentences) and selection_largest.csv. Self-check with tools/choice_accuracy.py reproduces
   the frame-0 rows: MeViS 74 of 191 (38.7%), Long-RVOS 12 of 29 (41.4%). Downstream: config_largest.yaml, bash run_arm.sh largest, every sentence,
   both datasets, same V0 tracker. Status: an additional reference / exploratory experiment. It is a trivial size rule, not a candidate method; it is
   reported in its own line and is not one of the candidate arms of the claim rule (Amendment 6 item 10). Where A1 or A3 are compared with it, the
   table says so.
5. Failure analysis (CPU only; no new tracking, no Gemini). For A3-5 and A3-10, with A1 alongside, per sentence and from the saved files only
   (selection_a3.csv frame scores and cosines, the cached pool, the ground-truth masks of frame 0, per_video_results.csv): the sentence text; whether
   the pool contains the target (a box with IoU >= 0.5 with the ground-truth box); the chosen box of A0 and of the arm and whether each is a hit; the
   J&F of each arm. For every wrong pick when the pool does contain the target, one reason from three fixed categories: (a) similar-object confusion =
   the chosen box has IoU >= 0.5 with the ground-truth box of another annotated object of the same video; (b) mask quality = the target candidate
   scored 0 on at least one tracked frame (mask under 200 pixels or empty) and its mean over its non-zero frames would have been the highest; (c)
   other = neither of these (for example the chosen box overlaps no annotated object, or the target candidate simply scored lower). The sentences that
   gained or lost are listed per dataset with their text. Pictures (tools/render_cases.py) of at most three gain and three loss sentences per dataset,
   chosen by a stated rule (largest and smallest J&F of A3-5 minus A0) and labelled as picked after seeing results. The tool and these definitions are
   committed before the numbers are produced.
6. Cost analysis. For every arm and dataset, in one table: selection time in total and per sentence, extra SAM 2 frames, CLIP passes, V0 tracking time
   and ms per frame, the ratio of selection time to V0 tracking time, and next to it the change in J&F and in choice accuracy. Two runs on the same
   videos differ in tracking time (Long-RVOS: A1 6244 s, A3-5 7038 s, 13%), so no time difference below about 15% is interpreted. A3 is an offline
   look-ahead of N frames, not an online method.
7. Consistency view (added after seeing the A0, A1 and A3-5 results; disclosed). Per video, for every arm minus A0 and minus A1: the number of videos
   improved, unchanged (|difference| < 1 J&F) and worse; and a leave-one-video-out range of the paired difference (the difference recomputed with each
   single video dropped in turn, minimum and maximum). tools/compare_pair.py gets a --leave-one-out option, committed before the numbers are produced.
   A gain is called consistent only if the sign of the difference holds in all leave-one-video-out recomputations; otherwise we say which video it
   depends on. The 3-largest-gain-sentence check of Amendment 6 item 8 stays.
8. VLM and Gemini. No VLM arm, no Gemini call and no V2-V4 run from the ground-truth start under this amendment. After A3-10 and the largest-box
   downstream results and the failure analysis are complete and reviewed, a separate protocol decides whether a VLM judge is needed or a better non-
   VLM selection mechanism is developed (Sir's instruction).
9. Claim rules unchanged (Amendment 6 item 10): A3 is not claimed as a successful method; results are described as they are, negative ones included.
   Single runs on 12 and 11 videos; one or two videos can dominate a result.
10. Reproducibility. Configs, selection files, logs, per_video_results.csv and timing of every arm stay in the run folders; nothing is overwritten or
   deleted. The interrupted run (a3_n5 on MeViS, 2 Nov, stopped at 18:44 IST after 6 of 12 videos because the laptop smelled hot) is kept as
   predictions_a3_n5_killed_partial_20261102 and results_a3_n5_killed_partial_20261102, is not part of any result, and the arm was then run again
   completely.
