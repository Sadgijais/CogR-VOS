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
