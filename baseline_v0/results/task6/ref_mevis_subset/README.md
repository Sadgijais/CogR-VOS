# Task 6 - MeViS valid_u subset: results, failure analysis, pictures

Label (use exactly): "MeViS valid_u subset (12 videos, single-target sentences) - not the full benchmark"
No predicted masks or frames are stored here, only small summary files. All runs are single runs.

## Contents
- tables/table_full_subset_all_videos.txt: V0, V1, V2 passive and V4 memory on ALL videos of the subset (V4 time = V2 time + search seconds).
- tables/table_vlm_subset.txt: all six versions on the same VLM-subset videos, with ACTUAL VLM calls (V3 and V4-vlm were run only there).
- tables/v3_PLANNED_calls_oracle_replay_projected.txt: V3 call counts for the whole subset, PROJECTED from the free oracle replay
  (estimated, not executed; the oracle is a ceiling, never a result).
- results_*/: per-video results, summaries, timing, memory / event logs. recovery_*/: whole subset; recovery_sub_*/: VLM-subset videos only.
- failure_analysis/: per-sentence failure types, picker output (selected_cases.csv, summary.txt) and case pictures
  (green outline = ground truth, red = prediction; tile label = frame file and IoU).
- configs/: configs and run scripts used. Code was frozen; only dataset root and tier differ from Ref-DAVIS17.
- conversion_report.json, full_chain_log_both_datasets.txt: dataset conversion facts and the run log (times are UTC).

## Cautions
- per_video_results.csv in results_v3_gemini_events_abstain/ and results_v4_vlm/ mix the VLM-subset videos of that version with copied V2
  masks of the other videos, and the "mean J&F" printed by stage3_evaluate for those folders is therefore meaningless. Use tables/table_vlm_subset.txt.
- The printed "estimated_v3_inference_s" adds the VLM latency of the subset to the V2 time of ALL videos: ignore it. The tables add the
  latency to the V2 time of the subset videos only. V4-vlm time = V2 time + search seconds + Gemini waiting seconds.
- The VLM rows are a small subset because of the Gemini API limit (free tier, 503 overload errors seen). A single-video or two-video row is a
  pilot, not an evaluation of the whole dataset; no overall claim is made from it.
- "loss episodes recovered" are counted from each version's own masks, so the denominators differ between versions (restarts and blanked frames
  create or split episodes). Do not read the fractions as an improvement.
- Failure labels follow the fixed order grounding_failure, identity_drift, track_lost_no_distractor, poor_mask_quality, ok.
- MeViS sentences describe motion; the frame-0 box (Grounding-DINO) cannot see motion, so most failures are grounding failures.
- V3 real Gemini: 40 answered calls, 3 of them from an earlier attempt stopped after HTTP 503 errors; total failed attempts cannot be counted.

## Facts from the committed lists (printed by python)
- mevis_valid_u_subset.json: seed_text=cogrvos-task6-mevis-v1, budget_tracked_frames=12000, videos=['16ae95660b3a', '329909730c21', '3ab5432a1927', '3dde46eaaf53', '410dae675d9a', '4fddcf8823f0', '7ca2e40c5af3', '945c47a50243', 'b4d6cf8c922c', 'd56a6ec78cfa', 'e339e02b5b3c', 'ee6855dcefee'], n_videos=12, n_sentences=213, tracked_frames=10744, not_run_in_chosen_videos={'no_target': 8, 'multi_target': 48}, content_sha256=bb3c31f0f35640e7dc425b7bfbb2d7067cac0d2751847d421fb083f2732f46d9
- mevis_vlm_subset.json: seed_text=cogrvos-task6-mevis-v1, max_calls=40, K=1, planned_calls_on_subset=40, sentences_on_subset=21, videos=['7ca2e40c5af3']
