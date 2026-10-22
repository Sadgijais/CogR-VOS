# Task 6 - Long-RVOS valid subset: results, failure analysis, pictures

Label (use exactly): "Long-RVOS valid subset (11 videos, one sentence per type) - not the full benchmark"
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
- Long-RVOS frames come every 5 frames; absence means an empty mask. Target-target overlaps lost during conversion are listed in conversion_report.json.
- V3 real Gemini: 28 calls, 6 answered "unsure" by the model itself (crop too small / unclear), none were errors.
- V4-vlm used the a-priori guard of at most 10 searches per sentence (V4_PLAN.md): 30 calls ran, 44 further searches were blocked by it.

## Facts from the committed lists (printed by python)
- longrvos_valid_subset.json: seed_text=cogrvos-task6-longrvos-v1, budget_tracked_frames=13500, videos=['092ba1d6df', '1a5dead3e3', '20eaede0fb', '235ff86d37', '24b7b95ca4', '2683deffcb', '39db436191', '8b0e20e373', 'bd4aefda10', 'e6f1a061ea', 'f6f82de62a'], sentence_types={'092ba1d6df': {'5': 'static', '6': 'dynamic', '9': 'hybrid'}, '1a5dead3e3': {'2': 'static', '1': 'dynamic', '4': 'hybrid'}, '20eaede0fb': {'0': 'static', '3': 'dynamic', '7': 'hybrid'}, '235ff86d37': {'4': 'static', '1': 'dynamic', '10': 'hybrid'}, '24b7b95ca4': {'6': 'static', '15': 'dynamic', '22': 'hybrid'}, '2683deffcb': {'0': 'static', '1': 'dynamic', '6': 'hybrid'}, '39db436191': {'4': 'static', '7': 'dynamic', '11': 'hybrid'}, '8b0e20e373': {'0': 'static', '1': 'dynamic', '6': 'hybrid'}, 'bd4aefda10': {'2': 'static', '3': 'dynamic', '8': 'hybrid'}, 'e6f1a061ea': {'12': 'static', '13': 'dynamic', '28': 'hybrid'}, 'f6f82de62a': {'0': 'static', '1': 'dynamic', '4': 'hybrid'}}, n_videos=11, n_sentences=33, tracked_frames=12474, content_sha256=d5afa5a267096eae1b01ac624051da90a464ea20a5be2f16ef302c77db7daf9a
- longrvos_vlm_subset.json: seed_text=cogrvos-task6-longrvos-v1, max_calls=40, K=2, planned_calls_on_subset=28, sentences_on_subset=6, videos=['bd4aefda10', 'f6f82de62a']
