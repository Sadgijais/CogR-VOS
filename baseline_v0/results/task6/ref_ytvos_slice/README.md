# Task 6 - Ref-YouTube-VOS slice: results, failure analysis, pictures

Label (use exactly): "Reserved subset from the training split - not official validation."
50 videos / 176 sentences (subsets/ref_ytvos_train_slice.json). V3 and V4-vlm used a 10-video subset
(subsets/ytvos_vlm_subset.json). No predicted masks or frames are stored here, only small summary files.

## Contents
- tables/: table_10_video_subset.txt (all versions on the same 10 videos, actual VLM calls) and
  table_50_videos.txt (V0, V1, V2 passive, V4 memory on all 50 videos). V3 and V4-vlm were NOT run on all 50 videos.
  V3's 171 calls on 50 videos are PROJECTED from the event schedule, not executed
  (results_v3_oracle_events_log_only/ holds that free replay).
- results_*/: per-video results, summaries, timing, memory / event logs. recovery_*/: reappearance and loss events.
- failure_analysis/: per-sentence failure types, the picker output (selected_cases.csv, summary.txt) and case pictures
  (green outline = ground truth, red = prediction; tile label = frame file and IoU).
- configs/: configs and run scripts that were used. Code was frozen; only dataset root and tier differ from Ref-DAVIS17.

## Cautions
- per_video_results.csv in results_v3_gemini_events_abstain/ and results_v4_vlm/ hold the 10 subset videos from that version
  plus 40 copied V2 videos. Use tables/table_10_video_subset.txt for those two versions.
- One of the 40 V4-vlm calls (video ab199e8dfb, sentence 5, frame 11) failed with HTTP 503 after 3 tries and was treated as "keep".
- F is low on YouTube-VOS because its ground-truth outlines are coarse. Compare versions with each other, never F against DAVIS.
- The good-tracking example 5104aa1fea exp 1 was chosen after seeing results (POST HOC). The fox pair (fffe5f8df6) scores high
  because the target is absent in most frames and both-empty frames count as correct.
- All runs are single runs.
