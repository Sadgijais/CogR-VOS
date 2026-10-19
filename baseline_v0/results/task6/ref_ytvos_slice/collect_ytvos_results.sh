#!/usr/bin/env bash
# Task 6: collect the SMALL summary files of the Ref-YouTube-VOS slice runs into the repo. Only copies files, changes nothing else.
set -u
SELF="$(readlink -f "$0")"
RUN="${RUN:-$HOME/cogrvos/task6_ytvos/baseline_v0}"
REPO="${REPO:-$HOME/cogrvos/cogrvos_baseline_v0/baseline_v0}"
OUT="$REPO/results/task6/ref_ytvos_slice"
cd "$RUN" || exit 1
mkdir -p "$OUT/tables" "$OUT/configs" "$OUT/failure_analysis/figures"

# 1) result and recovery folders: csv/txt/json files under 100 KB (no logs, no mask folders, no frames)
for d in results results_v1_final results_v2_passive results_v4_memory results_v4_vlm results_v3_gemini_events_abstain results_v3_oracle_events_log_only recovery_v0 recovery_v1 recovery_v2_passive recovery_v4_memory recovery_subset_V2 recovery_subset_V3 recovery_subset_V4mem recovery_subset_V4vlm; do
  [ -d "$d" ] || { echo "missing folder: $d"; continue; }
  mkdir -p "$OUT/$d"
  find "$d" -maxdepth 1 -type f \( -name '*.csv' -o -name '*.txt' -o -name '*.json' \) -size -100k -exec cp -p {} "$OUT/$d/" \;
done

# 2) failure analysis: per-sentence tables, picker output, case pictures
cp -p failure_analysis*.csv "$OUT/failure_analysis/"
cp -p failure_cases_v2_passive/selected_cases.csv failure_cases_v2_passive/summary.txt "$OUT/failure_analysis/"
cp -p failure_cases_v2_passive/figures/*.png "$OUT/failure_analysis/figures/"

# 3) configs and run scripts that were used (copied if present)
for f in config.yaml config_v1_final.yaml config_v2_passive.yaml config_v3_oracle_events_log_only.yaml config_v3_gemini_events_abstain.yaml config_v4_memory.yaml config_v4_vlm.yaml run_ytvos_ladder.sh run_ytvos_v4.sh; do
  if [ -f "$f" ]; then cp -p "$f" "$OUT/configs/"; else echo "not found (fine if it never existed): $f"; fi
done

# 4) the two tables, made with the committed tools/subset_table.py
SUBSET="$REPO/subsets/ytvos_vlm_subset.json"
python -I "$REPO/tools/subset_table.py" --subset "$SUBSET" \
  --variant V0=results --variant V1=results_v1_final --variant V2passive=results_v2_passive \
  --variant V4memory=results_v4_memory:results_v2_passive \
  --variant V3gemini=results_v3_gemini_events_abstain:results_v2_passive \
  --variant V4vlm=results_v4_vlm:results_v2_passive \
  --extra V3gemini=243.8 --extra V4vlm=833.5 | tee "$OUT/tables/table_10_video_subset.txt"
python -I "$REPO/tools/subset_table.py" \
  --variant V0=results --variant V1=results_v1_final --variant V2passive=results_v2_passive \
  --variant V4memory=results_v4_memory:results_v2_passive \
  --extra V4memory=11.9 | tee "$OUT/tables/table_50_videos.txt"

# 5) README
cat > "$OUT/README.md" <<'RDME'
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
RDME
cp -p "$SELF" "$OUT/collect_ytvos_results.sh"
echo; du -sh "$OUT"; echo "files:"; find "$OUT" -type f | wc -l
