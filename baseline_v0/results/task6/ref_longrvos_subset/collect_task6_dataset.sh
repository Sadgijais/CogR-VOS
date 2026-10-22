#!/usr/bin/env bash
# Task 6: collect the SMALL summary files of one dataset run (MeViS or Long-RVOS) into the repo. Only copies files and builds tables
# from them; changes nothing else. No masks, no frames, no model outputs, no API key.
#
#   DS=mevis    bash tools/collect_task6_dataset.sh
#   DS=longrvos bash tools/collect_task6_dataset.sh
set -u
SELF="$(readlink -f "$0")"
DS="${DS:?set DS=mevis or DS=longrvos}"
RUN="${RUN:-$HOME/cogrvos/task6_${DS}/baseline_v0}"
REPO="${REPO:-$HOME/cogrvos/cogrvos_baseline_v0/baseline_v0}"
case "$DS" in
  mevis)    OUTNAME=ref_mevis_subset;    MAN=mevis_valid_u_subset.json;  VLM=mevis_vlm_subset.json;    DATA="$HOME/data/MeViS-subset"
            TITLE="MeViS valid_u subset"; LAB="MeViS valid_u subset (12 videos, single-target sentences) - not the full benchmark"
            X3=339.1; X4=494.6 ;;
  longrvos) OUTNAME=ref_longrvos_subset; MAN=longrvos_valid_subset.json; VLM=longrvos_vlm_subset.json; DATA="$HOME/data/LongRVOS-subset"
            TITLE="Long-RVOS valid subset"; LAB="Long-RVOS valid subset (11 videos, one sentence per type) - not the full benchmark"
            X3=221.1; X4=393.7 ;;
  *) echo "DS must be mevis or longrvos"; exit 1 ;;
esac
OUT="$REPO/results/task6/$OUTNAME"
cd "$RUN" || exit 1
mkdir -p "$OUT/tables" "$OUT/configs" "$OUT/failure_analysis/figures"

# 1) result and recovery folders: csv/txt/json files under 100 KB (no logs of frames, no mask folders, no frames)
for d in results results_v1_final results_v2_passive results_v4_memory results_v4_vlm results_v3_gemini_events_abstain results_v3_oracle_events_log_only \
         recovery_v0 recovery_v1 recovery_v2_passive recovery_v4_memory \
         recovery_sub_v0 recovery_sub_v1 recovery_sub_v2_passive recovery_sub_v4_memory recovery_sub_v3_gemini recovery_sub_v4_vlm; do
  [ -d "$d" ] || { echo "missing folder: $d"; continue; }
  mkdir -p "$OUT/$d"
  find "$d" -maxdepth 1 -type f \( -name '*.csv' -o -name '*.txt' -o -name '*.json' \) -size -100k -exec cp -p {} "$OUT/$d/" \;
done

# 2) failure analysis: per-sentence tables, picker output, case pictures
cp -p failure_analysis*.csv "$OUT/failure_analysis/" 2>/dev/null || echo "no failure_analysis*.csv"
cp -p failure_cases_v2_passive/selected_cases.csv failure_cases_v2_passive/summary.txt "$OUT/failure_analysis/" 2>/dev/null || echo "no picker output"
cp -p failure_cases_v2_passive/figures/*.png "$OUT/failure_analysis/figures/" 2>/dev/null || echo "no pictures"

# 3) configs and run scripts that were used (copied if present), the conversion report and the chain log
for f in config.yaml config_v1_final.yaml config_v2_passive.yaml config_v3_oracle_events_log_only.yaml config_v3_gemini_events_abstain.yaml \
         config_v4_memory.yaml config_v4_vlm.yaml run_ladder.sh run_v4_memory.sh; do
  if [ -f "$f" ]; then cp -p "$f" "$OUT/configs/"; else echo "not found (fine if it never existed): $f"; fi
done
[ -f "$DATA/conversion_report.json" ] && cp -p "$DATA/conversion_report.json" "$OUT/conversion_report.json"
[ -f "$HOME/cogrvos/full_chain.log" ] && cp -p "$HOME/cogrvos/full_chain.log" "$OUT/full_chain_log_both_datasets.txt"

# 4) tables, made with the committed tools/subset_table.py
S=$(grep total_search_s results_v4_memory/reid_summary.txt 2>/dev/null | awk '{print $2}'); S="${S:-0}"
python -I "$REPO/tools/subset_table.py" \
  --variant V0=results --variant V1=results_v1_final --variant V2passive=results_v2_passive \
  --variant V4memory=results_v4_memory:results_v2_passive --extra V4memory="$S" | tee "$OUT/tables/table_full_subset_all_videos.txt"
python -I "$REPO/tools/subset_table.py" --subset "$REPO/subsets/$VLM" \
  --variant V0=results --variant V1=results_v1_final --variant V2passive=results_v2_passive \
  --variant V4memory=results_v4_memory:results_v2_passive \
  --variant V3gemini=results_v3_gemini_events_abstain:results_v2_passive \
  --variant V4vlm=results_v4_vlm:results_v2_passive \
  --extra V3gemini="$X3" --extra V4vlm="$X4" | tee "$OUT/tables/table_vlm_subset.txt"
sed -n 1,22p results_v3_oracle_events_log_only/v3_summary.txt > "$OUT/tables/v3_PLANNED_calls_oracle_replay_projected.txt" 2>/dev/null

# 5) README (facts from the committed lists are printed by python, not typed)
{
cat <<RDME
# Task 6 - $TITLE: results, failure analysis, pictures

Label (use exactly): "$LAB"
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
RDME
case "$DS" in
mevis) cat <<RDME
- MeViS sentences describe motion; the frame-0 box (Grounding-DINO) cannot see motion, so most failures are grounding failures.
- V3 real Gemini: 40 answered calls, 3 of them from an earlier attempt stopped after HTTP 503 errors; total failed attempts cannot be counted.
RDME
;;
longrvos) cat <<RDME
- Long-RVOS frames come every 5 frames; absence means an empty mask. Target-target overlaps lost during conversion are listed in conversion_report.json.
- V3 real Gemini: 28 calls, 6 answered "unsure" by the model itself (crop too small / unclear), none were errors.
- V4-vlm used the a-priori guard of at most 10 searches per sentence (V4_PLAN.md): 30 calls ran, 44 further searches were blocked by it.
RDME
;;
esac
echo
echo "## Facts from the committed lists (printed by python)"
python -I - "$REPO/subsets/$MAN" "$REPO/subsets/$VLM" <<'PY'
import json, sys
for p in sys.argv[1:]:
    d = json.load(open(p))
    print("- " + p.split("/")[-1] + ": " + ", ".join(f"{k}={d[k]}" for k in d if k in
          ("K", "n_videos", "n_sentences", "tracked_frames", "budget_tracked_frames", "seed_text", "max_calls", "planned_calls_on_subset",
           "sentences_on_subset", "content_sha256", "videos", "not_run_in_chosen_videos", "sentence_types")))
PY
} > "$OUT/README.md"
cp -p "$SELF" "$OUT/collect_task6_dataset.sh"
echo; du -sh "$OUT"; echo "files:"; find "$OUT" -type f | wc -l
