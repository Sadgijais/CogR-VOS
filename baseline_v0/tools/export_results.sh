#!/usr/bin/env bash
# Copy the SMALL result files of V2 and V3 into ../results/ (summaries, per-video tables, logs, plots).
# Skips large per-frame logs and predicted masks. Run from baseline_v0/:  bash tools/export_results.sh
set -u
OUT=../results
MAXKB=2048
copy_run () {          # copy_run <src results dir> <dest dir> <failure_analysis csv or ''>
  src="$1"; dst="$2"; fa="${3:-}"
  [ -d "$src" ] || { echo "skip (missing): $src"; return; }
  mkdir -p "$dst"
  for f in summary.csv per_video_results.csv tcs_per_expression.csv sttm_per_expression.csv timing_per_video.csv \
           timing_summary.txt TCS_ANALYSIS.txt tcs_failure_cases.csv tcs_frames.csv v3_summary.txt vlm_log.csv \
           events_per_expression.csv failure_analysis.csv; do
    if [ -f "$src/$f" ]; then
      kb=$(( $(stat -c %s "$src/$f") / 1024 ))
      if [ "$kb" -le "$MAXKB" ]; then cp "$src/$f" "$dst/"; else echo "  too large, skipped: $src/$f (${kb} KB)"; fi
    fi
  done
  [ -n "$fa" ] && [ -f "$fa" ] && cp "$fa" "$dst/failure_analysis.csv"
  [ -d "$src/plots" ] && { mkdir -p "$dst/plots"; cp "$src"/plots/*.png "$dst/plots/" 2>/dev/null; }
  (cd "$dst" && find . -type f ! -name CHECKSUMS.sha256 -print0 | sort -z | xargs -0 sha256sum > CHECKSUMS.sha256)
  echo "ok: $dst  ($(find "$dst" -type f | wc -l) files)"
}
copy_run results_v2_passive "$OUT/v2_tcs/passive" failure_analysis_v2_passive.csv
copy_run results_v2_gated   "$OUT/v2_tcs/gated"   failure_analysis_v2_gated.csv
[ -f config_v2_final.yaml ] && cp config_v2_final.yaml "$OUT/v2_tcs/config_v2_final.yaml"
[ -f ladder_summary.md ]    && cp ladder_summary.md "$OUT/v2_tcs/ladder_summary.md"
copy_run results_v3_gemini_events_abstain "$OUT/v3_event_vlm/real_gemini" failure_analysis_v3_gemini_events_abstain.csv
for d in results_v3_oracle_*; do
  [ -d "$d" ] || continue
  name="${d#results_v3_}"
  copy_run "$d" "$OUT/v3_event_vlm/$name" "failure_analysis_v3_${name}.csv"
done
du -sh "$OUT/v2_tcs" "$OUT/v3_event_vlm"
echo "Done. Next: cd .. && git status --short results/"
