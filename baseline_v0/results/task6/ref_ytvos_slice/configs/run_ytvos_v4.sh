#!/usr/bin/env bash
# Task 6, Ref-YouTube-VOS slice: candidate cache + V4-memory on ALL videos, with a GPU temperature guard.
# Run from ~/cogrvos/task6_ytvos/baseline_v0. Needs no API key. Logs: v4_run.log, gpu_temp_v4.log (UTC).
set -e
set -o pipefail
[ -f config_v2_final.yaml ] && [ -d predictions_v2_passive ] || { echo "STOP: run this inside task6_ytvos/baseline_v0"; exit 1; }

MAIN=$$
(
  hot=0
  while true; do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -n1)
    echo "$(date -u +%H:%M:%S) ${t}" >> gpu_temp_v4.log
    if [ "${t:-0}" -ge 85 ]; then hot=$((hot+1)); else hot=0; fi
    if [ "$hot" -ge 3 ]; then
      echo "$(date -u +%H:%M:%S) TOO HOT (${t} C three times): stopping" >> v4_run.log
      pkill -f "stage_candidates|stage_reid" || true
      kill "$MAIN" 2>/dev/null || true
      exit 0
    fi
    sleep 15
  done
) &
GUARD=$!
trap 'kill $GUARD 2>/dev/null || true' EXIT

F='propagate in video\|frame loading\|Loading weights\|skip '
say() { echo "$(date -u +%H:%M:%S) $*" | tee -a v4_run.log; }

say "start: candidates for all expressions that fire V3 events"
python stage_candidates.py --config config_v2_final.yaml \
  --events-csv results_v3_oracle_events_log_only/events_per_expression.csv 2>&1 \
  | tr '\r' '\n' | { grep -v "$F" || true; } | tee -a v4_run.log | tail -n 8

say "candidates done; start V4 memory"
V4_V3_CONFIG=config_v3_oracle_events_log_only.yaml bash tools/run_v4.sh memory 2>&1 \
  | tr '\r' '\n' | { grep -v "$F" || true; } | tee -a v4_run.log | tail -n 30

say "all done"
cat results_v4_memory/reid_summary.txt | tee -a v4_run.log
