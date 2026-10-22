#!/usr/bin/env bash
# Task 6, Ref-YouTube-VOS slice: V0, V1, V2-passive, one after another, unattended.
# Progress: tail -n 5 ladder_run.log      Temperatures: tail -n 3 gpu_temp.log
set -e
set -o pipefail
cd "$(dirname "$0")"
MAIN=$$

# GPU safety guard: logs the temperature every 15 s; if it is >= 85 C three reads in a row, it stops the run.
(
  cnt=0
  while true; do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    t=${t:-0}
    echo "$(date +%T) ${t}C" >> gpu_temp.log
    if [ "$t" -ge 85 ]; then cnt=$((cnt+1)); else cnt=0; fi
    if [ "$cnt" -ge 3 ]; then
      echo "$(date +%T) TOO HOT - stopping the run" >> gpu_temp.log
      echo "[$(date '+%F %T')] STOPPED BY GPU GUARD (temperature)" >> ladder_run.log
      pkill -f "stage2_" || true
      kill -TERM "$MAIN" 2>/dev/null || true
      exit 0
    fi
    sleep 15
  done
) &
GUARD=$!
trap 'kill $GUARD 2>/dev/null || true' EXIT

log() { echo "[$(date '+%F %T')] $*" | tee -a ladder_run.log; }
run() {   # run NAME command...   (progress bars are filtered out of the log; a failing command stops everything)
  local name="$1" rc; shift
  log "START $name"
  set +e
  "$@" 2>&1 | tr '\r' '\n' | { grep -v 'propagate in video\|frame loading\|Loading weights' || true; } >> ladder_run.log
  rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ]; then log "FAILED (exit $rc)  $name"; return "$rc"; fi
  log "DONE  $name"
}
soft() { "$@" || log "WARNING: step failed (not fatal): $*"; }

log "=== ladder start ==="
run "V0 stage1 grounding"     python stage1_grounding.py --config config.yaml
run "V0 stage2 propagation"   python stage2_propagation.py --config config.yaml
run "V0 stage3 evaluate"      python stage3_evaluate.py --config config.yaml
soft run "V0 stage4 failures" python stage4_failure_analysis.py --config config.yaml
soft run "V0 recovery"        python tools/recovery_metrics.py --config config.yaml --pred-dir predictions --out-dir recovery_v0 --label v0

run "V1 stage1b embeddings"   python stage1b_embeddings.py --config config_v1_final.yaml
run "V1 stage2 STTM"          python stage2_sttm.py --config config_v1_final.yaml
run "V1 stage3 evaluate"      python stage3_evaluate.py --config config_v1_final.yaml
soft run "V1 stage4 failures" python stage4_failure_analysis.py --config config_v1_final.yaml
soft run "V1 recovery"        python tools/recovery_metrics.py --config config_v1_final.yaml --pred-dir predictions_v1_final --out-dir recovery_v1 --label v1

run "V2 passive (all steps)"  bash tools/run_v2_full.sh passive
soft run "V2 recovery"        python tools/recovery_metrics.py --config config_v2_passive.yaml --pred-dir predictions_v2_passive --out-dir recovery_v2_passive --label v2_passive

log "=== ladder finished OK ==="
