#!/usr/bin/env bash
# PHASE 2 - full 30-video val split with the FROZEN config_v2_final.yaml (~50 min per mode).
#   bash tools/run_v2_full.sh passive   TCS measures only. Masks = V1. Gives the TCS diagnostics on all 61 targets.
#   bash tools/run_v2_full.sh gated     V2: a frame is written to memory only if its band is HIGH.
set -e
cd "$(dirname "$0")/.."
mode="$1"
case "$mode" in
  passive) tag=v2_passive; gate=false ;;
  gated)   tag=v2_gated;   gate=true ;;
  *) echo "usage: bash tools/run_v2_full.sh passive|gated"; exit 1 ;;
esac
test -f config_v2_final.yaml || { echo "config_v2_final.yaml missing - run Phase 1 (tools/run_v2_tune.sh) first"; exit 1; }
sed -e "s#predictions_v2/#predictions_${tag}/#" -e "s#results_v2/#results_${tag}/#" \
    -e "s#visualizations_v2/#visualizations_${tag}/#" -e "s#failure_analysis_v2.csv#failure_analysis_${tag}.csv#" \
    -e "s/^\(\s*\)gate_writes: .*/\1gate_writes: ${gate}/" config_v2_final.yaml > config_${tag}.yaml
mkdir -p results_${tag}
echo "[1/4] stage 2, mode=${mode} (slow)"
python stage2_tcs.py --config config_${tag}.yaml 2>&1 | tee results_${tag}/run.log
echo "[2/4] J / F / J&F"
python stage3_evaluate.py --config config_${tag}.yaml
echo "[3/4] failure modes"
python stage4_failure_analysis.py --config config_${tag}.yaml
echo "[4/4] TCS analysis"
python tools/tcs_analysis.py --config config_${tag}.yaml
if [ "$mode" = "passive" ] && [ -d predictions_v1_final ]; then
  echo "sanity: passive masks vs V1 final (expect 0.0 frames differ on the MEAN line)"
  python tools/compare_v0_v1.py --config config_${tag}.yaml --v0 predictions_v1_final --v1 predictions_${tag} | tail -2
fi
echo
echo "DONE ${mode}. Results: results_${tag}/   failure modes: failure_analysis_${tag}.csv"
