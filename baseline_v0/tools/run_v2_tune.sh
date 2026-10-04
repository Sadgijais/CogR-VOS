#!/usr/bin/env bash
# PHASE 1 - tune the TCS band thresholds on the ABLATION tier (15 videos), passive run (~25 min).
# Passive = TCS only measures; masks must equal V1's. Writes config_v2_final.yaml with tuned thresholds.
# Run from anywhere:   bash tools/run_v2_tune.sh
set -e
cd "$(dirname "$0")/.."
echo "[1/6] checking the ablation manifest has not changed"
(cd subsets && sha256sum -c ref_davis17_ablation.json.sha256)
VIDS=$(python -c "import json;print(' '.join('--video '+v for v in json.load(open('subsets/ref_davis17_ablation.json'))['videos']))")
echo "[2/6] building config_v2_tune.yaml (own folders, gate OFF)"
sed -e 's#predictions_v2/#predictions_v2_tune/#' -e 's#results_v2/#results_v2_tune/#' \
    -e 's#visualizations_v2/#visualizations_v2_tune/#' -e 's#failure_analysis_v2.csv#failure_analysis_v2_tune.csv#' \
    -e 's/^\(\s*\)gate_writes: .*/\1gate_writes: false/' config_v2.yaml > config_v2_tune.yaml
mkdir -p results_v2_tune
echo "[3/6] running stage 2 (SAM 2 + memory + TCS) on the ablation tier - the slow step"
python stage2_tcs.py --config config_v2_tune.yaml $VIDS 2>&1 | tee results_v2_tune/run.log
echo "[4/6] scoring J / F / J&F"
python stage3_evaluate.py --config config_v2_tune.yaml 2>&1 | grep -v "^skip " | tail -8
echo "[5/6] sanity: passive TCS must not change the masks (compare with V1 final)"
if [ -d predictions_v1_final ]; then
  python tools/compare_v0_v1.py --config config_v2_tune.yaml --v0 predictions_v1_final --v1 predictions_v2_tune $VIDS | tail -3
  echo "   -> expect %frames_differ = 0.0 on the MEAN line"
else
  echo "   (predictions_v1_final/ not found here - skip this check, or point --v0 at your V1 predictions)"
fi
echo "[6/6] TCS analysis + threshold suggestion (ablation tier only)"
python tools/tcs_analysis.py --config config_v2_tune.yaml --manifest subsets/ref_davis17_ablation.json \
    --suggest --write-config config_v2_final.yaml --template config_v2.yaml
echo
echo "DONE. Read results_v2_tune/TCS_ANALYSIS.txt and config_v2_final.yaml, then COMMIT config_v2_final.yaml before Phase 2."
