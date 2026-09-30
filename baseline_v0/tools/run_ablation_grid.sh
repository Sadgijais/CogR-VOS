#!/usr/bin/env bash
# Run the pre-registered grid on the ablation tier. Run from anywhere; ~20 min per config on your GPU.
set -e
cd "$(dirname "$0")/.."
(cd subsets && sha256sum -c ref_davis17_ablation.json.sha256)
VIDS=$(python -c "import json;print(' '.join('--video '+v for v in json.load(open('subsets/ref_davis17_ablation.json'))['videos']))")
mkdir -p results_grid
make_cfg() {  # name base token delta
  sed -e "s/^\(\s*\)delta: .*/\1delta: $4/" \
      -e "s#predictions_$3#predictions_grid_$1#g" -e "s#results_$3#results_grid_$1#g" \
      -e "s#visualizations_$3#visualizations_grid_$1#g" -e "s#failure_analysis_$3#failure_analysis_grid_$1#g" \
      "$2" > config_grid_$1.yaml
}
make_cfg g0 config_v1.yaml        v1  5
make_cfg g1 config_v1_recent.yaml v1r 5
make_cfg g2 config_v1_recent.yaml v1r 10
make_cfg g3 config_v1_recent.yaml v1r 15
for g in g0 g1 g2 g3; do
  echo "=== $g ==="
  python stage2_sttm.py --config config_grid_$g.yaml $VIDS > results_grid/run_$g.log 2>&1
  python tools/compare_v0_v1.py --config config_grid_$g.yaml --v1 predictions_grid_$g > results_grid/compare_$g.txt
  tail -3 results_grid/compare_$g.txt
done
python tools/grid_summary.py results_grid/compare_g*.txt | tee results_grid/SUMMARY.txt
