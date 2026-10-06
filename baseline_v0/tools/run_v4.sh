#!/usr/bin/env bash
# V4 = V2 passive + memory-grounded re-identification. Live SAM 2 reruns of the expressions that can search. Design: V4_PLAN.md
#
#   bash tools/run_v4.sh off                      safety variant: never restarts. Masks must equal V2 passive (see tools/compare_pred_dirs.py)
#   bash tools/run_v4.sh oracle                   ground-truth chooser: a CEILING, never a result
#   bash tools/run_v4.sh memory                   memory score chooser, no VLM
#   bash tools/run_v4.sh vlm                      the V4 result.  First:  export GEMINI_API_KEY=...
#   bash tools/run_v4.sh oracle --video india --video kite-surf     partial run (smoke test): only those videos, recovery numbers only
#
# Needs: config_v2_passive.yaml + predictions_v2_passive/ (the V2 passive run), candidates/ (stage_candidates.py), and the V3 config
# whose event settings and VLM V4 reuses (default config_v3_gemini_events_abstain.yaml; override: V4_V3_CONFIG=...).
set -e
cd "$(dirname "$0")/.."
variant="$1"; shift || true
case "$variant" in off|oracle|memory|vlm) ;; *) echo "usage: bash tools/run_v4.sh off|oracle|memory|vlm [--video NAME ...]"; exit 1 ;; esac
src=v2_passive; tag="v4_${variant}"; v3cfg="${V4_V3_CONFIG:-config_v3_gemini_events_abstain.yaml}"
if [ "$variant" = "vlm" ] && [ -z "${GEMINI_API_KEY}" ]; then echo "set your key first:  export GEMINI_API_KEY=...your key..."; exit 1; fi
test -f "config_${src}.yaml" && test -d "predictions_${src}" || { echo "need the V2 passive run: config_${src}.yaml and predictions_${src}/"; exit 1; }
test -f "$v3cfg" || { echo "missing $v3cfg (the V3 config V4 takes its event settings and VLM from)"; exit 1; }
test -d candidates || { echo "missing candidates/ - run stage_candidates.py first"; exit 1; }
sed -e "s#predictions_${src}/#predictions_${tag}/#" -e "s#results_${src}/#results_${tag}/#" \
    -e "s#visualizations_${src}/#visualizations_${tag}/#" -e "s#failure_analysis_${src}.csv#failure_analysis_${tag}.csv#" \
    "config_${src}.yaml" > "config_${tag}.yaml"
mkdir -p "results_${tag}"
echo "[1/4] V4 live loop, variant=${variant}  ($*)"
python stage_reid.py --config "config_${tag}.yaml" --v3-config "$v3cfg" --variant "$variant" \
    --src-preds "predictions_${src}" --cand-dir candidates --pred-dir "predictions_${tag}" --results-dir "results_${tag}" "$@" \
    2>&1 | tee "results_${tag}/run.log"
if [ $# -gt 0 ]; then
  echo "[2/4] partial run: recovery numbers for the chosen videos only (J / F / J&F need the full run)"
  python tools/recovery_metrics.py --config "config_${tag}.yaml" --pred-dir "predictions_${tag}" --out-dir "recovery_${tag}_partial" --label "${tag}_partial" "$@" | sed -n 1,20p
  echo; echo "DONE (partial). Per-search log: results_${tag}/reid_log.csv   summary: results_${tag}/reid_summary.txt"
  exit 0
fi
echo "[2/4] videos without a live expression keep their V2 passive masks"
cp -rn "predictions_${src}/." "predictions_${tag}/"
echo "[3/4] J / F / J&F, failure modes, recovery"
python stage3_evaluate.py --config "config_${tag}.yaml"
python stage4_failure_analysis.py --config "config_${tag}.yaml"
python tools/recovery_metrics.py --config "config_${tag}.yaml" --pred-dir "predictions_${tag}" --out-dir "recovery_${tag}" --label "${tag}" > /dev/null
echo "[4/4] compare with the V2 passive run"
grep -h "mean_JF\|mean_J,\|mean_F," "results_${src}/summary.csv" | sed "s/^/   V2 (${src}): /"
grep -h "mean_JF\|mean_J,\|mean_F," "results_${tag}/summary.csv" | sed "s/^/   V4 (${variant}): /"
echo
echo "DONE. Searches, restarts, labels, VLM calls: results_${tag}/reid_summary.txt   per-search log: results_${tag}/reid_log.csv   recovery: recovery_${tag}/recovery_summary.txt"
