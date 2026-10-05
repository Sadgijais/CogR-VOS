#!/usr/bin/env bash
# V3 = V2 + event-driven VLM verification, run as an OFFLINE REPLAY of a finished V2 run.
# CPU only, no SAM 2, no GPU: minutes, not an hour. Design and fixed settings: V3_PLAN.md.
#
#   bash tools/run_v3.sh oracle                    free ceiling run (ground-truth oracle, no API key)
#   bash tools/run_v3.sh gemini                    real VLM; first:  export GEMINI_API_KEY=...   (openai | anthropic also work)
#   bash tools/run_v3.sh oracle periodic           control: same number of calls, evenly spaced (matched budget)
#   bash tools/run_v3.sh oracle every_frame        upper-bound cost: one call per tracked frame
#   bash tools/run_v3.sh gemini events log_only    only measure the VLM, change no mask
#   V3_ORACLE_IOU=0.10 bash tools/run_v3.sh oracle   post-hoc identity oracle (no_match only if the mask is on the wrong object / lost)
#
# Arguments: PROVIDER [SCHEDULE=events|periodic|every_frame] [ACTION=abstain|log_only]
# Source V2 run (default v2_gated):   V3_SOURCE=v2_passive bash tools/run_v3.sh oracle
set -e
cd "$(dirname "$0")/.."
prov="${1:-oracle}"; sched="${2:-events}"; act="${3:-abstain}"; src="${V3_SOURCE:-v2_gated}"
oiou="${V3_ORACLE_IOU:-}"
tag="v3_${prov}_${sched}_${act}${oiou:+_iou${oiou}}"
case "$prov" in
  oracle) key="" ;; gemini) key=GEMINI_API_KEY ;; openai) key=OPENAI_API_KEY ;; anthropic) key=ANTHROPIC_API_KEY ;;
  *) echo "provider must be oracle | gemini | openai | anthropic"; exit 1 ;;
esac
if [ -n "$key" ] && [ -z "${!key}" ]; then echo "set your key first:  export $key=...your key..."; exit 1; fi
test -f "config_${src}.yaml" && test -d "results_${src}/tcs_log" || {
  echo "need a finished V2 run: config_${src}.yaml and results_${src}/tcs_log/ (run: bash tools/run_v2_full.sh gated)"; exit 1; }

sed -e "s#predictions_${src}/#predictions_${tag}/#" -e "s#results_${src}/#results_${tag}/#" \
    -e "s#visualizations_${src}/#visualizations_${tag}/#" -e "s#failure_analysis_${src}.csv#failure_analysis_${tag}.csv#" \
    "config_${src}.yaml" > "config_${tag}.yaml"
cat >> "config_${tag}.yaml" <<EOF

# ---- V3 settings (fixed in advance, see V3_PLAN.md; none of these are tuned on the val split)
v3:
  source_results: results_${src}
  source_preds: predictions_${src}
  schedule: ${sched}          # events | periodic | every_frame
  period: auto                # periodic: auto = same number of calls as the events schedule
  action:
    mode: ${act}              # abstain | log_only
    conf_min: 0.6             # ignore VLM verdicts below this confidence
    recover_high: 5           # stop blanking after this many consecutive HIGH-coherence frames
    max_abstain: 40           # safety cap on one blanking run (frames)
  events:
    drop_delta: 0.30
    drop_window: 3
    persist_frames: 5
    margin_thresh: 0.0
    refractory: 10            # minimum frames between two calls on one target
    keepalive: 0              # 0 = off
    max_calls: 0              # 0 = unlimited per target
    fire_on_disappear: true
  vlm:
    provider: ${prov}
    model: null               # null = built-in default for the provider; set a model your account has if needed
    api_key_env: null
    oracle_iou: ${oiou:-0.5}      # oracle only. 0.5 = pre-registered; 0.10 = post-hoc identity oracle (V3_ORACLE_IOU=0.10)
    cache_dir: vlm_cache/
    workers: 8
    timeout_s: 60
    max_tokens: 200
EOF
mkdir -p "results_${tag}"
echo "[1/4] replay: events -> VLM (${prov}) -> verdicts -> masks   (${tag})"
python stage_events.py --config "config_${tag}.yaml" 2>&1 | tee "results_${tag}/run.log"
echo "[2/4] J / F / J&F"
python stage3_evaluate.py --config "config_${tag}.yaml"
echo "[3/4] failure modes"
python stage4_failure_analysis.py --config "config_${tag}.yaml"
echo "[4/4] compare with the source V2 run"
grep -h "mean_JF\|mean_J,\|mean_F," "results_${src}/summary.csv" | sed "s/^/   V2 (${src}): /"
grep -h "mean_JF\|mean_J,\|mean_F," "results_${tag}/summary.csv" | sed "s/^/   V3 (${tag}): /"
echo
echo "DONE. VLM calls, calls/frame, accuracy, latency: results_${tag}/v3_summary.txt   per-call log: results_${tag}/vlm_log.csv"
