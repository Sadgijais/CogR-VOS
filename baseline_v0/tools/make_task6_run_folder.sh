#!/usr/bin/env bash
# Task 6: build a run folder for ONE converted dataset from the FROZEN repo code and configs.
# Only the dataset root and the tier name are changed in the configs (same rule as for Ref-YouTube-VOS).
#   usage: bash tools/make_task6_run_folder.sh <converted dataset root> <new run folder> [tier name]
# Example: bash tools/make_task6_run_folder.sh ~/data/MeViS-pilot ~/cogrvos/task6_mevis_pilot/baseline_v0 mevis_pilot
# Never writes into the dataset. Refuses to use a non-empty run folder.
set -e
ROOT="$(readlink -f "$1" 2>/dev/null || true)"; DST="$2"; TIER="${3:-task6}"
[ -n "$ROOT" ] && [ -n "$DST" ] || { echo "usage: bash tools/make_task6_run_folder.sh <converted dataset root> <new run folder> [tier]"; exit 1; }
[ -f "$ROOT/meta_expressions/meta_expressions.json" ] || { echo "no converted dataset at $ROOT"; exit 1; }
if [ -d "$DST" ] && [ -n "$(ls -A "$DST" 2>/dev/null)" ]; then echo "REFUSING: $DST is not empty"; exit 1; fi
HERE="$(cd "$(dirname "$0")/.." && pwd)"       # the repo's baseline_v0 folder (frozen code)
mkdir -p "$DST/subsets"
cp "$HERE"/*.py "$DST"/
cp -r "$HERE/common" "$HERE/tools" "$DST"/
find "$DST" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
for c in config.yaml config_v1_final.yaml config_v2_final.yaml; do
  cp "$HERE/$c" "$DST/$c"
  [ "$(grep -c '^[[:space:]]*root:' "$DST/$c")" = "1" ] && [ "$(grep -c '^[[:space:]]*tier:' "$DST/$c")" = "1" ] \
    || { echo "STOP: $c does not have exactly one root: and one tier: line"; exit 1; }
  sed -i -E "s|^([[:space:]]*root:[[:space:]]*).*|\1$ROOT|; s|^([[:space:]]*tier:[[:space:]]*).*|\1$TIER|" "$DST/$c"
done
cp "$HERE/results/task6/ref_ytvos_slice/configs/run_ytvos_ladder.sh" "$DST/run_ladder.sh"
cp "$HERE/results/task6/ref_ytvos_slice/configs/run_ytvos_v4.sh" "$DST/run_v4_memory.sh" 2>/dev/null || true
python - "$ROOT" "$DST/subsets/ref_davis17_${TIER}.json" "$TIER" <<'PY'
import json, sys
root, out, tier = sys.argv[1:4]
v = sorted(json.load(open(root + "/meta_expressions/meta_expressions.json"))["videos"])
json.dump({"videos": v, "note": f"Task 6 tier '{tier}': every video in the converted dataset at {root}"}, open(out, "w"), indent=1)
print(f"tier manifest: {len(v)} videos -> {out}")
PY
echo "run folder ready: $DST"
echo "--- lines changed in the configs (dataset root and tier only) ---"
grep -n "^[[:space:]]*root:\|^[[:space:]]*tier:" "$DST"/config*.yaml
