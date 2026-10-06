#!/usr/bin/env bash
# Run every CPU-only test (no GPU, no dataset). Each in its own process so the fakes do not leak.
# Run from anywhere:   bash tools/run_all_tests.sh
set -e
cd "$(dirname "$0")/.."
for t in test_sttm test_sttm_glue test_tcs test_tcs_analysis test_stage2_tcs_smoke test_events test_vlm test_stage_events test_recovery_metrics test_stage_candidates test_reid; do
  echo "=== $t ==="
  python tools/$t.py 2>&1 | grep -E "^(ok|all|smoke|Traceback|Assertion|  File|    )" | tail -4
  python tools/$t.py >/dev/null 2>&1 || { echo "FAILED: $t"; exit 1; }
done
echo; echo "ALL TESTS PASSED"
