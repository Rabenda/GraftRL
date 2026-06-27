#!/usr/bin/env bash
# Export per-request e2e decomposition CSV for geo3k vision stress runs.
#
# Usage (after run_geo3k_vision_stress.sh):
#   bash examples/profile/workloads/geo3k/export_stress_breakdown.sh          # 1img 2img 4img
#   bash examples/profile/workloads/geo3k/export_stress_breakdown.sh 2img     # single variant
#
# Requires profile CSVs under SGLANG_INFERENCE_LOG_DIR (default profile_logs_stress).

set -euo pipefail

cd /workspace/repo/verl_vision
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"

LOG_DIR="${SGLANG_INFERENCE_LOG_DIR:-/workspace/repo/verl_vision/profile_logs_stress}"
VARIANT=${1:-all}

export_one() {
  local tag=$1
  local suffix="stress_${tag}"
  local out="${LOG_DIR}/e2e_module_breakdown_${suffix}.csv"
  echo "=== Export ${suffix} -> ${out} ==="
  python3 examples/profile/shared/analysis/analyze_profiling_logs.py \
    --log-dir "${LOG_DIR}" \
    --suffix "${suffix}" \
    --export-breakdown-csv "${out}" \
    --epd-only
}

case "${VARIANT}" in
  1img|2img|4img) export_one "${VARIANT}" ;;
  all)
    export_one 1img
    export_one 2img
    export_one 4img
    echo ""
    echo "Summary files:"
    ls -1 "${LOG_DIR}"/e2e_module_breakdown_stress_*_summary.csv 2>/dev/null || true
    ;;
  *)
    echo "Unknown variant: ${VARIANT}. Use 1img|2img|4img|all" >&2
    exit 1
    ;;
esac
