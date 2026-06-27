#!/usr/bin/env bash
# Run merged image-token output-level reuse probes across thresholds.
#
# This sweep validates the semantic tradeoff of lowering the merged-token
# cosine threshold. It is not expected to improve E2E latency yet because the
# target image still runs the full ViT before output-level token replacement.

set -euo pipefail

VERL_VISION_ROOT="${VERL_VISION_ROOT:-/workspace/repo/verl_vision}"
LOG_ROOT="${TOKEN_REUSE_LOG_ROOT:-${VERL_VISION_ROOT}/profile_logs_vtool_chart_merged_reuse_sweep}"
THRESHOLDS="${MERGED_TOKEN_THRESHOLDS:-0.95 0.90 0.85 0.80 0.75}"

cd "${VERL_VISION_ROOT}"

for threshold in ${THRESHOLDS}; do
  tag="t${threshold/./}"
  suffix="vtool_chart_merged_reuse_probe_${tag}"
  echo ""
  echo "=== merged-token reuse threshold=${threshold} suffix=${suffix} ==="

  rm -f "${LOG_ROOT}/vision_encoder_log_${suffix}.csv"
  rm -f "${LOG_ROOT}/model_forward_log_${suffix}.csv"
  rm -f "${LOG_ROOT}/verl_sglang_generate_log_${suffix}.csv"
  rm -rf "${LOG_ROOT}/image_dump_${suffix}"
  rm -rf "${LOG_ROOT}/rollout_data_${suffix}"

  TOKEN_REUSE_LOG_ROOT="${LOG_ROOT}" \
  TOKEN_REUSE_SUFFIX="${suffix}" \
  TOKEN_REUSE_GRANULARITY=merged \
  TOKEN_REUSE_THRESHOLD="${threshold}" \
  bash examples/profile/workloads/chart/run_token_reuse_probe.sh "$@" \
    2>&1 | tee "/tmp/${suffix}.log"
done

echo ""
echo "sweep logs: ${LOG_ROOT}"
echo "summary:"
echo "  /workspace/miniconda3/envs/verl_vision/bin/python3 \\"
echo "    examples/profile/shared/analysis/summarize_merged_token_reuse_sweep.py \\"
echo "    --log-root ${LOG_ROOT} --thresholds ${THRESHOLDS}"
