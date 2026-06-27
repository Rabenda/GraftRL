#!/usr/bin/env bash
# Run E2E-oriented merged-token-to-window reuse probes across thresholds.
#
# This is the accelerated path: a cheap patch-embed proxy estimates stable
# merged image tokens, maps them back to ViT windows, and skips donor-matched
# pre-full-attention window blocks.

set -euo pipefail

VERL_VISION_ROOT="${VERL_VISION_ROOT:-/workspace/repo/verl_vision}"
LOG_ROOT="${TOKEN_REUSE_LOG_ROOT:-${VERL_VISION_ROOT}/profile_logs_vtool_chart_merged_window_e2e_sweep}"
THRESHOLDS="${MERGED_WINDOW_THRESHOLDS:-0.90 0.85 0.80 0.75 0.70}"
MIN_TOKEN_RATIO="${MERGED_WINDOW_MIN_TOKEN_RATIO:-0.75}"

cd "${VERL_VISION_ROOT}"

for threshold in ${THRESHOLDS}; do
  tag="t${threshold/./}"
  ratio_tag="r${MIN_TOKEN_RATIO/./}"
  suffix="vtool_chart_merged_window_e2e_${ratio_tag}_${tag}"
  echo ""
  echo "=== merged-window E2E threshold=${threshold} min_token_ratio=${MIN_TOKEN_RATIO} suffix=${suffix} ==="

  rm -f "${LOG_ROOT}/vision_encoder_log_${suffix}.csv"
  rm -f "${LOG_ROOT}/model_forward_log_${suffix}.csv"
  rm -f "${LOG_ROOT}/verl_sglang_generate_log_${suffix}.csv"
  rm -rf "${LOG_ROOT}/image_dump_${suffix}"
  rm -rf "${LOG_ROOT}/rollout_data_${suffix}"

  TOKEN_REUSE_LOG_ROOT="${LOG_ROOT}" \
  TOKEN_REUSE_SUFFIX="${suffix}" \
  TOKEN_REUSE_GRANULARITY=merged_window \
  TOKEN_REUSE_THRESHOLD="${threshold}" \
  SGLANG_GRPO_MERGED_WINDOW_MIN_TOKEN_RATIO="${MIN_TOKEN_RATIO}" \
  bash examples/profile/workloads/chart/run_token_reuse_probe.sh "$@" \
    2>&1 | tee "/tmp/${suffix}.log"
done

echo ""
echo "sweep logs: ${LOG_ROOT}"
echo "summary:"
echo "  /workspace/miniconda3/envs/verl_vision/bin/python3 \\"
echo "    examples/profile/shared/analysis/summarize_merged_token_reuse_sweep.py \\"
echo "    --log-root ${LOG_ROOT} \\"
echo "    --suffix-prefix vtool_chart_merged_window_e2e_${ratio_tag} \\"
echo "    --thresholds ${THRESHOLDS}"
