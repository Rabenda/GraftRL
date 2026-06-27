#!/usr/bin/env bash
# Token-level partial ViT reuse for Chart refocus (route B: token-sparse compute).
#
# Modes (TOKEN_REUSE_MODE):
#   token_sparse  (default) reused tokens skip their FFN; changed tokens run full
#                 window attention with donor hidden states as K/V. This is the
#                 compute-saving path -- reused tokens do NOT re-run the MLP.
#   token         probe path: full block runs, donor hidden states overwrite
#                 reused tokens afterwards (no compute saved; mechanism check).
#   window        window-granularity reuse (whole similar windows skipped).
#   merged        merged image-token output-level reuse diagnostics.
#   baseline      cache OFF -- reference timing for speedup measurement.
#
# For a clean speedup measurement run `baseline` and `token_sparse` with the
# SAME data/batch and read vision_encoder_time_ms via
# examples/profile/shared/analysis/compare_token_sparse_timing.py.
#
# NOTE: merged-token-sim logging runs a second full ViT forward and MUST be off
# when measuring speed. It defaults off here; set TOKEN_REUSE_LOG_MERGED_SIM=1
# only for similarity diagnostics.

set -euo pipefail

VERL_VISION_ROOT="${VERL_VISION_ROOT:-/workspace/repo/verl_vision}"

export PATH="/workspace/miniconda3/envs/verl_vision/bin:${PATH:-}"
export PYTHON="${PYTHON:-/workspace/miniconda3/envs/verl_vision/bin/python3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NGPUS="${NGPUS:-4}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"

export TRAIN_FILE="${TOKEN_REUSE_TRAIN_FILE:-/data/refocus_chart_multiturn_oracle_changed/train.parquet}"
export DATA_ROOT="${TOKEN_REUSE_DATA_ROOT:-/data/refocus_chart_multiturn_oracle_changed}"
export LOG_ROOT="${TOKEN_REUSE_LOG_ROOT:-${VERL_VISION_ROOT}/profile_logs_vtool_chart_token_reuse}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export TOTAL_STEPS="${TOTAL_STEPS:-1}"

export VTOOL_ORACLE_DIVERSIFY="${VTOOL_ORACLE_DIVERSIFY:-1}"

TOKEN_REUSE_MODE="${TOKEN_REUSE_MODE:-token_sparse}"
TOKEN_REUSE_THRESHOLD="${TOKEN_REUSE_THRESHOLD:-0.84}"
# Tag like t084 / t090 from the threshold (0.84 -> t084, 0.90 -> t090).
_THR_TAG="t$(printf '%s' "${TOKEN_REUSE_THRESHOLD}" | sed 's/\.//g')"

if [[ "${TOKEN_REUSE_MODE}" == "baseline" ]]; then
  export SGLANG_GRPO_SIM_CACHE=0
  export SUFFIX="${TOKEN_REUSE_SUFFIX:-vtool_chart_baseline_nocache}"
else
  export SGLANG_GRPO_SIM_CACHE=1
  export SGLANG_GRPO_REUSE_MODE=token_or_window_partial_reuse
  export SGLANG_GRPO_ENABLE_PARTIAL_VIT_REUSE=1
  export SGLANG_GRPO_PARTIAL_REUSE_GRANULARITY="${TOKEN_REUSE_MODE}"
  export SGLANG_GRPO_SIM_TARGET_TURNS="${SGLANG_GRPO_SIM_TARGET_TURNS:-1}"
  export SGLANG_GRPO_LOG_MERGED_TOKEN_SIM="${TOKEN_REUSE_LOG_MERGED_SIM:-0}"
  # Slot-level gate only selects candidate donor pairs; the per-token decision
  # uses SGLANG_GRPO_PARTIAL_REUSE_THRESHOLD on patch-hidden cosine.
  export SGLANG_GRPO_SIM_RAW_COSINE_THRESH="${TOKEN_REUSE_SLOT_COSINE_THRESHOLD:-0.90}"
  export SGLANG_GRPO_SIM_RAW_COSINE_RATIO="${TOKEN_REUSE_SLOT_COSINE_RATIO:-0.0}"
  export SGLANG_GRPO_PARTIAL_REUSE_THRESHOLD="${TOKEN_REUSE_THRESHOLD}"
  export SUFFIX="${TOKEN_REUSE_SUFFIX:-vtool_chart_${TOKEN_REUSE_MODE}_${_THR_TAG}}"
fi

echo "TOKEN_REUSE_MODE=${TOKEN_REUSE_MODE} THRESHOLD=${TOKEN_REUSE_THRESHOLD} SUFFIX=${SUFFIX}"
echo "LOG_MERGED_TOKEN_SIM=${SGLANG_GRPO_LOG_MERGED_TOKEN_SIM:-0} (must be 0 for timing)"

cd "${VERL_VISION_ROOT}"
bash examples/profile/workloads/chart/run_rollout_profile.sh "$@"
