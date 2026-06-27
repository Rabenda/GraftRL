#!/usr/bin/env bash
# Chart profile: clean parquet + model-driven refocus (no teacher oracle at rollout).
#
# Logs (isolated from origin / clean-oracle / diversified):
#   profile_logs_vtool_chart_model_refocus/
#
# Usage (bs64 × n4 only — smaller batches are not meaningful for this workload):
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash verl_vision/examples/profile/workloads/chart/run_model_refocus_profile.sh

set -euo pipefail

VERL_VISION_ROOT="${VERL_VISION_ROOT:-/workspace/repo/verl_vision}"

export VTOOL_MODEL_REFOCUS=1
export VTOOL_ORACLE_DIVERSIFY=0
export TRAIN_FILE="${TRAIN_FILE:-/data/refocus_chart_multiturn_oracle_changed/train.parquet}"
export LOG_ROOT="${LOG_ROOT:-${VERL_VISION_ROOT}/profile_logs_vtool_chart_model_refocus}"

exec bash "${VERL_VISION_ROOT}/examples/profile/workloads/chart/run_rollout_profile.sh" "$@"
