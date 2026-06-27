#!/usr/bin/env bash
# Chart profile: clean parquet + forced teacher oracle (cross-branch turn1 identical).
#
# Logs:
#   profile_logs_vtool_chart_clean/
#
# Usage (bs64 × n4):
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash verl_vision/examples/profile/workloads/chart/run_clean_oracle_profile.sh

set -euo pipefail

VERL_VISION_ROOT="${VERL_VISION_ROOT:-/workspace/repo/verl_vision}"

export VTOOL_MODEL_REFOCUS=0
export VTOOL_ORACLE_DIVERSIFY=0
export VTOOL_ORACLE_REFOCUS=1
export TRAIN_FILE="${TRAIN_FILE:-/data/refocus_chart_multiturn_oracle_changed/train.parquet}"
export LOG_ROOT="${LOG_ROOT:-${VERL_VISION_ROOT}/profile_logs_vtool_chart_clean}"

exec bash "${VERL_VISION_ROOT}/examples/profile/workloads/chart/run_rollout_profile.sh" "$@"
