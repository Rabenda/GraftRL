#!/usr/bin/env bash
# Similarity analysis for clean + oracle rollout logs.

set -euo pipefail

VERL_VISION_ROOT="${VERL_VISION_ROOT:-/workspace/repo/verl_vision}"
LOG_ROOT="${LOG_ROOT:-${VERL_VISION_ROOT}/profile_logs_vtool_chart_clean}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-4}"
SUFFIX="${SUFFIX:-vtool_chart_bs${TRAIN_BATCH_SIZE}_n${ROLLOUT_N}}"

export LOG_ROOT SUFFIX
export DUMP_DIR="${DUMP_DIR:-${LOG_ROOT}/image_dump_${SUFFIX}}"
export OUT_DIR="${OUT_DIR:-${LOG_ROOT}/similarity}"
export CASE_DIR="${CASE_DIR:-${LOG_ROOT}/similarity/case_studies}"
export CASE_CROSS_DIR="${CASE_CROSS_DIR:-${LOG_ROOT}/similarity/case_studies_crossbranch}"

exec bash "${VERL_VISION_ROOT}/examples/profile/workloads/chart/run_similarity.sh" "$@"
