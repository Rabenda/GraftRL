#!/usr/bin/env bash
# Similarity analysis for model-refocus rollout logs.
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0
#   bash verl_vision/examples/profile/workloads/chart/run_similarity_model_refocus.sh

set -euo pipefail

VERL_VISION_ROOT="${VERL_VISION_ROOT:-/workspace/repo/verl_vision}"
LOG_ROOT="${LOG_ROOT:-${VERL_VISION_ROOT}/profile_logs_vtool_chart_model_refocus}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-4}"
SUFFIX="${SUFFIX:-vtool_chart_bs${TRAIN_BATCH_SIZE}_n${ROLLOUT_N}_model_refocus}"

export LOG_ROOT
export SUFFIX
export DUMP_DIR="${DUMP_DIR:-${LOG_ROOT}/image_dump_${SUFFIX}}"
export OUT_DIR="${OUT_DIR:-${LOG_ROOT}/similarity}"
export CASE_DIR="${CASE_DIR:-${LOG_ROOT}/similarity/case_studies}"
export CASE_CROSS_DIR="${CASE_CROSS_DIR:-${LOG_ROOT}/similarity/case_studies_crossbranch}"

exec bash "${VERL_VISION_ROOT}/examples/profile/workloads/chart/run_similarity.sh" "$@"
