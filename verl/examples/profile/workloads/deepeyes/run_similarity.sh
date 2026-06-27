#!/usr/bin/env bash
# DeepEyes ViT token similarity + heatmaps (shared analyze_similarity_unified stack).
#
# Usage (after rollout, needs GPU):
#   export CUDA_VISIBLE_DEVICES=0
#   bash verl_vision/examples/profile/workloads/deepeyes/run_similarity.sh

set -euo pipefail

VERL_VISION_ROOT="${VERL_VISION_ROOT:-/workspace/repo/verl_vision}"
SUFFIX="${SUFFIX:-deepeyes_bs64_n4}"
LOG_ROOT="${LOG_ROOT:-${VERL_VISION_ROOT}/profile_logs_deepeyes}"
DUMP_DIR="${DUMP_DIR:-${LOG_ROOT}/image_dump_${SUFFIX}}"
OUT_DIR="${OUT_DIR:-${LOG_ROOT}/similarity}"
CASE_DIR="${OUT_DIR}/case_studies"
CASE_CROSS_DIR="${OUT_DIR}/case_studies_crossbranch"
MAX_GROUPS="${MAX_GROUPS:-64}"
GROUP_HEATMAPS="${GROUP_HEATMAPS:-8}"
PAIR_HEATMAPS="${PAIR_HEATMAPS:-12}"

cd "${VERL_VISION_ROOT}"

echo "[1/3] structure check"
python3 examples/profile/shared/analysis/analyze_similarity_unified.py \
  --dump-dir "${DUMP_DIR}" \
  --out-dir "${OUT_DIR}" \
  --check-only

echo "[2/3] ViT token similarity + heatmaps (needs GPU)"
python3 examples/profile/shared/analysis/analyze_similarity_unified.py \
  --dump-dir "${DUMP_DIR}" \
  --out-dir "${OUT_DIR}" \
  --max-groups "${MAX_GROUPS}" \
  --group-heatmaps "${GROUP_HEATMAPS}" \
  --pair-heatmaps "${PAIR_HEATMAPS}"

echo "[3/3] case studies (needs GPU)"
python3 examples/profile/shared/analysis/case_study_groups.py \
  --dump-dir "${DUMP_DIR}" \
  --pairwise-csv "${OUT_DIR}/pairwise_similarity.csv" \
  --out-dir "${CASE_DIR}" \
  --per-bucket 2

python3 examples/profile/shared/analysis/case_study_groups.py \
  --dump-dir "${DUMP_DIR}" \
  --pairwise-csv "${OUT_DIR}/pairwise_similarity.csv" \
  --out-dir "${CASE_CROSS_DIR}" \
  --select cross-branch \
  --top-groups 6

echo ""
echo "Done. LOG_ROOT=${LOG_ROOT}"
echo "  pairwise + summary: ${OUT_DIR}/pairwise_similarity.csv"
echo "  heatmaps:           ${OUT_DIR}/heatmaps/"
