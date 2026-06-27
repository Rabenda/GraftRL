#!/usr/bin/env bash
# VTool Chart ViT token similarity + heatmaps + case studies.
#
# Usage (GPU 5):
#   export CUDA_VISIBLE_DEVICES=5
#   bash verl_vision/examples/profile/workloads/chart/run_similarity.sh
#
# Optional env:
#   DUMP_DIR=...  OUT_DIR=...  MAX_GROUPS=64  GROUP_HEATMAPS=8  PAIR_HEATMAPS=12

set -euo pipefail

VERL_VISION_ROOT="${VERL_VISION_ROOT:-/workspace/repo/verl_vision}"
SUFFIX="${SUFFIX:-vtool_chart_bs64_n4}"

# LOG_ROOT selects the experiment line; set explicitly or via run_similarity_*.sh wrappers.
LOG_ROOT="${LOG_ROOT:-${VERL_VISION_ROOT}/profile_logs_vtool_chart_raw}"
DUMP_DIR="${DUMP_DIR:-${LOG_ROOT}/image_dump_${SUFFIX}}"
OUT_DIR="${OUT_DIR:-${LOG_ROOT}/similarity}"
CASE_DIR="${CASE_DIR:-${LOG_ROOT}/similarity/case_studies}"
CASE_CROSS_DIR="${CASE_CROSS_DIR:-${LOG_ROOT}/similarity/case_studies_crossbranch}"
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

echo "[3/3] auto-selected case study groups (needs GPU for token stats)"
python3 examples/profile/shared/analysis/case_study_groups.py \
  --dump-dir "${DUMP_DIR}" \
  --pairwise-csv "${OUT_DIR}/pairwise_similarity.csv" \
  --out-dir "${CASE_DIR}" \
  --per-bucket 2

echo "[3b] cross-branch case studies (groups with >=2 distinct turn1)"
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
echo "  case studies:       ${CASE_DIR}/"
echo "  cross-branch:       ${CASE_CROSS_DIR}/"
