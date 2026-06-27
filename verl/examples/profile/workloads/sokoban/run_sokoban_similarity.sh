#!/usr/bin/env bash
# Sokoban ViT token similarity + heatmaps (Refocus analyze_similarity_unified stack).
#
# Usage (GPU 5):
#   export CUDA_VISIBLE_DEVICES=5
#   bash verl_vision/examples/profile/workloads/sokoban/run_sokoban_similarity.sh
#
# Optional env:
#   DUMP_DIR=...  OUT_DIR=...  MAX_GROUPS=64  GROUP_HEATMAPS=8  PAIR_HEATMAPS=12

set -euo pipefail

VERL_VISION_ROOT="${VERL_VISION_ROOT:-/workspace/repo/verl_vision}"
DUMP_DIR="${DUMP_DIR:-${VERL_VISION_ROOT}/profile_logs_sokoban/image_dump_sokoban_bs64_n4}"
OUT_DIR="${OUT_DIR:-${VERL_VISION_ROOT}/profile_logs_sokoban/similarity}"
CASE_DIR="${CASE_DIR:-${VERL_VISION_ROOT}/profile_logs_sokoban/case_studies}"
MAX_GROUPS="${MAX_GROUPS:-64}"
GROUP_HEATMAPS="${GROUP_HEATMAPS:-8}"
PAIR_HEATMAPS="${PAIR_HEATMAPS:-12}"
MAX_STEP="${MAX_STEP:-10}"
CASE_GROUPS="${CASE_GROUPS:-4,16,49}"

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

echo "[3/3] visual case study PNG grid (no GPU)"
python3 examples/profile/workloads/sokoban/export_sokoban_case_study.py \
  --dump-dir "${DUMP_DIR}" \
  --out-dir "${CASE_DIR}" \
  --groups "${CASE_GROUPS}" \
  --max-step "${MAX_STEP}"

echo ""
echo "Done."
echo "  pairwise + summary: ${OUT_DIR}/pairwise_similarity.csv"
echo "  heatmaps:           ${OUT_DIR}/heatmaps/"
echo "  case study HTML:    ${CASE_DIR}/index.html"
