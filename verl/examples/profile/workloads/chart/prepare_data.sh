#!/usr/bin/env bash
# Step 1: prepare VTool-R1 / Refocus Chart Split for verl_vision profiling.
#
# Data source: HuggingFace VTOOL/Refocus_Chart (same Refocus Chart Split used by
# VTool-R1 training). Original bar-chart image + oracle refocus code in extra_info.
#
# If you already have full parquet at /data/refocus_chart_multiturn (from an earlier
# download), this script skips re-download unless FORCE=1.
#
# Usage:
#   bash verl_vision/examples/profile/workloads/chart/prepare_data.sh
#
# Smoke subset (512 train / 128 test):
#   REFOCUS_MAX_TRAIN_ROWS=512 REFOCUS_MAX_TEST_ROWS=128 \
#     DATA_ROOT=/data/vtool_chart_smoke bash .../prepare_data.sh

set -euo pipefail

VERL_VISION_ROOT="${VERL_VISION_ROOT:-/workspace/repo/verl_vision}"
VTOOL_R1_ROOT="${VTOOL_R1_ROOT:-/workspace/repo/VTool-R1}"
DATA_ROOT="${DATA_ROOT:-/data/refocus_chart_multiturn}"
FORCE="${FORCE:-0}"
REFOCUS_MAX_TRAIN_ROWS="${REFOCUS_MAX_TRAIN_ROWS:-}"
REFOCUS_MAX_TEST_ROWS="${REFOCUS_MAX_TEST_ROWS:-128}"

TRAIN_FILE="${DATA_ROOT}/train.parquet"
TEST_FILE="${DATA_ROOT}/test.parquet"

cd "${VERL_VISION_ROOT}"

if [[ -f "${TRAIN_FILE}" && -f "${TEST_FILE}" && "${FORCE}" != "1" ]]; then
  echo "[prepare] found existing parquet under ${DATA_ROOT} (set FORCE=1 to re-download)"
else
  echo "[prepare] downloading VTOOL/Refocus_Chart -> ${DATA_ROOT}"
  _ARGS=(--local_save_dir "${DATA_ROOT}")
  if [[ -n "${REFOCUS_MAX_TRAIN_ROWS}" ]]; then
    _ARGS+=(--max_train_rows "${REFOCUS_MAX_TRAIN_ROWS}")
  else
    # Full train split is large (~15k rows, ~1GB). Omit cap to stream all rows.
    _ARGS+=(--max_train_rows 15170)
  fi
  _ARGS+=(--max_test_rows "${REFOCUS_MAX_TEST_ROWS}")
  python3 examples/profile/data_preprocess/chart/download_refocus_chart.py "${_ARGS[@]}"
fi

python3 - <<PY
import pyarrow.parquet as pq
from pathlib import Path

train = pq.read_table("${TRAIN_FILE}")
test = pq.read_table("${TEST_FILE}")
print(f"train rows={train.num_rows} cols={train.column_names}")
print(f"test  rows={test.num_rows}")

# Sanity: oracle refocus code + bbox metadata present
sample = train.slice(0, 1).to_pydict()
extra = (sample.get("extra_info") or [{}])[0]
assert extra.get("oracle_refocus_code"), "missing oracle_refocus_code in extra_info"
import json
tk = extra.get("tools_kwargs") or {}
raw_meta = tk.get("metadata") or {}
meta = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta)
has_bbox = bool(meta.get("x_values_bbox") or meta.get("y_values_bbox"))
print(f"oracle_refocus ok  chart_type={extra.get('source_chart')}  has_bbox={has_bbox}")
PY

if [[ -d "${VTOOL_R1_ROOT}" ]]; then
  echo "[prepare] VTool-R1 repo at ${VTOOL_R1_ROOT} (reference only; HF parquet is canonical for profile)"
fi

echo "[prepare] ready:"
ls -lh "${DATA_ROOT}/"*.parquet
