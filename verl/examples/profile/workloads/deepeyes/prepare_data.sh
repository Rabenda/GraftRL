#!/usr/bin/env bash
# Step 1: prepare DeepEyes visual_toolbox_v2 parquet for verl_vision profiling.
#
# Source: HuggingFace ChenShawn/DeepEyes-Datasets-47k (env_name=visual_toolbox_v2)
#
# Usage:
#   bash verl_vision/examples/profile/workloads/deepeyes/prepare_data.sh
#
# Smoke subset:
#   DEEPEYES_MAX_ROWS=512 DEEPEYES_TEST_ROWS=64 bash .../prepare_data.sh

set -euo pipefail

VERL_VISION_ROOT="${VERL_VISION_ROOT:-/workspace/repo/verl_vision}"
DATA_ROOT="${DATA_ROOT:-/data/deepeyes_visual_toolbox_v2}"
FORCE="${FORCE:-0}"
DEEPEYES_MAX_ROWS="${DEEPEYES_MAX_ROWS:-2000}"
DEEPEYES_TEST_ROWS="${DEEPEYES_TEST_ROWS:-200}"

TRAIN_FILE="${DATA_ROOT}/train.parquet"
TEST_FILE="${DATA_ROOT}/test.parquet"

cd "${VERL_VISION_ROOT}"

if [[ -f "${TRAIN_FILE}" && -f "${TEST_FILE}" && "${FORCE}" != "1" ]]; then
  echo "[prepare] found existing parquet under ${DATA_ROOT} (set FORCE=1 to re-download)"
elif [[ -f "${TRAIN_FILE}" && ! -f "${TEST_FILE}" && "${FORCE}" != "1" ]]; then
  echo "[prepare] train exists but test missing — holding out last ${DEEPEYES_TEST_ROWS} rows"
  python3 - <<PY
import pyarrow.parquet as pq
from pathlib import Path

root = Path("${DATA_ROOT}")
train_path = root / "train.parquet"
test_path = root / "test.parquet"
holdout = int("${DEEPEYES_TEST_ROWS}")
table = pq.read_table(train_path)
if table.num_rows <= holdout + 64:
    raise SystemExit(f"train rows={table.num_rows} too small to hold out {holdout} for test")
cut = table.num_rows - holdout
pq.write_table(table.slice(0, cut), train_path)
pq.write_table(table.slice(cut, holdout), test_path)
print(f"split train={cut} test={holdout}")
PY
else
  echo "[prepare] downloading ChenShawn/DeepEyes-Datasets-47k -> ${DATA_ROOT}"
  python3 examples/profile/data_preprocess/deepeyes/download_visual_toolbox_v2.py \
    --local_save_dir "${DATA_ROOT}" \
    --max_rows "${DEEPEYES_MAX_ROWS}" \
    --max_test_rows "${DEEPEYES_TEST_ROWS}"
fi

python3 - <<PY
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

train = pq.read_table("${TRAIN_FILE}")
test = pq.read_table("${TEST_FILE}")
print(f"train rows={train.num_rows} cols={train.column_names}")
print(f"test  rows={test.num_rows}")

sample = train.slice(0, 1).to_pylist()[0]
assert sample.get("images"), "missing images"
assert sample.get("prompt"), "missing prompt"
assert sample.get("agent_name") in ("deepeyes_agent", "deepeyes_visual_toolbox_v2"), sample.get("agent_name")
assert sample.get("env_name") == "visual_toolbox_v2", sample.get("env_name")
extra = sample.get("extra_info") or {}
assert extra.get("question"), "missing extra_info.question"
print(f"ok data_source={sample.get('data_source')} question={extra.get('question')[:80]!r}")
PY

echo "[prepare] ready:"
ls -lh "${DATA_ROOT}/"*.parquet
