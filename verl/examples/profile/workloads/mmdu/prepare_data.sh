#!/usr/bin/env bash
# Prepare a small MMDU benchmark subset for rollout profiling.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${VERL_ROOT}"

DATA_ROOT="${DATA_ROOT:-data/mmdu_benchmark_small}"
RAW_DIR="${RAW_DIR:-data/mmdu_raw_hf}"
MMDU_MAX_TRAIN_ROWS="${MMDU_MAX_TRAIN_ROWS:-256}"
MMDU_MAX_TEST_ROWS="${MMDU_MAX_TEST_ROWS:-64}"
MMDU_DATASET="${MMDU_DATASET:-benchmark}"
MMDU_MAX_DIALOGUES="${MMDU_MAX_DIALOGUES:-0}"
MMDU_TURNS_PER_DIALOGUE="${MMDU_TURNS_PER_DIALOGUE:-1}"
FORCE="${FORCE:-0}"

TRAIN_FILE="${DATA_ROOT}/train.parquet"
TEST_FILE="${DATA_ROOT}/test.parquet"

if [[ -f "${TRAIN_FILE}" && -f "${TEST_FILE}" && "${FORCE}" != "1" ]]; then
  echo "[mmdu] found existing parquet under ${DATA_ROOT} (set FORCE=1 to rebuild)"
else
  args=(
    --raw_dir "${RAW_DIR}"
    --local_save_dir "${DATA_ROOT}"
    --dataset "${MMDU_DATASET}"
    --max_train_rows "${MMDU_MAX_TRAIN_ROWS}"
    --max_test_rows "${MMDU_MAX_TEST_ROWS}"
    --turns_per_dialogue "${MMDU_TURNS_PER_DIALOGUE}"
  )
  if [[ "${MMDU_MAX_DIALOGUES}" != "0" ]]; then
    args+=(--max_dialogues "${MMDU_MAX_DIALOGUES}")
  fi
  if [[ "${FORCE}" == "1" ]]; then
    args+=(--force_download)
  fi
  python3 examples/profile/data_preprocess/mmdu/download_mmdu_benchmark.py "${args[@]}"
fi

python3 - <<PY
import pyarrow.parquet as pq

for split, path in [("train", "${TRAIN_FILE}"), ("test", "${TEST_FILE}")]:
    table = pq.read_table(path)
    row = table.slice(0, 1).to_pylist()[0]
    prompt = row["prompt"]
    image_count = len(row["images"])
    placeholder_count = sum(str(message["content"]).count("<image>") for message in prompt)
    assert image_count == placeholder_count, (image_count, placeholder_count)
    print(f"{split} rows={table.num_rows} images={image_count} prompt_messages={len(prompt)}")
PY

echo "[mmdu] ready:"
ls -lh "${DATA_ROOT}/"*.parquet
