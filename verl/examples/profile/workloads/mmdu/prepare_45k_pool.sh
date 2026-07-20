#!/usr/bin/env bash
# Build a larger MMDU pool from MMDU-45k for offline <=8192 filtering.
#
# The 110-dialogue benchmark caps at ~550 rows; after <=8192 filter only ~92 train
# rows remain.  MMDU-45k has ~45k dialogues (avg ~5k tokens) so we can harvest
# enough short-context samples for 64×4 profiling.
#
# Usage:
#   export PATH=/data/conda_envs/verl_vision/bin:$PATH
#   bash examples/profile/workloads/mmdu/prepare_45k_pool.sh
#
# Then filter:
#   python3 examples/profile/workloads/mmdu/filter_overlong.py \
#     --input-dir data/mmdu_45k_pool \
#     --output-dir data/mmdu_45k_pool_filtered_8192 \
#     --max-prompt-length 8192

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${VERL_ROOT}"

export PATH="${PATH:-}"
if [[ -x /data/conda_envs/verl_vision/bin/python3 ]]; then
  export PATH="/data/conda_envs/verl_vision/bin:${PATH}"
fi

RAW_DIR="${RAW_DIR:-data/mmdu_raw_hf}"
DATA_ROOT="${DATA_ROOT:-data/mmdu_45k_pool}"
MMDU_DATASET="${MMDU_DATASET:-45k}"
# Scan enough 45k dialogues to survive <=8192 filtering with train>=256.
MMDU_MAX_DIALOGUES="${MMDU_MAX_DIALOGUES:-2500}"
MMDU_MAX_TRAIN_ROWS="${MMDU_MAX_TRAIN_ROWS:-1200}"
MMDU_MAX_TEST_ROWS="${MMDU_MAX_TEST_ROWS:-300}"
MMDU_TURNS_PER_DIALOGUE="${MMDU_TURNS_PER_DIALOGUE:-1}"
FORCE="${FORCE:-0}"

TRAIN_FILE="${DATA_ROOT}/train.parquet"
TEST_FILE="${DATA_ROOT}/test.parquet"

if [[ -f "${TRAIN_FILE}" && -f "${TEST_FILE}" && "${FORCE}" != "1" ]]; then
  echo "[mmdu-45k] found existing parquet under ${DATA_ROOT} (set FORCE=1 to rebuild)"
else
  args=(
    --raw_dir "${RAW_DIR}"
    --local_save_dir "${DATA_ROOT}"
    --dataset "${MMDU_DATASET}"
    --max_dialogues "${MMDU_MAX_DIALOGUES}"
    --max_train_rows "${MMDU_MAX_TRAIN_ROWS}"
    --max_test_rows "${MMDU_MAX_TEST_ROWS}"
    --turns_per_dialogue "${MMDU_TURNS_PER_DIALOGUE}"
  )
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

echo "[mmdu-45k] ready:"
ls -lh "${DATA_ROOT}/"*.parquet
