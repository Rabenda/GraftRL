#!/usr/bin/env bash
# Prepare OSWorld GUI offline-replay parquet for GraftRL.
#
# Default: synthetic trajectories (no Docker). To convert real ARPO/OSWorld
# result dumps, set RESULTS_ROOT to a directory tree containing traj.jsonl.
#
# Usage:
#   bash examples/profile/workloads/osworld/prepare_data.sh
#   RESULTS_ROOT=/path/to/OSWorld/results bash examples/profile/workloads/osworld/prepare_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
REPO_ROOT="$(cd "${VERL_ROOT}/../.." && pwd)"
cd "${VERL_ROOT}"

if [[ -x /data/conda_envs/verl_vision/bin/python3 ]]; then
  export PATH="/data/conda_envs/verl_vision/bin:${PATH}"
fi

DEFAULT_TASK_ROOT="${REPO_ROOT}/ARPO/OSWorld/evaluation_examples"
USE_OSWORLD_TASKS="${USE_OSWORLD_TASKS:-1}"
if [[ -z "${DATA_ROOT:-}" ]]; then
  if [[ "${USE_OSWORLD_TASKS}" == "1" && -d "${DEFAULT_TASK_ROOT}" ]]; then
    DATA_ROOT="data/osworld_gui_tasksynth"
  else
    DATA_ROOT="data/osworld_gui_synth"
  fi
fi
RESULTS_ROOT="${RESULTS_ROOT:-}"
TASK_ROOT="${TASK_ROOT:-${DEFAULT_TASK_ROOT}}"
FORCE="${FORCE:-0}"
TRAIN_ROWS="${TRAIN_ROWS:-256}"
TEST_ROWS="${TEST_ROWS:-64}"
N_STEPS="${N_STEPS:-8}"
THOUGHT_CHARS="${THOUGHT_CHARS:-800}"
MAX_STEPS="${MAX_STEPS:-15}"
MAX_SIDE="${MAX_SIDE:-1280}"

TRAIN_FILE="${DATA_ROOT}/train.parquet"
TEST_FILE="${DATA_ROOT}/test.parquet"

if [[ -f "${TRAIN_FILE}" && -f "${TEST_FILE}" && "${FORCE}" != "1" && -z "${RESULTS_ROOT}" ]]; then
  echo "[osworld] found existing parquet under ${DATA_ROOT} (FORCE=1 to rebuild)"
else
  if [[ -n "${RESULTS_ROOT}" ]]; then
    echo "[osworld] converting real trajectories from ${RESULTS_ROOT}"
    python3 examples/profile/data_preprocess/osworld/convert_osworld_traj.py \
      --results-root "${RESULTS_ROOT}" \
      --output-dir "${DATA_ROOT}" \
      --max-steps "${MAX_STEPS}" \
      --max-side "${MAX_SIDE}" \
      --max-train-rows "${TRAIN_ROWS}" \
      --max-test-rows "${TEST_ROWS}"
  elif [[ "${USE_OSWORLD_TASKS}" == "1" && -d "${TASK_ROOT}" ]]; then
    echo "[osworld] building OSWorld task-backed GUI pool from ${TASK_ROOT}"
    python3 examples/profile/data_preprocess/osworld/make_task_synthetic_traj.py \
      --task-root "${TASK_ROOT}" \
      --output-dir "${DATA_ROOT}" \
      --train-rows "${TRAIN_ROWS}" \
      --test-rows "${TEST_ROWS}" \
      --n-steps "${N_STEPS}" \
      --thought-chars "${THOUGHT_CHARS}"
  else
    echo "[osworld] building synthetic GUI pool under ${DATA_ROOT}"
    python3 examples/profile/data_preprocess/osworld/make_synthetic_traj.py \
      --output-dir "${DATA_ROOT}" \
      --train-rows "${TRAIN_ROWS}" \
      --test-rows "${TEST_ROWS}" \
      --n-steps "${N_STEPS}" \
      --thought-chars "${THOUGHT_CHARS}"
  fi
fi

python3 - <<PY
import pyarrow.parquet as pq
for split, path in [("train", "${TRAIN_FILE}"), ("test", "${TEST_FILE}")]:
    table = pq.read_table(path)
    row = table.slice(0, 1).to_pylist()[0]
    extra = row["extra_info"]
    n_shots = len(extra.get("screenshots") or [])
    print(f"{split} rows={table.num_rows} screenshots={n_shots} instruction={str(extra.get('instruction',''))[:60]!r}")
PY

echo "[osworld] ready:"
ls -lh "${DATA_ROOT}/"*.parquet
