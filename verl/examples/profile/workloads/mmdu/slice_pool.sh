#!/usr/bin/env bash
# Slice a standard 256/64 profiling subset from a filtered MMDU pool.
#
# Example:
#   SRC_ROOT=data/mmdu_45k_pool_filtered_8192 \
#   DST_ROOT=data/mmdu_benchmark_8192 \
#   bash examples/profile/workloads/mmdu/slice_pool.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${VERL_ROOT}"

SRC_ROOT="${SRC_ROOT:?set SRC_ROOT to filtered pool dir}"
DST_ROOT="${DST_ROOT:-data/mmdu_benchmark_8192}"
TRAIN_ROWS="${TRAIN_ROWS:-256}"
TEST_ROWS="${TEST_ROWS:-64}"

python3 - <<PY
import pyarrow.parquet as pq
from pathlib import Path

src = Path("${SRC_ROOT}")
dst = Path("${DST_ROOT}")
dst.mkdir(parents=True, exist_ok=True)

for split, n in [("train", int("${TRAIN_ROWS}")), ("test", int("${TEST_ROWS}"))]:
    src_path = src / f"{split}.parquet"
    table = pq.read_table(src_path)
    if table.num_rows < n:
        raise SystemExit(f"{src_path} has {table.num_rows} rows < requested {n}")
    out = table.slice(0, n)
    out_path = dst / f"{split}.parquet"
    pq.write_table(out, out_path)
    print(f"{split}: {table.num_rows} -> {out.num_rows} rows -> {out_path}")
PY

ls -lh "${DST_ROOT}/"*.parquet
