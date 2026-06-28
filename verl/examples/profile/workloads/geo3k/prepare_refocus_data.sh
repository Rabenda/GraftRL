#!/usr/bin/env bash
# Build Geo3K refocus-style multiturn parquet datasets.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${VERL_ROOT}"

SRC_DIR="${SRC_DIR:-data/geo3k}"
OUT_DIR="${OUT_DIR:-data}"
VARIANT="${VARIANT:-all}"

if [[ ! -f "${SRC_DIR}/train.parquet" ]]; then
  echo "Missing ${SRC_DIR}/train.parquet. Prepare Geo3K first, for example:" >&2
  echo "  python3 examples/data_preprocess/geo3k.py --local_save_dir data/geo3k" >&2
  exit 1
fi

python3 examples/profile/data_preprocess/geo3k/geo3k_refocus_multiturn.py \
  --src_dir "${SRC_DIR}" \
  --local_save_dir "${OUT_DIR}" \
  --variant "${VARIANT}" \
  "$@"

echo "Ready:"
for name in exact diversified stress; do
  dir="${OUT_DIR}/geo3k_refocus_${name}"
  if [[ -f "${dir}/train.parquet" ]]; then
    ls -lh "${dir}/"*.parquet
  fi
done
