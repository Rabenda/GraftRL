#!/usr/bin/env bash
# Export browsable HTML galleries for Geo3K / stress / refocus datasets.
#
# Usage:
#   bash examples/profile/workloads/geo3k/export_dataset_gallery.sh
#   bash examples/profile/workloads/geo3k/export_dataset_gallery.sh geo3k_full

set -euo pipefail
cd /workspace/repo/verl_vision

OUT_ROOT="${OUT_ROOT:-/workspace/repo/verl_vision/profile_gallery}"
mkdir -p "${OUT_ROOT}"

export_one() {
  local name=$1 parquet=$2 max_rows=${3:-}
  local out="${OUT_ROOT}/${name}"
  local args=(--parquet "${parquet}" --out-dir "${out}" --split-name "${name}")
  if [[ -n "${max_rows}" ]]; then
    args+=(--max-rows "${max_rows}")
  fi
  echo "=== ${name} ==="
  python3 examples/profile/workloads/geo3k/inspect_dataset_samples.py "${args[@]}"
}

case "${1:-all}" in
  geo3k_stress)
    export_one geo3k_stress_1img data/geo3k_stress_1img/train.parquet
    ;;
  geo3k_full)
    export_one geo3k_train data/geo3k/train.parquet 16
    export_one geo3k_test data/geo3k/test.parquet 8
    ;;
  refocus)
    export_one refocus_chart data/refocus_chart/train.parquet 8
    ;;
  all)
    export_one geo3k_stress_1img data/geo3k_stress_1img/train.parquet
    export_one geo3k_train data/geo3k/train.parquet 16
    export_one refocus_chart data/refocus_chart/train.parquet 8
    ;;
  *)
    echo "Usage: $0 [all|geo3k_stress|geo3k_full|refocus]" >&2
    exit 1
    ;;
esac

echo ""
echo "Open galleries under: ${OUT_ROOT}/*/index.html"
