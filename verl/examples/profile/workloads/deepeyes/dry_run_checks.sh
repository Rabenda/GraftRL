#!/usr/bin/env bash
# CPU-only checks for DeepEyes profile workload (no GPU rollout).
#
# Usage:
#   bash verl_vision/examples/profile/workloads/deepeyes/dry_run_checks.sh

set -euo pipefail

VERL_VISION_ROOT="${VERL_VISION_ROOT:-/workspace/repo/verl_vision}"
DATA_ROOT="${DATA_ROOT:-/data/deepeyes_visual_toolbox_v2}"

cd "${VERL_VISION_ROOT}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"

echo "[1/5] prepare data (skip download if parquet exists)"
bash examples/profile/workloads/deepeyes/prepare_data.sh

echo "[2/5] python imports + agent yaml"
python3 - <<'PY'
import importlib.util
import yaml
from pathlib import Path

checks = ["pyarrow", "PIL", "datasets"]
missing = [n for n in checks if importlib.util.find_spec(n) is None]
if missing:
    raise SystemExit(f"missing packages: {missing}")

from examples.profile.workloads.deepeyes.deepeyes_agent_loop import DeepEyesAgentLoop

cfg = yaml.safe_load(Path("examples/profile/workloads/deepeyes/deepeyes_agent_loop.yaml").read_text())
names = {entry["name"] for entry in cfg}
assert names >= {"deepeyes_agent", "deepeyes_visual_toolbox_v2"}, names
for entry in cfg:
    assert "DeepEyesAgentLoop" in entry["_target_"]
print("imports OK yaml_names=", sorted(names))
PY

echo "[3/5] zoom tool smoke"
python3 examples/profile/workloads/deepeyes/smoke_deepeyes_tools.py --parquet "${DATA_ROOT}/train.parquet" --row 0

echo "[4/5] rollout script syntax"
bash -n examples/profile/workloads/deepeyes/run_rollout_profile.sh
bash -n examples/profile/workloads/deepeyes/run_similarity.sh

echo "[5/5] batch size vs parquet rows"
python3 - <<PY
import pyarrow.parquet as pq
rows = pq.read_table("${DATA_ROOT}/train.parquet").num_rows
print(f"train rows={rows} (need >=64 for default bs64 rollout)")
if rows < 64:
    raise SystemExit("train.parquet too small for default TRAIN_BATCH_SIZE=64")
PY

echo ""
echo "dry_run_checks OK — when GPU is back:"
echo "  export CUDA_VISIBLE_DEVICES=0,1,2,3"
echo "  bash examples/profile/workloads/deepeyes/run_rollout_profile.sh"
