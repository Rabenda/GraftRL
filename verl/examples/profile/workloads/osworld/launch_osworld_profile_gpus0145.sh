#!/usr/bin/env bash
# Launch OSWorld GUI task-synth profile on free GPUs 0,1,4,5.
#
# From graftrl/verl:
#   bash examples/profile/workloads/osworld/launch_osworld_profile_gpus0145.sh
#
# Optional:
#   TOTAL_STEPS=2 bash .../launch_osworld_profile_gpus0145.sh
#   SGLANG_VLM_CACHEBLEND=1 bash .../launch_osworld_profile_gpus0145.sh
#   nohup bash .../launch_osworld_profile_gpus0145.sh > /tmp/osworld_0145.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${VERL_ROOT}"

if [[ -x /data/conda_envs/verl_vision/bin/python3 ]]; then
  export PATH="/data/conda_envs/verl_vision/bin:${PATH}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,4,5}"
export NGPUS="${NGPUS:-4}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export OSWORLD_RUNTIME_TURNS="${OSWORLD_RUNTIME_TURNS:-8}"
export TOTAL_STEPS="${TOTAL_STEPS:-1}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.25}"
export AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-8}"
export VERL_PROFILE_ROLLOUT_ONLY="${VERL_PROFILE_ROLLOUT_ONLY:-1}"

# Short path required: long RAY_TMPDIR breaks Ray Unix sockets.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/r4g}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-180}"
# Must be shell env string, not Hydra int.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
mkdir -p "${RAY_TMPDIR}"

LOG_ROOT="${LOG_ROOT:-${VERL_ROOT}/profile_logs_osworld_gui}"
mkdir -p "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="${LAUNCH_LOG:-${LOG_ROOT}/launch_gpus0145_${STAMP}.log}"

echo "[launch] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NGPUS=${NGPUS}"
echo "[launch] batch=${TRAIN_BATCH_SIZE} n=${ROLLOUT_N} turns=${OSWORLD_RUNTIME_TURNS} steps=${TOTAL_STEPS}"
echo "[launch] log=${LAUNCH_LOG}"
nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=index,name,memory.free --format=csv,noheader || true

# Extra Hydra overrides pass through via "$@".
bash examples/profile/workloads/osworld/run_osworld_profile.sh "$@" 2>&1 | tee "${LAUNCH_LOG}"
