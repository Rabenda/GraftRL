#!/usr/bin/env bash
# Real-scale Geo3K sparse-decoding A/B. Both arms keep the same LLM-prefill
# CacheBlend path; only decode-side context sparsification changes.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1,2 bash examples/profile/workloads/geo3k/run_geo3k_sparse_decode_ab.sh control
#   CUDA_VISIBLE_DEVICES=1,2 bash examples/profile/workloads/geo3k/run_geo3k_sparse_decode_ab.sh sparse

set -euo pipefail

MODE="${1:-}"
VARIANT="${2:-stress}"
if [[ "${MODE}" != "control" && "${MODE}" != "sparse" ]]; then
  echo "Usage: $0 {control|sparse} [exact|diversified|stress]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${VERL_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NGPUS="${NGPUS:-2}"

# Formal runs must start from idle physical devices. Refuse before Ray/model init so
# an external job cannot turn a measurement into contention noise or wasted GPU time.
IFS=',' read -r -a _geo3k_gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
for _geo3k_gpu_id in "${_geo3k_gpu_ids[@]}"; do
  _geo3k_gpu_state="$(/usr/bin/nvidia-smi \
    --id="${_geo3k_gpu_id}" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | head -1)"
  IFS=',' read -r _geo3k_gpu_mem _geo3k_gpu_util <<< "${_geo3k_gpu_state}"
  _geo3k_gpu_mem="${_geo3k_gpu_mem//[[:space:]]/}"
  _geo3k_gpu_util="${_geo3k_gpu_util//[[:space:]]/}"
  if (( _geo3k_gpu_mem > 2048 || _geo3k_gpu_util > 10 )); then
    echo "Refusing formal run: physical GPU ${_geo3k_gpu_id} is busy " \
      "(${_geo3k_gpu_mem} MiB, ${_geo3k_gpu_util}% util)." >&2
    exit 2
  fi
done
unset _geo3k_gpu_ids _geo3k_gpu_id _geo3k_gpu_state \
  _geo3k_gpu_mem _geo3k_gpu_util

export TRAIN_BATCH_SIZE=64
export ROLLOUT_N=4
export TOTAL_STEPS="${TOTAL_STEPS:-2}"
export AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-2}"
export CACHEBLEND_SELECTOR=kvdev
export CACHEBLEND_IMAGE_SLOTS=-1
export CACHEBLEND_FAST_APPLY=0
export CACHEBLEND_COMPACT_PREFILL=0
export RUN_TAG="${RUN_TAG:-sparse_decode_${MODE}}"

# Do not inherit old sparse-policy experiments; the runtime owns these defaults.
unset SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MODE \
      SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_RECENT \
      SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_FIRST \
      SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROPPED_TOKENS \
      SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROP_RATIO \
      SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_INCREMENTAL || true
export SGLANG_VLM_CACHEBLEND_SPARSE_DECODE=0
if [[ "${MODE}" == "sparse" ]]; then
  # All policy/kernel knobs use the validated runtime defaults. Keep the formal
  # A/B interface to one switch so stale tuning env cannot silently change it.
  export SGLANG_VLM_CACHEBLEND_SPARSE_DECODE=1
fi

bash examples/profile/workloads/geo3k/run_geo3k_rollout_ab.sh "${VARIANT}" \
  actor_rollout_ref.rollout.enforce_eager=False \
  ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS:-32}"
