#!/usr/bin/env bash
# MMDU rollout-only验收 for sparse-decode P0/P0.5 (2 GPUs).
#
# Keeps TRAIN_BATCH_SIZE=64 × ROLLOUT_N=4. Only NGPUS is reduced because the user
# explicitly requested 2 cards.
#
# Modes (default: baseline + sparse-decode A/B only):
#   baseline | prefill_decode | decode_ab | all(=decode_ab) | prefill
#
# Usage (from graftrl/verl):
#   bash examples/profile/workloads/mmdu/launch_mmdu_decode_opt_2gpu_rolloutonly.sh
#   bash examples/profile/workloads/mmdu/launch_mmdu_decode_opt_2gpu_rolloutonly.sh decode_ab
#   bash examples/profile/workloads/mmdu/launch_mmdu_decode_opt_2gpu_rolloutonly.sh prefill_decode
#   CUDA_VISIBLE_DEVICES=6,7 bash .../launch_mmdu_decode_opt_2gpu_rolloutonly.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${VERL_ROOT}"

if [[ -x /data/conda_envs/verl_vision/bin/python3 ]]; then
  export PATH="/data/conda_envs/verl_vision/bin:${PATH}"
fi

MODE="${1:-decode_ab}"
case "${MODE}" in
  baseline|prefill|prefill_decode|decode_ab|all) ;;
  *)
    echo "Usage: $0 {baseline|prefill_decode|decode_ab|prefill|all}" >&2
    echo "  decode_ab (default) = baseline + prefill_decode (sparse)" >&2
    exit 1
    ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export NGPUS="${NGPUS:-2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export TOTAL_STEPS="${TOTAL_STEPS:-2}"
export MMDU_RUNTIME_TURNS="${MMDU_RUNTIME_TURNS:-4}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
# 2 GPUs carry the same 64×4 load as 4 GPUs; keep KV pool from collapsing at init.
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
export FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-False}"
export AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-8}"

export VERL_PROFILE_ROLLOUT_ONLY=1
export VERL_PROFILE_ROLLOUT_ONLY_STEPS="${VERL_PROFILE_ROLLOUT_ONLY_STEPS:-2}"

export DATA_ROOT="${DATA_ROOT:-${VERL_ROOT}/data/mmdu_benchmark_8192}"
export LOG_ROOT="${LOG_ROOT:-${VERL_ROOT}/profile_logs_mmdu_2gpu_decode_opt}"

export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/r4g}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-180}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
mkdir -p "${RAY_TMPDIR}" "${LOG_ROOT}"

TAG="${TAG:-2gpu_ro_s${VERL_PROFILE_ROLLOUT_ONLY_STEPS}}"

run_one() {
  local arm="$1"
  local suffix gmem_tag
  gmem_tag="$(python3 -c "print(f'{float(\"${GPU_MEMORY_UTILIZATION}\")*100:.0f}')")"

  # Reset CacheBlend env between arms.
  unset SGLANG_VLM_CACHEBLEND \
        SGLANG_VLM_CACHEBLEND_SELECT \
        SGLANG_VLM_CACHEBLEND_SELECTOR \
        SGLANG_VLM_CACHEBLEND_TARGET_TURNS \
        SGLANG_VLM_CACHEBLEND_TARGET_IMAGE_SLOTS \
        SGLANG_VLM_CACHEBLEND_FAST_APPLY \
        SGLANG_VLM_CACHEBLEND_COMPACT_PREFILL \
        SGLANG_VLM_CACHEBLEND_SPARSE_DECODE \
        SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROP_RATIO \
        SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROPPED_TOKENS || true

  case "${arm}" in
    baseline)
      # Dense-decode control: keep the same prefill reuse used by the sparse arm,
      # otherwise the A/B would conflate prefill and decode effects.
      export SGLANG_VLM_CACHEBLEND=1
      export SGLANG_VLM_CACHEBLEND_SELECT=kvdev
      export SGLANG_VLM_CACHEBLEND_TARGET_TURNS=all
      export SGLANG_VLM_CACHEBLEND_TARGET_IMAGE_SLOTS=all
      export SGLANG_VLM_CACHEBLEND_FAST_APPLY=0
      export SGLANG_VLM_CACHEBLEND_COMPACT_PREFILL=0
      export SGLANG_VLM_CACHEBLEND_SPARSE_DECODE=0
      suffix="mmdu8192_decodeopt_densecontrol_cb_kvdev_allslots_fa0_cp0_sd0_bs${TRAIN_BATCH_SIZE}_n${ROLLOUT_N}_t${MMDU_RUNTIME_TURNS}_${TAG}_gmem${gmem_tag}"
      ;;
    prefill)
      export SGLANG_VLM_CACHEBLEND=1
      export SGLANG_VLM_CACHEBLEND_SELECT=kvdev
      export SGLANG_VLM_CACHEBLEND_TARGET_TURNS=all
      export SGLANG_VLM_CACHEBLEND_TARGET_IMAGE_SLOTS=all
      export SGLANG_VLM_CACHEBLEND_FAST_APPLY=1
      export SGLANG_VLM_CACHEBLEND_COMPACT_PREFILL=1
      export SGLANG_VLM_CACHEBLEND_SPARSE_DECODE=0
      suffix="mmdu8192_decodeopt_prefill_cb_kvdev_allslots_fa1_cp1_bs${TRAIN_BATCH_SIZE}_n${ROLLOUT_N}_t${MMDU_RUNTIME_TURNS}_${TAG}_gmem${gmem_tag}"
      ;;
    prefill_decode)
      export SGLANG_VLM_CACHEBLEND=1
      export SGLANG_VLM_CACHEBLEND_SELECT=kvdev
      export SGLANG_VLM_CACHEBLEND_TARGET_TURNS=all
      export SGLANG_VLM_CACHEBLEND_TARGET_IMAGE_SLOTS=all
      # Decode A/B must not silently enable the two default-off prefill
      # experiments; otherwise their overhead is charged to sparse decoding.
      export SGLANG_VLM_CACHEBLEND_FAST_APPLY=0
      export SGLANG_VLM_CACHEBLEND_COMPACT_PREFILL=0
      export SGLANG_VLM_CACHEBLEND_SPARSE_DECODE=1
      suffix="mmdu8192_decodeopt_prefill_decode_cb_kvdev_allslots_fa0_cp0_sd1_bs${TRAIN_BATCH_SIZE}_n${ROLLOUT_N}_t${MMDU_RUNTIME_TURNS}_${TAG}_gmem${gmem_tag}"
      ;;
  esac

  export SUFFIX="${suffix}"
  local launch_log="${LOG_ROOT}/launch_${arm}_${TAG}.log"
  echo ""
  echo "============================================================"
  echo "[decode-opt] ARM=${arm}"
  echo "[decode-opt] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NGPUS=${NGPUS}"
  echo "[decode-opt] batch=${TRAIN_BATCH_SIZE}×${ROLLOUT_N} turns=${MMDU_RUNTIME_TURNS} gmem=${GPU_MEMORY_UTILIZATION}"
  echo "[decode-opt] rollout_only=${VERL_PROFILE_ROLLOUT_ONLY} steps=${VERL_PROFILE_ROLLOUT_ONLY_STEPS}"
  echo "[decode-opt] SUFFIX=${SUFFIX}"
  echo "[decode-opt] log=${launch_log}"
  echo "============================================================"
  nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=index,memory.free --format=csv,noheader || true

  bash examples/profile/workloads/mmdu/run_mmdu_profile.sh 2>&1 | tee "${launch_log}"

  echo "[decode-opt] analyzing ${SUFFIX}"
  python3 examples/profile/shared/analysis/analyze_profiling_logs.py \
    --log-dir "${LOG_ROOT}" --suffix "${SUFFIX}" --report || true
}

ARMS=()
case "${MODE}" in
  decode_ab|all)
    # Sparse decode needs CacheBlend plans to know what image KV to drop, so arm2 is
    # prefill reuse + sparse decode (not sparse-alone).
    ARMS=(baseline prefill_decode)
    ;;
  *)
    ARMS=("${MODE}")
    ;;
esac

for arm in "${ARMS[@]}"; do
  run_one "${arm}"
done

echo ""
echo "[decode-opt] done. Reports under ${LOG_ROOT}"
echo "Key metric for sparse decode: DECODE mean ms / sparse decode forward mean vs baseline."
echo "Previous 4gpu reference: profile_logs_mmdu_0145_rolloutonly_s2/"
