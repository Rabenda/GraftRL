#!/usr/bin/env bash
# One final real-scale dense/query-aware sparse decode A/B. No smoke/ABBA runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${VERL_ROOT}"

VERL_ENV_BIN="${VERL_ENV_BIN:-/data/conda_envs/verl_vision/bin}"
if [[ ! -x "${VERL_ENV_BIN}/python3" ]]; then
  VERL_ENV_BIN="/workspace/miniconda3/envs/verl_vision/bin"
fi
if [[ ! -x "${VERL_ENV_BIN}/python3" ]]; then
  echo "Cannot find the verl_vision Python environment." >&2
  exit 2
fi
export PATH="${VERL_ENV_BIN}:${PATH}"
python3 -c 'import ray, torch' >/dev/null

env_flag_enabled() {
  local value
  value="$(printf '%s' "${1:-0}" | tr '[:upper:]' '[:lower:]')"
  [[ "${value}" == "1" || "${value}" == "true" || "${value}" == "yes" || "${value}" == "on" ]]
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export NGPUS="${NGPUS:-2}"
export TRAIN_BATCH_SIZE=64
export ROLLOUT_N=4
export TOTAL_STEPS="${TOTAL_STEPS:-2}"
export VERL_PROFILE_ROLLOUT_ONLY=1
export VERL_PROFILE_ROLLOUT_ONLY_STEPS="${VERL_PROFILE_ROLLOUT_ONLY_STEPS:-2}"
export MMDU_RUNTIME_TURNS="${MMDU_RUNTIME_TURNS:-4}"
export MMDU_INTERMEDIATE_MAX_NEW_TOKENS="${MMDU_INTERMEDIATE_MAX_NEW_TOKENS:-256}"
export MMDU_FINAL_MAX_NEW_TOKENS="${MMDU_FINAL_MAX_NEW_TOKENS:-1024}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-16384}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export DATA_ROOT="${DATA_ROOT:-${VERL_ROOT}/data/mmdu_benchmark_small_filtered_16384}"
export FILTER_OVERLONG_PROMPTS=False
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
export AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-8}"
export ENFORCE_EAGER=False
export SAMPLING_TEMPERATURE=1.0
export DATA_SEED=42
# Keep stochastic temperature=1 sampling, but give the same sample/branch/turn the
# same RNG stream in both arms. This reduces scheduler-induced length drift without
# forcing greedy decoding or making dense/sparse logits artificially equal.
export MMDU_PAIRED_SAMPLING_SEED="${MMDU_PAIRED_SAMPLING_SEED:-42}"
export LOG_ROOT="${LOG_ROOT:-${VERL_ROOT}/profile_logs_mmdu_query_sparse_final}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_TMPDIR="${RAY_TMPDIR:-/data/cache/ray/mmdu_query_sparse_final}"
mkdir -p "${LOG_ROOT}" "${RAY_TMPDIR}"

# Formal measurements must start from idle physical devices.
IFS=',' read -r -a _mmdu_gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
for _mmdu_gpu_id in "${_mmdu_gpu_ids[@]}"; do
  _mmdu_gpu_state="$(/usr/bin/nvidia-smi \
    --id="${_mmdu_gpu_id}" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | head -1)"
  IFS=',' read -r _mmdu_gpu_mem _mmdu_gpu_util <<< "${_mmdu_gpu_state}"
  _mmdu_gpu_mem="${_mmdu_gpu_mem//[[:space:]]/}"
  _mmdu_gpu_util="${_mmdu_gpu_util//[[:space:]]/}"
  if (( _mmdu_gpu_mem > 2048 || _mmdu_gpu_util > 10 )); then
    echo "Refusing formal run: physical GPU ${_mmdu_gpu_id} is busy " \
      "(${_mmdu_gpu_mem} MiB, ${_mmdu_gpu_util}% util)." >&2
    exit 2
  fi
done
unset _mmdu_gpu_ids _mmdu_gpu_id _mmdu_gpu_state _mmdu_gpu_mem _mmdu_gpu_util

unset SGLANG_VLM_CACHEBLEND \
      SGLANG_VLM_CACHEBLEND_SPARSE_DECODE \
      SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MODE \
      SGLANG_VLM_CACHEBLEND_FAST_APPLY \
      SGLANG_VLM_CACHEBLEND_COMPACT_PREFILL || true

# Minimal public policy surface. The selector/kernel details use tested code defaults;
# both arms receive the same values and only the enable bit differs.
export SGLANG_ROLLOUT_SPARSE_DECODE_MODE=query_blocks
export SGLANG_ROLLOUT_SPARSE_DECODE_MAX_DROP_RATIO="${SPARSE_MAX_DROP_RATIO:-0.70}"
export SGLANG_ROLLOUT_SPARSE_DECODE_MAX_DROPPED_SCORE_MASS="${SPARSE_MAX_DROPPED_SCORE_MASS:-0.05}"
export SGLANG_ROLLOUT_SPARSE_DECODE_MIN_CONTEXT_TOKENS="${SPARSE_MIN_CONTEXT_TOKENS:-4096}"
export SGLANG_ROLLOUT_SPARSE_DECODE_MIN_DROPPED_TOKENS="${SPARSE_MIN_DROPPED_TOKENS:-4096}"

TAG="${TAG:-bs64n4_s${VERL_PROFILE_ROLLOUT_ONLY_STEPS}}"
# A sparse-only recovery/optimization run may reuse a completed dense arm.  This
# avoids spending another 64x4 baseline after a startup bug or execution-only change.
DENSE_REFERENCE_TAG="${DENSE_REFERENCE_TAG:-${TAG}}"
DENSE_SUFFIX="mmdu_query_sparse_dense_${DENSE_REFERENCE_TAG}"
SPARSE_SUFFIX="mmdu_query_sparse_skip_${TAG}"
DENSE_LOG="${LOG_ROOT}/launch_${DENSE_SUFFIX}.log"
SPARSE_LOG="${LOG_ROOT}/launch_${SPARSE_SUFFIX}.log"
SPARSE_FORWARD_CSV="${LOG_ROOT}/model_forward_log_${SPARSE_SUFFIX}.csv"
RUN_DENSE="${RUN_DENSE:-1}"
RUN_SPARSE="${RUN_SPARSE:-1}"

_mmdu_outputs=()
if env_flag_enabled "${RUN_DENSE}"; then
  _mmdu_outputs+=("${DENSE_LOG}")
fi
if env_flag_enabled "${RUN_SPARSE}"; then
  _mmdu_outputs+=("${SPARSE_LOG}" "${SPARSE_FORWARD_CSV}")
fi
if (( ${#_mmdu_outputs[@]} == 0 )); then
  echo "At least one of RUN_DENSE or RUN_SPARSE must be enabled." >&2
  exit 2
fi
for _mmdu_output in "${_mmdu_outputs[@]}"; do
  if [[ -e "${_mmdu_output}" ]]; then
    echo "Refusing to mix a formal pair with existing output: ${_mmdu_output}" >&2
    echo "Set a new TAG and rerun; existing measurements were not modified." >&2
    exit 2
  fi
done
unset _mmdu_output _mmdu_outputs

run_arm() {
  local arm="$1"
  local suffix="$2"
  local launch_log="$3"
  if [[ "${arm}" == "sparse" ]]; then
    export SGLANG_ROLLOUT_SPARSE_DECODE=1
  else
    export SGLANG_ROLLOUT_SPARSE_DECODE=0
  fi
  export SUFFIX="${suffix}"
  echo "[mmdu-query-sparse] arm=${arm} suffix=${suffix} batch=64x4 graph=on"
  bash examples/profile/workloads/mmdu/run_mmdu_profile.sh 2>&1 | tee "${launch_log}"
}

# Exactly one paired A/B: dense then sparse. Recovery may resume only the missing arm
# after an initialization failure; a completed arm is never rerun just to rebuild it.
if env_flag_enabled "${RUN_DENSE}"; then
  run_arm dense "${DENSE_SUFFIX}" "${DENSE_LOG}"
fi
if env_flag_enabled "${RUN_SPARSE}"; then
  run_arm sparse "${SPARSE_SUFFIX}" "${SPARSE_LOG}"
fi

python3 examples/profile/workloads/mmdu/summarize_mmdu_sparse_pair.py \
  --dense-log "${DENSE_LOG}" \
  --sparse-log "${SPARSE_LOG}" \
  --sparse-forward-csv "${SPARSE_FORWARD_CSV}" \
  --min-rollout-speedup "${MIN_ROLLOUT_SPEEDUP:-0.05}" \
  --min-token-throughput-speedup "${MIN_TOKEN_THROUGHPUT_SPEEDUP:-0.0}" \
  --max-reward-drop "${MAX_REWARD_DROP:-0.01}" \
  --max-response-length-drift "${MAX_RESPONSE_LENGTH_DRIFT:-0.05}"
