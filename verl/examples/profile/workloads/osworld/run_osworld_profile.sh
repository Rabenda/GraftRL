#!/usr/bin/env bash
# OSWorld / ARPO-style GUI offline-replay profiling (rollout-heavy).
#
# Uses dumped or synthetic trajectories (instruction + ordered screenshots).
# The ``osworld_gui_agent`` loop snowballs: model generates Thought+Action each
# turn, then consumes the next screenshot — long decode × many turns so rollout
# dominates. Default VERL_PROFILE_ROLLOUT_ONLY=1.
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0,1,4,7
#   bash examples/profile/workloads/osworld/run_osworld_profile.sh
#
# Real ARPO dumps:
#   RESULTS_ROOT=/path/to/results DATA_ROOT=data/osworld_gui_real \
#     bash examples/profile/workloads/osworld/prepare_data.sh
#   DATA_ROOT=data/osworld_gui_real bash examples/profile/workloads/osworld/run_osworld_profile.sh
#
# CacheBlend:
#   SGLANG_VLM_CACHEBLEND=1 SGLANG_VLM_CACHEBLEND_TARGET_TURNS=all \
#     bash examples/profile/workloads/osworld/run_osworld_profile.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
GRAFTRL_ROOT="$(cd "${VERL_ROOT}/.." && pwd)"
cd "${VERL_ROOT}"

export PYTHONPATH="${SGLANG_PROFILE_ROOT:-${GRAFTRL_ROOT}/sglang/python}:${VERL_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/data/huggingface_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/huggingface_cache/datasets}"
export SGLANG_LOG_INFERENCE_STEP="${SGLANG_LOG_INFERENCE_STEP:-1}"
export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"

CACHE_ROOT="${CACHE_ROOT:-/data/cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/inductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${CACHE_ROOT}/torch_extensions}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${CACHE_ROOT}/nv}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${CACHE_ROOT}/flashinfer}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
mkdir -p "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}" "${CUDA_CACHE_PATH}" "${FLASHINFER_WORKSPACE_BASE}" "${TORCH_HOME}"

env_flag_enabled() {
  local value
  value="$(printf '%s' "${1:-0}" | tr '[:upper:]' '[:lower:]')"
  [[ "${value}" == "1" || "${value}" == "true" || "${value}" == "yes" || "${value}" == "on" ]]
}

DATA_ROOT="${DATA_ROOT:-${VERL_ROOT}/data/osworld_gui_tasksynth}"
if [[ ! -f "${DATA_ROOT}/train.parquet" || ! -f "${DATA_ROOT}/test.parquet" || "${FORCE_OSWORLD_DATA:-0}" == "1" ]]; then
  DATA_ROOT="${DATA_ROOT}" FORCE="${FORCE_OSWORLD_DATA:-0}" \
    bash examples/profile/workloads/osworld/prepare_data.sh
fi

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-1}"
NGPUS="${NGPUS:-${trainer_n_gpus_per_node:-4}}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-8}"
OSWORLD_RUNTIME_TURNS="${OSWORLD_RUNTIME_TURNS:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-32768}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.25}"
LOGPROB_MAX_TOKEN_LEN_PER_GPU="${LOGPROB_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 512))}"
USE_FUSED_KERNELS="${USE_FUSED_KERNELS:-False}"

export VERL_PROFILE_ROLLOUT_ONLY="${VERL_PROFILE_ROLLOUT_ONLY:-1}"
export OSWORLD_RUNTIME_TURNS
export OSWORLD_INTERMEDIATE_MAX_NEW_TOKENS="${OSWORLD_INTERMEDIATE_MAX_NEW_TOKENS:-512}"
export OSWORLD_FINAL_MAX_NEW_TOKENS="${OSWORLD_FINAL_MAX_NEW_TOKENS:-1024}"

if env_flag_enabled "${SGLANG_VLM_CACHEBLEND:-0}"; then
  export SGLANG_VLM_CACHEBLEND_TARGET_TURNS="${SGLANG_VLM_CACHEBLEND_TARGET_TURNS:-all}"
  export SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_WAIT_POLICY="${SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_WAIT_POLICY:-bounded}"
  export SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_MAX_WAIT_S="${SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_MAX_WAIT_S:-10}"
fi

TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_ROOT}/test.parquet}"
SUFFIX="${SUFFIX:-osworld_gui_bs${TRAIN_BATCH_SIZE}_n${ROLLOUT_N}_t${OSWORLD_RUNTIME_TURNS}}"
LOG_ROOT="${LOG_ROOT:-${VERL_ROOT}/profile_logs_osworld_gui}"
PROFILE_ROLLOUT_DATA_DIR="${PROFILE_ROLLOUT_DATA_DIR:-${LOG_ROOT}/rollout_data_${SUFFIX}}"

export SGLANG_INFERENCE_LOG_DIR="${LOG_ROOT}"
export SGLANG_INFERENCE_LOG_SUFFIX="${SUFFIX}"
mkdir -p "${LOG_ROOT}" "${PROFILE_ROLLOUT_DATA_DIR}"

train_rows="$(python3 - <<PY
import pyarrow.parquet as pq
print(pq.read_table("${TRAIN_FILE}").num_rows)
PY
)"
if (( train_rows < TRAIN_BATCH_SIZE )); then
  echo "train.parquet rows=${train_rows} < TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}" >&2
  exit 1
fi

echo "[osworld] train=${TRAIN_FILE}"
echo "[osworld] batch=${TRAIN_BATCH_SIZE} n=${ROLLOUT_N} steps=${TOTAL_STEPS} turns=${OSWORLD_RUNTIME_TURNS}"
echo "[osworld] max_prompt=${MAX_PROMPT_LENGTH} max_response=${MAX_RESPONSE_LENGTH} rollout_only=${VERL_PROFILE_ROLLOUT_ONLY}"

max_agent_turns=$((OSWORLD_RUNTIME_TURNS + 2))

INFER_BACKEND=sglang \
bash examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh \
  trainer.n_gpus_per_node="${NGPUS}" \
  trainer.nnodes=1 \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  'trainer.logger=["console"]' \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.project_name='verl_osworld_gui_profile' \
  trainer.experiment_name="${SUFFIX}" \
  trainer.rollout_data_dir="${PROFILE_ROLLOUT_DATA_DIR}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=osworld_gui_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=examples/profile/workloads/osworld/osworld_gui_agent_loop.yaml \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="${max_agent_turns}" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${max_agent_turns}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${LOGPROB_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOGPROB_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${LOGPROB_MAX_TOKEN_LEN_PER_GPU}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts="${FILTER_OVERLONG_PROMPTS:-False}" \
  actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
  actor_rollout_ref.model.use_fused_kernels="${USE_FUSED_KERNELS}" \
  "$@"

echo ""
echo "Profiling report:"
echo "  python3 examples/profile/shared/analysis/analyze_profiling_logs.py \\"
echo "    --log-dir ${LOG_ROOT} --suffix ${SUFFIX} --report"
echo "Rollout dump: ${PROFILE_ROLLOUT_DATA_DIR}"
