#!/usr/bin/env bash
# Geo3K refocus-style multiturn profiling.
#
# Variants:
#   exact       deterministic full-canvas refocus image; best-case reuse.
#   diversified branch-dependent style around same ROI; similar-image reuse.
#   stress      upscaled image; stronger E+P signal.
#   all         run all three variants in sequence.
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash examples/profile/workloads/geo3k/run_geo3k_refocus_profile.sh exact
#   bash examples/profile/workloads/geo3k/run_geo3k_refocus_profile.sh all

set -euo pipefail

VARIANT="${1:-exact}"
if [[ "${VARIANT}" != "exact" && "${VARIANT}" != "diversified" && "${VARIANT}" != "stress" && "${VARIANT}" != "all" ]]; then
  echo "Usage: $0 [exact|diversified|stress|all]" >&2
  exit 1
fi

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

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-2}"
NGPUS="${NGPUS:-${trainer_n_gpus_per_node:-4}}"
DATA_PARENT="${DATA_PARENT:-${VERL_ROOT}/data}"

prepare_if_missing() {
  local variant="$1"
  local data_dir="${DATA_PARENT}/geo3k_refocus_${variant}"
  if [[ ! -f "${data_dir}/train.parquet" || ! -f "${data_dir}/test.parquet" || "${FORCE_GEO3K_REFOCUS_DATA:-0}" == "1" ]]; then
    echo "[geo3k-refocus] preparing ${variant} data under ${DATA_PARENT}"
    OUT_DIR="${DATA_PARENT}" VARIANT="${variant}" bash examples/profile/workloads/geo3k/prepare_refocus_data.sh
  fi
}

run_one() {
  local variant="$1"
  shift || true
  prepare_if_missing "${variant}"

  local data_dir="${DATA_PARENT}/geo3k_refocus_${variant}"
  local train_file="${TRAIN_FILE:-${data_dir}/train.parquet}"
  local val_file="${VAL_FILE:-${data_dir}/test.parquet}"
  local suffix="${SUFFIX:-geo3k_refocus_${variant}_bs${TRAIN_BATCH_SIZE}_n${ROLLOUT_N}}"
  local log_root="${LOG_ROOT:-${VERL_ROOT}/profile_logs_geo3k_refocus_${variant}}"

  local mode="exact"
  if [[ "${variant}" == "diversified" ]]; then
    mode="diversified"
  fi
  local max_response_length="${MAX_RESPONSE_LENGTH:-1024}"
  if [[ "${variant}" == "stress" && -z "${MAX_RESPONSE_LENGTH:-}" ]]; then
    max_response_length=4096
  fi

  export SGLANG_INFERENCE_LOG_DIR="${log_root}"
  export SGLANG_INFERENCE_LOG_SUFFIX="${suffix}"
  export PROFILE_IMAGE_DUMP_DIR="${log_root}/image_dump_${suffix}"
  export PROFILE_ROLLOUT_DATA_DIR="${PROFILE_ROLLOUT_DATA_DIR:-${log_root}/rollout_data_${suffix}}"
  export GEO3K_REFOCUS_MODE="${GEO3K_REFOCUS_MODE:-${mode}}"
  export GEO3K_REFOCUS_FIRST_TURN_MAX_NEW_TOKENS="${GEO3K_REFOCUS_FIRST_TURN_MAX_NEW_TOKENS:-16}"
  export GEO3K_REFOCUS_FINAL_TURN_MAX_NEW_TOKENS="${GEO3K_REFOCUS_FINAL_TURN_MAX_NEW_TOKENS:-64}"
  export VERL_PROFILE_ROLLOUT_ONLY="${VERL_PROFILE_ROLLOUT_ONLY:-1}"

  mkdir -p "${log_root}" "${PROFILE_IMAGE_DUMP_DIR}" "${PROFILE_ROLLOUT_DATA_DIR}"
  if [[ "${CLEAN_PROFILE_LOGS:-1}" == "1" ]]; then
    rm -f \
      "${log_root}/model_forward_log_${suffix}.csv" \
      "${log_root}/vision_encoder_log_${suffix}.csv" \
      "${log_root}/verl_sglang_generate_log_${suffix}.csv"
    rm -rf "${PROFILE_IMAGE_DUMP_DIR}" "${PROFILE_ROLLOUT_DATA_DIR}"
    mkdir -p "${PROFILE_IMAGE_DUMP_DIR}" "${PROFILE_ROLLOUT_DATA_DIR}"
  fi

  local train_rows
  train_rows="$(python3 - <<PY
import pyarrow.parquet as pq
print(pq.read_table("${train_file}").num_rows)
PY
)"
  if (( train_rows < TRAIN_BATCH_SIZE )); then
    echo "train.parquet rows=${train_rows} < TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}" >&2
    exit 1
  fi

  echo "[geo3k-refocus] variant=${variant} mode=${GEO3K_REFOCUS_MODE} batch=${TRAIN_BATCH_SIZE} n=${ROLLOUT_N} steps=${TOTAL_STEPS}"
  echo "[geo3k-refocus] train=${train_file} log_root=${log_root} suffix=${suffix}"

  INFER_BACKEND=sglang \
  bash examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh \
    trainer.n_gpus_per_node="${NGPUS}" \
    trainer.nnodes=1 \
    data.train_files="${train_file}" \
    data.val_files="${val_file}" \
    'trainer.logger=["console"]' \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.project_name='verl_geo3k_refocus_profile' \
    trainer.experiment_name="${suffix}" \
    trainer.rollout_data_dir="${PROFILE_ROLLOUT_DATA_DIR}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.25}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=geo3k_refocus_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=examples/profile/workloads/geo3k/geo3k_refocus_agent_loop.yaml \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-8192}" \
    data.max_response_length="${max_response_length}" \
    actor_rollout_ref.rollout.response_length="${max_response_length}" \
    data.filter_overlong_prompts="${FILTER_OVERLONG_PROMPTS:-False}" \
    actor_rollout_ref.model.use_fused_kernels=False \
    "$@"

  echo ""
  echo "Profiling report:"
  echo "  python3 examples/profile/shared/analysis/analyze_profiling_logs.py \\"
  echo "    --log-dir ${log_root} --suffix ${suffix} --report"
  echo "Images: ${PROFILE_IMAGE_DUMP_DIR}"
}

case "${VARIANT}" in
  exact|diversified|stress)
    shift || true
    run_one "${VARIANT}" "$@"
    ;;
  all)
    shift || true
    for v in exact diversified stress; do
      TRAIN_FILE="" VAL_FILE="" SUFFIX="" LOG_ROOT="" PROFILE_ROLLOUT_DATA_DIR="" run_one "${v}" "$@"
    done
    ;;
esac
