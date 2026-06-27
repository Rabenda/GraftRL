#!/usr/bin/env bash
# Controlled vision stress test: Geo3K same text, 1/2/4 images.
#
# Prerequisites:
#   - Patched sglang (sglang_vision_profile) on PYTHONPATH with model_forward timestamp log
#   - SGLANG_LOG_INFERENCE_STEP=1 (set below) enables vision/model_forward + generate timing logs
#
# Usage (2 GPUs):
#   export CUDA_VISIBLE_DEVICES=1,4
#   bash examples/profile/workloads/geo3k/run_geo3k_vision_stress.sh 1img
#   bash examples/profile/workloads/geo3k/run_geo3k_vision_stress.sh 2img
#   bash examples/profile/workloads/geo3k/run_geo3k_vision_stress.sh 4img
#   bash examples/profile/workloads/geo3k/run_geo3k_vision_stress.sh all

set -xeuo pipefail

VARIANT=${1:-1img}
cd /workspace/repo/verl_vision

# Patched sglang for vision/model_forward profiling logs (prepend so it overrides pip install).
SGLANG_PROFILE_ROOT="${SGLANG_PROFILE_ROOT:-/workspace/repo/sglang_vision_profile/python}"
export PYTHONPATH="${SGLANG_PROFILE_ROOT}:${PWD}${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME="${HF_HOME:-/data/huggingface_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/huggingface_cache/datasets}"
export SGLANG_LOG_INFERENCE_STEP=1
export SGLANG_INFERENCE_LOG_DIR="${SGLANG_INFERENCE_LOG_DIR:-/workspace/repo/verl_vision/profile_logs_stress_1bs4rolloutn}"
# PyTorch 2.9.1 + CuDNN < 9.15 triggers a hard fail in sglang server startup.
# For profiling runs either upgrade cudnn (pip install nvidia-cudnn-cu12==9.16.0.29)
# or skip the check:
export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"

# Megatron (optional import via verl.workers.engine) JIT-compiles CUDA on import.
# Ray workers may not expose GPU arch to torch; set explicitly (H100 = 9.0).
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  TORCH_CUDA_ARCH_LIST="$(
    python3 - <<'PY'
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"{major}.{minor}")
else:
    print("9.0")
PY
  )"
  export TORCH_CUDA_ARCH_LIST
fi
export TORCH_CUDA_ARCH_LIST
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

run_one() {
  local tag=$1
  local train_file="/workspace/repo/verl_vision/data/geo3k_stress_${tag}/train.parquet"
  local val_file="/workspace/repo/verl_vision/data/geo3k_stress_${tag}/test.parquet"
  export SGLANG_INFERENCE_LOG_SUFFIX="stress_${tag}"

  local train_batch_size="${TRAIN_BATCH_SIZE:-4}"
  local rollout_n="${ROLLOUT_N:-1}"
  local ppo_mini="${PPO_MINI_BATCH_SIZE:-2}"
  local ngpus="${trainer_n_gpus_per_node:-2}"

  if [[ ! -f "${train_file}" ]]; then
    echo "Missing ${train_file}. Run geo3k_multimage_stress.py first." >&2
    exit 1
  fi

  local train_rows
  train_rows="$(python3 - <<PY
import pyarrow.parquet as pq
print(pq.read_table("${train_file}").num_rows)
PY
)"
  if (( train_rows < train_batch_size )); then
    echo "stress train has ${train_rows} rows < TRAIN_BATCH_SIZE=${train_batch_size}" >&2
    echo "Use TRAIN_BATCH_SIZE<=${train_rows} or expand the parquet first." >&2
    exit 1
  fi

  INFER_BACKEND=sglang \
  bash examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh \
    trainer.n_gpus_per_node="${ngpus}" \
    trainer.nnodes=1 \
    data.train_files="${train_file}" \
    data.val_files="${val_file}" \
    'trainer.logger=["console"]' \
    trainer.total_training_steps=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    data.train_batch_size="${train_batch_size}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini}" \
    actor_rollout_ref.rollout.n="${rollout_n}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    data.max_prompt_length=2048 \
    data.max_response_length=64 \
    actor_rollout_ref.model.use_fused_kernels=False

  echo "Done ${tag}. Logs: ${SGLANG_INFERENCE_LOG_DIR}/*stress_${tag}*"
  bash examples/profile/workloads/geo3k/export_stress_breakdown.sh "${tag}"
}

case "${VARIANT}" in
  1img) run_one 1img ;;
  2img) run_one 2img ;;
  4img) run_one 4img ;;
  all)
    run_one 1img
    run_one 2img
    run_one 4img
    ;;
  *)
    echo "Unknown variant: ${VARIANT}. Use 1img|2img|4img|all" >&2
    exit 1
    ;;
esac
