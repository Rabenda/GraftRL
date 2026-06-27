#!/usr/bin/env bash
# Full Geo3K single-turn profiling: train_batch_size=64, rollout.n=4, 1 image per sample.
#
# Uses the full geo3k train parquet (2101 rows). One training step = 64 prompts x 4 rollouts = 256 generations.
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash examples/profile/workloads/geo3k/run_geo3k_full_profile.sh
#
# Optional:
#   TRAIN_BATCH_SIZE=64 ROLLOUT_N=4 TOTAL_STEPS=1 bash examples/profile/workloads/geo3k/run_geo3k_full_profile.sh

set -xeuo pipefail

cd /workspace/repo/verl_vision
SGLANG_PROFILE_ROOT="${SGLANG_PROFILE_ROOT:-/workspace/repo/sglang_vision_profile/python}"
export PYTHONPATH="${SGLANG_PROFILE_ROOT}:${PWD}${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME="${HF_HOME:-/data/huggingface_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/huggingface_cache/datasets}"
export SGLANG_LOG_INFERENCE_STEP=1
export SGLANG_INFERENCE_LOG_DIR="${SGLANG_INFERENCE_LOG_DIR:-/workspace/repo/verl_vision/profile_logs_geo3k_full}"
export SGLANG_INFERENCE_LOG_SUFFIX="${SGLANG_INFERENCE_LOG_SUFFIX:-geo3k_full_bs64_n4}"
export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"

# Keep all runtime JIT/compile caches off the workspace (HOME defaults to /workspace).
# These dirs are written on every run (Triton kernels, torch extensions, flashinfer
# JIT, CUDA compute cache, matplotlib/hf/pip). Redirect them to /data.
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
# Default to 2 steps so analysis can drop the cold-start step (step 0): the first
# forward on each replica pays one-time FSDP onload + allocator + autotune cost
# (~6-7s) that otherwise dominates the EXTEND/prefill bucket. See analyze script
# --drop-cold-step.
TOTAL_STEPS="${TOTAL_STEPS:-2}"
NGPUS="${trainer_n_gpus_per_node:-4}"

# Dataset lives on /data (copied out of the workspace). Override TRAIN_FILE/VAL_FILE
# to point elsewhere if needed.
TRAIN_FILE="${TRAIN_FILE:-/data/verl_data/geo3k/train.parquet}"
VAL_FILE="${VAL_FILE:-/data/verl_data/geo3k/test.parquet}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing ${TRAIN_FILE}. Run: python examples/data_preprocess/geo3k.py" >&2
  exit 1
fi

# Sanity: drop_last=True requires train rows >= batch size
TRAIN_ROWS="$(python3 - <<PY
import pyarrow.parquet as pq
print(pq.read_table("${TRAIN_FILE}").num_rows)
PY
)"
if (( TRAIN_ROWS < TRAIN_BATCH_SIZE )); then
  echo "train.parquet has ${TRAIN_ROWS} rows < train_batch_size=${TRAIN_BATCH_SIZE}" >&2
  exit 1
fi

echo "Config: batch=${TRAIN_BATCH_SIZE} rollout.n=${ROLLOUT_N} generations=$((TRAIN_BATCH_SIZE * ROLLOUT_N)) gpus=${NGPUS}"

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
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  data.max_prompt_length=2048 \
  data.max_response_length=64 \
  actor_rollout_ref.model.use_fused_kernels=False

echo ""
echo "Done. Logs:"
echo "  ${SGLANG_INFERENCE_LOG_DIR}/verl_sglang_generate_log_${SGLANG_INFERENCE_LOG_SUFFIX}.csv"
echo "  ${SGLANG_INFERENCE_LOG_DIR}/vision_encoder_log_${SGLANG_INFERENCE_LOG_SUFFIX}.csv"
echo "  ${SGLANG_INFERENCE_LOG_DIR}/model_forward_log_${SGLANG_INFERENCE_LOG_SUFFIX}.csv"
echo ""
echo "Export breakdown:"
echo "  SGLANG_INFERENCE_LOG_DIR=${SGLANG_INFERENCE_LOG_DIR} \\"
echo "  python3 examples/profile/shared/analysis/analyze_profiling_logs.py \\"
echo "    --log-dir ${SGLANG_INFERENCE_LOG_DIR} --suffix ${SGLANG_INFERENCE_LOG_SUFFIX} \\"
echo "    --export-breakdown-csv ${SGLANG_INFERENCE_LOG_DIR}/e2e_module_breakdown_${SGLANG_INFERENCE_LOG_SUFFIX}.csv"
