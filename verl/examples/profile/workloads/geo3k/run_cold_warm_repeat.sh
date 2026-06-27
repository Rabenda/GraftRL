#!/usr/bin/env bash
# Repeat the same 4 geo3k stress samples for 3 steps to observe cold vs warm vision/e2e.
#
# Expectation:
#   step1: more cold path (~10-18% vision/e2e on first encode per replica)
#   step2/3: if same images repeat, more batched_only / lower vision%
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=1,4
#   bash examples/profile/workloads/geo3k/run_cold_warm_repeat.sh

set -xeuo pipefail

cd /workspace/repo/verl_vision
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/data/huggingface_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/huggingface_cache/datasets}"
export SGLANG_LOG_INFERENCE_STEP=1
export SGLANG_INFERENCE_LOG_DIR="${SGLANG_INFERENCE_LOG_DIR:-/workspace/repo/verl_vision/profile_logs_coldwarm}"
export SGLANG_INFERENCE_LOG_SUFFIX=coldwarm_repeat_3step

TRAIN_FILE=/workspace/repo/verl_vision/data/geo3k_stress_1img/train.parquet
VAL_FILE=/workspace/repo/verl_vision/data/geo3k_stress_1img/test.parquet

INFER_BACKEND=sglang \
bash examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  'trainer.logger=["console"]' \
  trainer.total_training_steps=3 \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  data.train_batch_size=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  data.max_prompt_length=2048 \
  data.max_response_length=16 \
  actor_rollout_ref.model.use_fused_kernels=False

echo "Done. Analyze with:"
echo "  python examples/profile/shared/analysis/analyze_vision_cold_warm.py \\"
echo "    --label coldwarm_3step \\"
echo "    --generate-log ${SGLANG_INFERENCE_LOG_DIR}/verl_sglang_generate_log_${SGLANG_INFERENCE_LOG_SUFFIX}.csv \\"
echo "    --vision-log ${SGLANG_INFERENCE_LOG_DIR}/vision_encoder_log_${SGLANG_INFERENCE_LOG_SUFFIX}.csv"
