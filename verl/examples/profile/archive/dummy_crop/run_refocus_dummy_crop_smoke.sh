#!/usr/bin/env bash
# Dummy crop smoke for Refocus_Chart: Image_0 -> forced center-crop Image_1.
#
# This validates that a second image generated between turns reaches SGLang
# rollout/train. It does not validate model-produced tool-call syntax.
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash examples/profile/run_refocus_dummy_crop_smoke.sh

set -xeuo pipefail

cd /workspace/repo/verl_vision
# Patched SGLang with vision_encoder_log / inference_step_log hooks.
SGLANG_PROFILE_ROOT="${SGLANG_PROFILE_ROOT:-/workspace/repo/sglang_vision_profile/python}"
export PYTHONPATH="${SGLANG_PROFILE_ROOT}:${PWD}${PYTHONPATH:+:$PYTHONPATH}"

SUFFIX=${SGLANG_INFERENCE_LOG_SUFFIX:-qwen25vl_refocus_dummy_crop_agent_smoke}
TRAIN_FILE=${TRAIN_FILE:-/workspace/repo/verl_vision/data/refocus_chart/train.parquet}
VAL_FILE=${VAL_FILE:-/workspace/repo/verl_vision/data/refocus_chart/test.parquet}

export SGLANG_LOG_INFERENCE_STEP=${SGLANG_LOG_INFERENCE_STEP:-1}
export SGLANG_INFERENCE_LOG_DIR=${SGLANG_INFERENCE_LOG_DIR:-/workspace/repo/verl_vision/profile_logs}
export SGLANG_INFERENCE_LOG_SUFFIX="${SUFFIX}"

INFER_BACKEND=sglang \
bash examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh \
  trainer.n_gpus_per_node="${trainer_n_gpus_per_node:-4}" \
  trainer.nnodes=1 \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  'trainer.logger=["console"]' \
  trainer.total_training_steps=1 \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  data.train_batch_size=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.agent.default_agent_loop=dummy_crop_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=examples/profile/archive/dummy_crop/dummy_crop_agent_loop.yaml \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  data.max_response_length=256 \
  actor_rollout_ref.model.use_fused_kernels=False

echo "Logs:"
echo "  ${SGLANG_INFERENCE_LOG_DIR}/verl_sglang_generate_log_${SUFFIX}.csv"
echo "  ${SGLANG_INFERENCE_LOG_DIR}/inference_step_log_${SUFFIX}.csv"
echo "Optional legacy logs, if enabled by the local SGLang patch:"
echo "  ${SGLANG_INFERENCE_LOG_DIR}/vision_encoder_log_${SUFFIX}.csv"
echo "  ${SGLANG_INFERENCE_LOG_DIR}/model_forward_log_${SUFFIX}.csv"
