#!/usr/bin/env bash
# Single-turn 1-step GRPO smoke for vision encoding profiling: Geo3K vs Refocus_Chart.
#
# Prereqs:
#   1. python examples/profile/data_preprocess/chart/refocus_chart_singleturn.py
#   2. Patched sglang with vision_encoder_log / model_forward_log (sglang_vision_profile)
#   3. export SGLANG_LOG_INFERENCE_STEP=1
#      export SGLANG_INFERENCE_LOG_DIR=/workspace/repo/verl_vision/profile_logs
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash examples/profile/workloads/geo3k/compare_geo3k_refocus_chart_smoke.sh geo3k
#   bash examples/profile/workloads/geo3k/compare_geo3k_refocus_chart_smoke.sh chartqa
#   bash examples/profile/workloads/geo3k/compare_geo3k_refocus_chart_smoke.sh dummy_crop

set -xeuo pipefail

cd /workspace/repo/verl_vision
# Patched SGLang with vision_encoder_log / inference_step_log hooks.
SGLANG_PROFILE_ROOT="${SGLANG_PROFILE_ROOT:-/workspace/repo/sglang_vision_profile/python}"
export PYTHONPATH="${SGLANG_PROFILE_ROOT}:${PWD}${PYTHONPATH:+:$PYTHONPATH}"

DATASET=${1:-geo3k}
SUFFIX=${SGLANG_INFERENCE_LOG_SUFFIX:-qwen25vl_${DATASET}_smoke}

case "${DATASET}" in
  geo3k)
    TRAIN_FILE=/workspace/repo/verl_vision/data/geo3k/train.parquet
    VAL_FILE=/workspace/repo/verl_vision/data/geo3k/test.parquet
    ;;
  chartqa|refocus|refocus_chart)
    TRAIN_FILE=/workspace/repo/verl_vision/data/refocus_chart/train.parquet
    VAL_FILE=/workspace/repo/verl_vision/data/refocus_chart/test.parquet
    ;;
  dummy_crop|refocus_dummy_crop)
    DUMMY_CROP_DIR=${DUMMY_CROP_DIR:-/workspace/repo/verl_vision/data/refocus_chart_dummy_crop}
    TRAIN_FILE="${DUMMY_CROP_DIR}/train.parquet"
    VAL_FILE="${DUMMY_CROP_DIR}/test.parquet"
    if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
      python3 examples/profile/data_preprocess/chart/refocus_chart_dummy_crop.py \
        --src_dir "${REFOCUS_CHART_DIR:-/workspace/repo/verl_vision/data/refocus_chart}" \
        --local_save_dir "${DUMMY_CROP_DIR}" \
        --crop_ratio "${DUMMY_CROP_RATIO:-0.5}" \
        --max_train_rows "${DUMMY_CROP_MAX_TRAIN_ROWS:-64}" \
        --max_test_rows "${DUMMY_CROP_MAX_TEST_ROWS:-16}"
    fi
    ;;
  *)
    echo "Unknown dataset: ${DATASET}. Use geo3k, chartqa, or dummy_crop." >&2
    exit 1
    ;;
esac

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
  actor_rollout_ref.model.use_fused_kernels=False

echo "Logs:"
echo "  ${SGLANG_INFERENCE_LOG_DIR}/verl_sglang_generate_log_${SUFFIX}.csv"
echo "  ${SGLANG_INFERENCE_LOG_DIR}/vision_encoder_log_${SUFFIX}.csv"
echo "  ${SGLANG_INFERENCE_LOG_DIR}/model_forward_log_${SUFFIX}.csv"
