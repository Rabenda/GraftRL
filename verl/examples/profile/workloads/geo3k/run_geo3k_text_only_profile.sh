#!/usr/bin/env bash
# Geo3K text-only profiling WITH prompt padding matched to image+text token lengths.
#
# Fair control: same Geo3K questions, no image, but total prompt tokens aligned to
# verl_sglang_generate_log from geo3k_full_bs64_n4 (via text_prompt_tokens → target).
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash examples/profile/workloads/geo3k/run_geo3k_text_only_profile.sh
#
# After run:
#   python3 examples/profile/shared/analysis/analyze_profiling_logs.py \
#     --log-dir profile_logs_geo3k_text_only \
#     --suffix geo3k_text_only_padded_bs64_n4 \
#     --report

set -xeuo pipefail

cd /workspace/repo/verl_vision
SGLANG_PROFILE_ROOT="${SGLANG_PROFILE_ROOT:-/workspace/repo/sglang_vision_profile/python}"
export PYTHONPATH="${SGLANG_PROFILE_ROOT}:${PWD}${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME="${HF_HOME:-/data/huggingface_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/huggingface_cache/datasets}"
export SGLANG_LOG_INFERENCE_STEP=1
export SGLANG_INFERENCE_LOG_DIR="${SGLANG_INFERENCE_LOG_DIR:-/workspace/repo/verl_vision/profile_logs_geo3k_text_only}"
export SGLANG_INFERENCE_LOG_SUFFIX="${SGLANG_INFERENCE_LOG_SUFFIX:-geo3k_text_only_padded_bs64_n4}"
export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"

if [[ "${SGLANG_INFERENCE_LOG_DIR}" == *"profile_logs_geo3k_full"* ]] \
   || [[ "${SGLANG_INFERENCE_LOG_SUFFIX}" == *"geo3k_full"* ]]; then
  echo "ERROR: text-only must use profile_logs_geo3k_text_only + geo3k_text_only_padded_bs64_n4" >&2
  exit 1
fi

IMAGE_GENERATE_LOG="${IMAGE_GENERATE_LOG:-/workspace/repo/verl_vision/profile_logs_geo3k_full/verl_sglang_generate_log_geo3k_full_bs64_n4.csv}"
TRAIN_FILE="${TRAIN_FILE:-/workspace/repo/verl_vision/data/geo3k_text_only_padded/train.parquet}"
VAL_FILE="${VAL_FILE:-/workspace/repo/verl_vision/data/geo3k_text_only_padded/test.parquet}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-1}"
NGPUS="${trainer_n_gpus_per_node:-4}"

echo "Step 0: build padded text-only parquet (matched to ${IMAGE_GENERATE_LOG})"
python3 examples/profile/data_preprocess/geo3k/geo3k_text_only.py \
  --match-generate-log "${IMAGE_GENERATE_LOG}" \
  --out-dir "$(dirname "${TRAIN_FILE}")"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing ${TRAIN_FILE}" >&2
  exit 1
fi

TRAIN_ROWS="$(python3 - <<PY
import pyarrow.parquet as pq
print(pq.read_table("${TRAIN_FILE}").num_rows)
PY
)"
if (( TRAIN_ROWS < TRAIN_BATCH_SIZE )); then
  echo "train.parquet has ${TRAIN_ROWS} rows < train_batch_size=${TRAIN_BATCH_SIZE}" >&2
  exit 1
fi

echo "Config (text-only padded): batch=${TRAIN_BATCH_SIZE} rollout.n=${ROLLOUT_N} gpus=${NGPUS}"
echo "  train=${TRAIN_FILE}"
echo "  logs=${SGLANG_INFERENCE_LOG_DIR} suffix=${SGLANG_INFERENCE_LOG_SUFFIX}"

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
echo "Done. Generate report:"
echo "  python3 examples/profile/shared/analysis/analyze_profiling_logs.py \\"
echo "    --log-dir ${SGLANG_INFERENCE_LOG_DIR} --suffix ${SGLANG_INFERENCE_LOG_SUFFIX} --report"
