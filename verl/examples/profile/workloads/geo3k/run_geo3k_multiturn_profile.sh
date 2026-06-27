#!/usr/bin/env bash
# Geo3K two-turn dummy-crop profiling: Image_0 -> forced center crop -> Image_1 -> answer.
#
# Geo3K has no refocus bbox metadata; this uses dummy_crop_agent (deterministic crop) instead of
# model-produced refocus tools. Refocus_Chart is only needed for real tool-call refocus training.
#
# Saves per-turn images when PROFILE_IMAGE_DUMP_DIR is set; run analyze_image_similarity.py after.
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash examples/profile/workloads/geo3k/run_geo3k_multiturn_profile.sh
#
# Smoke (4 samples, 16 rollouts):
#   TRAIN_FILE=data/geo3k_stress_1img/train.parquet TRAIN_BATCH_SIZE=4 ROLLOUT_N=4 \\
#     bash examples/profile/workloads/geo3k/run_geo3k_multiturn_profile.sh

set -xeuo pipefail

cd /workspace/repo/verl_vision
SGLANG_PROFILE_ROOT="${SGLANG_PROFILE_ROOT:-/workspace/repo/sglang_vision_profile/python}"
export PYTHONPATH="${SGLANG_PROFILE_ROOT}:${PWD}${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME="${HF_HOME:-/data/huggingface_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/huggingface_cache/datasets}"
export SGLANG_LOG_INFERENCE_STEP=1
export SGLANG_INFERENCE_LOG_DIR="${SGLANG_INFERENCE_LOG_DIR:-/workspace/repo/verl_vision/profile_logs_geo3k_multiturn}"
export SGLANG_INFERENCE_LOG_SUFFIX="${SGLANG_INFERENCE_LOG_SUFFIX:-geo3k_multiturn_bs64_n4}"
export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"
export PROFILE_IMAGE_DUMP_DIR="${PROFILE_IMAGE_DUMP_DIR:-${SGLANG_INFERENCE_LOG_DIR}/image_dump_${SGLANG_INFERENCE_LOG_SUFFIX}}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-1}"
NGPUS="${trainer_n_gpus_per_node:-4}"

TRAIN_FILE="${TRAIN_FILE:-/workspace/repo/verl_vision/data/geo3k/train.parquet}"
VAL_FILE="${VAL_FILE:-/workspace/repo/verl_vision/data/geo3k/test.parquet}"

rm -rf "${PROFILE_IMAGE_DUMP_DIR}"
mkdir -p "${PROFILE_IMAGE_DUMP_DIR}"

TRAIN_ROWS="$(python3 - <<PY
import pyarrow.parquet as pq
print(pq.read_table("${TRAIN_FILE}").num_rows)
PY
)"
if (( TRAIN_ROWS < TRAIN_BATCH_SIZE )); then
  echo "train.parquet has ${TRAIN_ROWS} rows < train_batch_size=${TRAIN_BATCH_SIZE}" >&2
  exit 1
fi

echo "Multiturn crop profile: batch=${TRAIN_BATCH_SIZE} n=${ROLLOUT_N} image_dump=${PROFILE_IMAGE_DUMP_DIR}"

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
  actor_rollout_ref.rollout.agent.default_agent_loop=dummy_crop_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=examples/profile/archive/dummy_crop/dummy_crop_agent_loop.yaml \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  data.max_prompt_length=2048 \
  data.max_response_length=256 \
  actor_rollout_ref.model.use_fused_kernels=False

echo ""
echo "Logs: ${SGLANG_INFERENCE_LOG_DIR}/*${SGLANG_INFERENCE_LOG_SUFFIX}*"
echo "Images: ${PROFILE_IMAGE_DUMP_DIR}"
echo ""
echo "Post-run (pick one request: original + crop + prompt):"
echo "  python3 examples/profile/shared/reports/generate_multiturn_request_flow_report.py \\"
echo "    --log-dir ${SGLANG_INFERENCE_LOG_DIR} --suffix ${SGLANG_INFERENCE_LOG_SUFFIX} \\"
echo "    --image-dump-dir ${PROFILE_IMAGE_DUMP_DIR} --list-requests"
echo "  python3 examples/profile/shared/reports/generate_multiturn_request_flow_report.py \\"
echo "    --log-dir ${SGLANG_INFERENCE_LOG_DIR} --suffix ${SGLANG_INFERENCE_LOG_SUFFIX} \\"
echo "    --image-dump-dir ${PROFILE_IMAGE_DUMP_DIR} --request-id <request_id>"
echo ""
echo "Profiling report:"
echo "  python3 examples/profile/shared/analysis/analyze_profiling_logs.py \\"
echo "    --log-dir ${SGLANG_INFERENCE_LOG_DIR} --suffix ${SGLANG_INFERENCE_LOG_SUFFIX} --report"
echo ""
echo "Image similarity (GRPO groups + turn0 vs turn1):"
echo "  python3 examples/profile/shared/analysis/analyze_image_similarity.py --dump-dir ${PROFILE_IMAGE_DUMP_DIR}"
