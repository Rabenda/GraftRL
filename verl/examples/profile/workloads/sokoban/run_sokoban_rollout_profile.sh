#!/usr/bin/env bash
# Sokoban profile rollout on verl_vision + SGLang (same stack as refocus).
#
# Uses a custom AgentLoop (sokoban_agent) — NOT verl-agent training entry.
# Reuses verl-agent's SokobanEnv implementation via PYTHONPATH for gym visuals only.
# Requires patched SGLang (sglang_vision_profile) with mem_cache/multimodal_cache.py present.
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash verl_vision/examples/profile/workloads/sokoban/run_sokoban_rollout_profile.sh
#
# 2-GPU:
#   export CUDA_VISIBLE_DEVICES=5,7
#   NGPUS=2 bash verl_vision/examples/profile/workloads/sokoban/run_sokoban_rollout_profile.sh

set -xeuo pipefail

VERL_VISION_ROOT="${VERL_VISION_ROOT:-/workspace/repo/verl_vision}"
VERL_AGENT_ROOT="${VERL_AGENT_ROOT:-/workspace/repo/verl-agent}"
DATA_ROOT="${DATA_ROOT:-/data/verl-agent_sokoban/visual}"
LOG_ROOT="${LOG_ROOT:-${VERL_VISION_ROOT}/profile_logs_sokoban}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-1}"
NGPUS="${NGPUS:-${trainer_n_gpus_per_node:-4}}"
SUFFIX="${SUFFIX:-sokoban_bs${TRAIN_BATCH_SIZE}_n${ROLLOUT_N}}"
AGENT_LOOP_YAML="${AGENT_LOOP_YAML:-examples/profile/workloads/sokoban/sokoban_agent_loop.yaml}"

export TMPDIR="${TMPDIR:-/workspace/tmp}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
mkdir -p "${TMPDIR}"

cd "${VERL_VISION_ROOT}"
SGLANG_PROFILE_ROOT="${SGLANG_PROFILE_ROOT:-/workspace/repo/sglang_vision_profile/python}"
_MM_CACHE="${SGLANG_PROFILE_ROOT}/sglang/srt/mem_cache/multimodal_cache.py"
if [[ ! -f "${_MM_CACHE}" ]]; then
  _FALLBACK="/workspace/repo/verl_sglang/verl_customize/sglang/python/sglang/srt/mem_cache/multimodal_cache.py"
  if [[ -f "${_FALLBACK}" ]]; then
    mkdir -p "$(dirname "${_MM_CACHE}")"
    cp "${_FALLBACK}" "${_MM_CACHE}"
    echo "patched missing ${_MM_CACHE} from verl_sglang"
  else
    echo "Missing ${_MM_CACHE} — SGLang multimodal rollout will fail." >&2
    exit 1
  fi
fi
export PYTHONPATH="${SGLANG_PROFILE_ROOT}:${PWD}:${VERL_AGENT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export VERL_AGENT_ROOT

export HF_HOME="${HF_HOME:-/data/huggingface_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/huggingface_cache/datasets}"
export SGLANG_LOG_INFERENCE_STEP="${SGLANG_LOG_INFERENCE_STEP:-1}"
export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"
export SGLANG_INFERENCE_LOG_DIR="${LOG_ROOT}"
export SGLANG_INFERENCE_LOG_SUFFIX="${SUFFIX}"
export PROFILE_IMAGE_DUMP_DIR="${LOG_ROOT}/image_dump_${SUFFIX}"

TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_ROOT}/test.parquet}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing ${TRAIN_FILE} — run prepare_sokoban_data.sh first." >&2
  exit 1
fi

PYTHON="${PYTHON:-python3}"
"${PYTHON}" - <<PY
import importlib.util, sys
checks = [("gym", "gym"), ("gym_sokoban", "gym_sokoban"), ("matplotlib", "matplotlib"), ("sglang", "sglang")]
missing = [n for _, n in checks if importlib.util.find_spec(n) is None]
if missing:
    raise SystemExit(f"Missing in {sys.executable}: {missing}")
print("deps OK")
PY

rm -rf "${PROFILE_IMAGE_DUMP_DIR}"
mkdir -p "${PROFILE_IMAGE_DUMP_DIR}"

TRAIN_ROWS="$("${PYTHON}" - <<PY
import pyarrow.parquet as pq
print(pq.read_table("${TRAIN_FILE}").num_rows)
PY
)"
if (( TRAIN_ROWS < TRAIN_BATCH_SIZE )); then
  echo "train.parquet rows=${TRAIN_ROWS} < batch=${TRAIN_BATCH_SIZE}" >&2
  exit 1
fi

echo "Sokoban (verl_vision AgentLoop): batch=${TRAIN_BATCH_SIZE} n=${ROLLOUT_N} yaml=${AGENT_LOOP_YAML}"

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
  trainer.project_name='verl_sokoban_profile' \
  trainer.experiment_name="${SUFFIX}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.agent.default_agent_loop=sokoban_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_YAML}" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=15 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=15 \
  data.return_multi_modal_inputs=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-16384}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-2048}" \
  actor_rollout_ref.model.use_fused_kernels=False \
  "$@"

echo ""
echo "manifest: ${PROFILE_IMAGE_DUMP_DIR}/manifest.jsonl"
echo "verify:"
echo "  python3 verl_vision/examples/profile/workloads/sokoban/verify_sokoban_manifest.py --dump-dir ${PROFILE_IMAGE_DUMP_DIR}"
