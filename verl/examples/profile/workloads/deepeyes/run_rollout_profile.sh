#!/usr/bin/env bash
# DeepEyes visual_toolbox_v2 profile rollout (verl_vision + SGLang).
#
# Workload: V* / chart-style visual QA with image_zoom_in_tool (multi-turn crop).
#
# Prereqs:
#   bash verl_vision/examples/profile/workloads/deepeyes/prepare_data.sh
#   Patched sglang_vision_profile on PYTHONPATH
#
# Usage (standard: bs64 × n4 = 256 rollouts):
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash verl_vision/examples/profile/workloads/deepeyes/run_rollout_profile.sh
#
# CPU-only dry run (no GPU rollout):
#   bash verl_vision/examples/profile/workloads/deepeyes/dry_run_checks.sh

set -xeuo pipefail

VERL_VISION_ROOT="${VERL_VISION_ROOT:-/workspace/repo/verl_vision}"
DATA_ROOT="${DATA_ROOT:-/data/deepeyes_visual_toolbox_v2}"
LOG_ROOT="${LOG_ROOT:-${VERL_VISION_ROOT}/profile_logs_deepeyes}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-1}"
NGPUS="${NGPUS:-${trainer_n_gpus_per_node:-4}}"
SUFFIX="${SUFFIX:-deepeyes_bs${TRAIN_BATCH_SIZE}_n${ROLLOUT_N}}"
AGENT_LOOP_YAML="${AGENT_LOOP_YAML:-examples/profile/workloads/deepeyes/deepeyes_agent_loop.yaml}"

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
export PYTHONPATH="${SGLANG_PROFILE_ROOT}:${PWD}${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME="${HF_HOME:-/data/huggingface_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/huggingface_cache/datasets}"
export SGLANG_LOG_INFERENCE_STEP="${SGLANG_LOG_INFERENCE_STEP:-1}"
export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"
export SGLANG_INFERENCE_LOG_DIR="${LOG_ROOT}"
export SGLANG_INFERENCE_LOG_SUFFIX="${SUFFIX}"
export PROFILE_IMAGE_DUMP_DIR="${LOG_ROOT}/image_dump_${SUFFIX}"
export PROFILE_ROLLOUT_DATA_DIR="${PROFILE_ROLLOUT_DATA_DIR:-${LOG_ROOT}/rollout_data_${SUFFIX}}"
export SGLANG_MM_CACHE_PROFILE="${SGLANG_MM_CACHE_PROFILE:-1}"

TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_ROOT}/test.parquet}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
# Profile only rolls out train_batch_size rows; skip full-parquet VLM length scan (very slow).
FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-False}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing ${TRAIN_FILE} — running prepare_data.sh ..." >&2
  bash examples/profile/workloads/deepeyes/prepare_data.sh
fi

PYTHON="${PYTHON:-python3}"
"${PYTHON}" - <<PY
import importlib.util, sys
checks = [("sglang", "sglang"), ("PIL", "PIL"), ("pyarrow", "pyarrow")]
missing = [n for _, n in checks if importlib.util.find_spec(n) is None]
if missing:
    raise SystemExit(f"Missing in {sys.executable}: {missing}")
print("deps OK")
PY

rm -rf "${PROFILE_IMAGE_DUMP_DIR}"
rm -rf "${PROFILE_ROLLOUT_DATA_DIR}"
mkdir -p "${PROFILE_IMAGE_DUMP_DIR}"
mkdir -p "${PROFILE_ROLLOUT_DATA_DIR}"

TRAIN_ROWS="$("${PYTHON}" - <<PY
import pyarrow.parquet as pq
print(pq.read_table("${TRAIN_FILE}").num_rows)
PY
)"
if (( TRAIN_ROWS < TRAIN_BATCH_SIZE )); then
  echo "train.parquet rows=${TRAIN_ROWS} < batch=${TRAIN_BATCH_SIZE}" >&2
  exit 1
fi

VAL_ROWS="$("${PYTHON}" - <<PY
import pyarrow.parquet as pq
from pathlib import Path
path = Path("${VAL_FILE}")
if not path.is_file():
    print(-1)
else:
    print(pq.read_table(path).num_rows)
PY
)"
if (( VAL_ROWS <= 0 )); then
  echo "VAL_FILE=${VAL_FILE} missing or empty (rows=${VAL_ROWS}); using TRAIN_FILE (profile does not validate)" >&2
  VAL_FILE="${TRAIN_FILE}"
fi

echo "DeepEyes visual_toolbox_v2: batch=${TRAIN_BATCH_SIZE} n=${ROLLOUT_N} yaml=${AGENT_LOOP_YAML}"
echo "LOG_ROOT=${LOG_ROOT}  SUFFIX=${SUFFIX}  TRAIN_FILE=${TRAIN_FILE}"

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
  trainer.project_name='verl_deepeyes_profile' \
  trainer.experiment_name="${SUFFIX}" \
  trainer.rollout_data_dir="${PROFILE_ROLLOUT_DATA_DIR}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.agent.default_agent_loop=deepeyes_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_YAML}" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
  data.return_multi_modal_inputs=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts="${FILTER_OVERLONG_PROMPTS}" \
  actor_rollout_ref.model.use_fused_kernels=False \
  "$@"

echo ""
echo "manifest: ${PROFILE_IMAGE_DUMP_DIR}/manifest.jsonl"
echo "rollout data: ${PROFILE_ROLLOUT_DATA_DIR}"
echo "verify:"
echo "  python3 examples/profile/workloads/deepeyes/verify_manifest.py --dump-dir ${PROFILE_IMAGE_DUMP_DIR}"
echo "positive rollout pairs (DeepEyes zoom_output @ turn1):"
echo "  python3 examples/profile/shared/analysis/run_discover_positive_rollout_pairs.py \\"
echo "    --rollout-data ${PROFILE_ROLLOUT_DATA_DIR} --image-dump-dir ${PROFILE_IMAGE_DUMP_DIR}"
echo ""
echo "similarity (needs GPU):"
echo "  LOG_ROOT=${LOG_ROOT} SUFFIX=${SUFFIX} bash examples/profile/workloads/deepeyes/run_similarity.sh"
echo ""
echo "SGLang logs: ${LOG_ROOT}/*${SUFFIX}*"
