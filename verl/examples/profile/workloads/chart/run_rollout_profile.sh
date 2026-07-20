#!/usr/bin/env bash
# VTool-R1 / Refocus Chart Split profile rollout (verl_vision + SGLang).
#
# Workload: bar-chart QA with original image + tool-generated refocus image (2-turn).
# Same agent loop as examples/profile/vtool_agent_loop.py; organized like sokoban/.
#
# Prereqs:
#   bash verl_vision/examples/profile/workloads/chart/prepare_data.sh
#   Patched sglang_vision_profile on PYTHONPATH
#
# Usage (standard: bs64 × n4 = 256 rollouts):
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash verl_vision/examples/profile/workloads/chart/run_rollout_profile.sh
#
# Diversified oracle (cross-branch turn1 experiment):
#   VTOOL_ORACLE_DIVERSIFY=1 bash .../run_rollout_profile.sh
#
# Model refocus (recommended entry — logs → profile_logs_vtool_chart_model_refocus/):
#   bash verl_vision/examples/profile/workloads/chart/run_model_refocus_profile.sh
#
# Or manually:
#   VTOOL_MODEL_REFOCUS=1 TRAIN_FILE=/data/refocus_chart_multiturn_oracle_changed/train.parquet \
#     bash .../run_rollout_profile.sh

set -xeuo pipefail

VERL_VISION_ROOT="${VERL_VISION_ROOT:-/workspace/repo/verl_vision}"
DATA_ROOT="${DATA_ROOT:-/data/refocus_chart_multiturn}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-1}"
NGPUS="${NGPUS:-${trainer_n_gpus_per_node:-4}}"

VTOOL_MODEL_REFOCUS="${VTOOL_MODEL_REFOCUS:-0}"
VTOOL_ORACLE_DIVERSIFY="${VTOOL_ORACLE_DIVERSIFY:-0}"
if [[ "${VTOOL_MODEL_REFOCUS}" == "1" && "${VTOOL_ORACLE_DIVERSIFY}" == "1" ]]; then
  echo "Set only one of VTOOL_MODEL_REFOCUS or VTOOL_ORACLE_DIVERSIFY" >&2
  exit 1
fi
if [[ "${VTOOL_MODEL_REFOCUS}" == "1" ]]; then
  _DIV_TAG="_model_refocus"
  AGENT_LOOP_YAML="${AGENT_LOOP_YAML:-examples/profile/workloads/chart/agent_loop_model_refocus.yaml}"
elif [[ "${VTOOL_ORACLE_DIVERSIFY}" == "1" ]]; then
  _DIV_TAG="_diversified"
  AGENT_LOOP_YAML="${AGENT_LOOP_YAML:-examples/profile/workloads/chart/agent_loop_diversified.yaml}"
else
  _DIV_TAG=""
  AGENT_LOOP_YAML="${AGENT_LOOP_YAML:-examples/profile/workloads/chart/agent_loop.yaml}"
fi

SUFFIX="${SUFFIX:-vtool_chart_bs${TRAIN_BATCH_SIZE}_n${ROLLOUT_N}${_DIV_TAG}}"

TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/train.parquet}"

# Separate log roots per experiment line (override anytime with LOG_ROOT=...)
if [[ -z "${LOG_ROOT:-}" ]]; then
  if [[ "${VTOOL_MODEL_REFOCUS}" == "1" ]]; then
    LOG_ROOT="${VERL_VISION_ROOT}/profile_logs_vtool_chart_model_refocus"
  elif [[ "${VTOOL_ORACLE_DIVERSIFY}" == "1" ]]; then
    LOG_ROOT="${VERL_VISION_ROOT}/profile_logs_vtool_chart_diversified"
  elif [[ "${TRAIN_FILE}" == *oracle_changed* ]]; then
    LOG_ROOT="${VERL_VISION_ROOT}/profile_logs_vtool_chart_clean"
  else
    LOG_ROOT="${VERL_VISION_ROOT}/profile_logs_vtool_chart_raw"
  fi
fi

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
export SGLANG_VLM_CACHE_SIZE_MB=0
export VTOOL_ORACLE_DIVERSIFY
export VTOOL_ORACLE_REFOCUS="${VTOOL_ORACLE_REFOCUS:-$([[ "${VTOOL_MODEL_REFOCUS}" == "1" ]] && echo 0 || echo 1)}"

# Profile never runs validation, but main_ppo still constructs val_dataset (must be non-empty).
VAL_FILE="${VAL_FILE:-${TRAIN_FILE}}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
# Profile only rolls out train_batch_size rows; skip full-parquet VLM length scan (very slow).
FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-False}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing ${TRAIN_FILE} — running prepare_data.sh ..." >&2
  bash examples/profile/workloads/chart/prepare_data.sh
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

echo "VTool Chart (Refocus split): batch=${TRAIN_BATCH_SIZE} n=${ROLLOUT_N} oracle=${VTOOL_ORACLE_REFOCUS} model_refocus=${VTOOL_MODEL_REFOCUS} diversify=${VTOOL_ORACLE_DIVERSIFY} yaml=${AGENT_LOOP_YAML}"
echo "LOG_ROOT=${LOG_ROOT}  SUFFIX=${SUFFIX}  TRAIN_FILE=${TRAIN_FILE}  VAL_FILE=${VAL_FILE}"

mkdir -p "${LOG_ROOT}"
if [[ "${CLEAN_PROFILE_LOGS:-1}" == "1" ]]; then
  rm -f \
    "${LOG_ROOT}/model_forward_log_${SUFFIX}.csv" \
    "${LOG_ROOT}/vision_encoder_log_${SUFFIX}.csv" \
    "${LOG_ROOT}/verl_sglang_generate_log_${SUFFIX}.csv"
fi

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
  trainer.project_name='verl_vtool_chart_profile' \
  trainer.experiment_name="${SUFFIX}" \
  trainer.rollout_data_dir="${PROFILE_ROLLOUT_DATA_DIR}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.agent.default_agent_loop=vtool_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_YAML}" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=3 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
  data.return_multi_modal_inputs=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-2048}" \
  data.filter_overlong_prompts="${FILTER_OVERLONG_PROMPTS}" \
  actor_rollout_ref.model.use_fused_kernels=False \
  "$@"

echo ""
echo "manifest: ${PROFILE_IMAGE_DUMP_DIR}/manifest.jsonl"
echo "rollout data: ${PROFILE_ROLLOUT_DATA_DIR}"
echo "verify:"
echo "  python3 examples/profile/workloads/chart/verify_manifest.py --dump-dir ${PROFILE_IMAGE_DUMP_DIR}"
echo "  python3 examples/profile/shared/analysis/check_rollout_dump_sanity.py \\"
echo "    --rollout-data ${PROFILE_ROLLOUT_DATA_DIR} --verify-refocus-score"
echo "positive rollout pairs:"
echo "  python3 examples/profile/shared/analysis/run_discover_positive_rollout_pairs.py \\"
echo "    --rollout-data ${PROFILE_ROLLOUT_DATA_DIR} --image-dump-dir ${PROFILE_IMAGE_DUMP_DIR} \\"
echo "    --use-diversified-oracle --verify-refocus-score"
echo ""
echo "similarity + case study:"
echo "  LOG_ROOT=${LOG_ROOT} SUFFIX=${SUFFIX} bash examples/profile/workloads/chart/run_similarity.sh"
echo ""
echo "SGLang logs: ${LOG_ROOT}/*${SUFFIX}*"
echo "Walkthrough:"
echo "  python3 examples/profile/shared/reports/generate_multiturn_request_flow_report.py \\"
echo "    --log-dir ${LOG_ROOT} --suffix ${SUFFIX} \\"
echo "    --image-dump-dir ${PROFILE_IMAGE_DUMP_DIR} --list-requests"
