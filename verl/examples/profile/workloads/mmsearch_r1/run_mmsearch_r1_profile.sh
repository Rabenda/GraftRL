#!/usr/bin/env bash
# MMSearch-R1-shaped SGLang rollout profiling.
#
# Default uses the local 5-row mini_data.pq for smoke/profile wiring checks.
# For real profiling, point TRAIN_FILE/VAL_FILE at the full parquet and raise
# TRAIN_BATCH_SIZE.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
GRAFTRL_ROOT="$(cd "${VERL_ROOT}/.." && pwd)"
MMSEARCH_ROOT="${MMSEARCH_ROOT:-/workspace/repo/multimodal-search-r1}"
cd "${VERL_ROOT}"

export PYTHONPATH="${SGLANG_PROFILE_ROOT:-${GRAFTRL_ROOT}/sglang/python}:${VERL_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/data/huggingface_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/huggingface_cache/datasets}"
export SGLANG_LOG_INFERENCE_STEP="${SGLANG_LOG_INFERENCE_STEP:-1}"
export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"
# Keep the caller's physical GPU mask intact.  Ray otherwise rewrites a mask
# such as ``CUDA_VISIBLE_DEVICES=3,4`` to ``0``/``1`` inside actors, which can
# make the workload use the machine's physical GPUs 0/1 instead.  verl maps
# each Ray resource id back to a local rank within the preserved mask.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export VERL_ROLLOUT_REUSE_GROUP_AFFINITY="${VERL_ROLLOUT_REUSE_GROUP_AFFINITY:-1}"
export VERL_ROLLOUT_REUSE_SERVER_AFFINITY="${VERL_ROLLOUT_REUSE_SERVER_AFFINITY:-1}"
export SGLANG_VLM_CACHE_SIZE_MB=0
export SGLANG_VLM_CACHEBLEND="${SGLANG_VLM_CACHEBLEND:-0}"
export SGLANG_VLM_CACHEBLEND_SPARSE_DECODE="${SGLANG_VLM_CACHEBLEND_SPARSE_DECODE:-0}"

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

TRAIN_FILE="${TRAIN_FILE:-${MMSEARCH_ROOT}/mmsearch_r1/data/mini_data.pq}"
VAL_FILE="${VAL_FILE:-${TRAIN_FILE}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-5}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-2}"
NGPUS="${NGPUS:-${trainer_n_gpus_per_node:-2}}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
LOG_ROOT="${LOG_ROOT:-${VERL_ROOT}/profile_logs_mmsearch_r1}"
SUFFIX="${SUFFIX:-mmsearch_r1_bs${TRAIN_BATCH_SIZE}_n${ROLLOUT_N}}"

export SGLANG_INFERENCE_LOG_DIR="${LOG_ROOT}"
export SGLANG_INFERENCE_LOG_SUFFIX="${SUFFIX}"
export PROFILE_ROLLOUT_DATA_DIR="${PROFILE_ROLLOUT_DATA_DIR:-${LOG_ROOT}/rollout_data_${SUFFIX}}"
export VERL_PROFILE_ROLLOUT_ONLY="${VERL_PROFILE_ROLLOUT_ONLY:-1}"
export MMSEARCH_R1_PROFILE_FORCE_TOOL="${MMSEARCH_R1_PROFILE_FORCE_TOOL:-image}"
export MMSEARCH_R1_IMAGE_SEARCH_TOPK="${MMSEARCH_R1_IMAGE_SEARCH_TOPK:-5}"
export MMSEARCH_R1_TEXT_SEARCH_TOPK="${MMSEARCH_R1_TEXT_SEARCH_TOPK:-5}"
if [[ -z "${MMSEARCH_R1_CONTEXT_TOPK:-}" ]]; then
  if (( MMSEARCH_R1_IMAGE_SEARCH_TOPK >= MMSEARCH_R1_TEXT_SEARCH_TOPK )); then
    MMSEARCH_R1_CONTEXT_TOPK="${MMSEARCH_R1_IMAGE_SEARCH_TOPK}"
  else
    MMSEARCH_R1_CONTEXT_TOPK="${MMSEARCH_R1_TEXT_SEARCH_TOPK}"
  fi
fi
export MMSEARCH_R1_CONTEXT_TOPK
export MMSEARCH_R1_ADAPT_IMAGE_PROMPT="${MMSEARCH_R1_ADAPT_IMAGE_PROMPT:-1}"
export MMSEARCH_R1_ADAPT_IMAGE_MAX_SIDE="${MMSEARCH_R1_ADAPT_IMAGE_MAX_SIDE:-448}"
MMSEARCH_DUMP_KEYS="agent_worker_pid,mmsearch_context_candidate_count,mmsearch_context_selected_count,mmsearch_context_reduction_ratio,mmsearch_exact_reuse_count,mmsearch_local_compute_count,mmsearch_skip_count,mmsearch_retrieval_cache_hits,mmsearch_selection_cache_hits,mmsearch_tokenization_cache_hits,mmsearch_context_events,mmsearch_assistant_turns"
if [[ -n "${VERL_ROLLOUT_DUMP_EXTRA_KEYS:-}" ]]; then
  export VERL_ROLLOUT_DUMP_EXTRA_KEYS="${VERL_ROLLOUT_DUMP_EXTRA_KEYS},${MMSEARCH_DUMP_KEYS}"
else
  export VERL_ROLLOUT_DUMP_EXTRA_KEYS="${MMSEARCH_DUMP_KEYS}"
fi

env_flag_enabled() {
  local value
  value="$(printf '%s' "${1:-0}" | tr '[:upper:]' '[:lower:]')"
  [[ "${value}" == "1" || "${value}" == "true" || "${value}" == "yes" || "${value}" == "on" ]]
}

has_hydra_override() {
  local key="$1"
  shift || true
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "${key}="* || "${arg}" == "+${key}="* || "${arg}" == "++${key}="* ]]; then
      return 0
    fi
  done
  return 1
}

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "TRAIN_FILE not found: ${TRAIN_FILE}" >&2
  exit 1
fi

adapt_mmsearch_parquet() {
  local src="$1"
  local label="$2"
  if [[ "${MMSEARCH_R1_ADAPT_IMAGE_PROMPT}" != "1" ]]; then
    printf '%s' "${src}"
    return 0
  fi
  local out_dir="${LOG_ROOT}/adapted_data"
  local out="${out_dir}/${label}_with_image_prompt.parquet"
  mkdir -p "${out_dir}"
  python3 - "${src}" "${out}" <<'PY'
import copy
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

src = Path(sys.argv[1])
out = Path(sys.argv[2])
df = pd.read_parquet(src)
image_max_side = int(os.environ.get("MMSEARCH_R1_ADAPT_IMAGE_MAX_SIDE", "0") or "0")


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return list(value) if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)) else []


def _content_has_image(content):
    if isinstance(content, str):
        return "<image>" in content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                return True
            if isinstance(item, str) and "<image>" in item:
                return True
            if isinstance(item, dict) and "<image>" in str(item.get("text", "")):
                return True
    return False


def _adapt_prompt(row):
    images = _as_list(row.get("images"))
    n_images = len(images)
    prompt = _as_list(row.get("prompt"))
    if n_images <= 0 or not prompt:
        return row.get("prompt")
    if any(isinstance(msg, dict) and _content_has_image(msg.get("content")) for msg in prompt):
        return row.get("prompt")

    adapted = []
    inserted = False
    for msg in prompt:
        msg = copy.deepcopy(msg)
        if not inserted and isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            prefix = "<image>" * n_images + "\n"
            msg["content"] = prefix + content if isinstance(content, str) else prefix + str(content)
            inserted = True
        adapted.append(msg)
    if not inserted:
        first = copy.deepcopy(prompt[0])
        if isinstance(first, dict):
            first["content"] = "<image>" * n_images + "\n" + str(first.get("content", ""))
            adapted[0] = first
    return np.array(adapted, dtype=object)


def _resize_image_value(value):
    if image_max_side <= 0:
        return value
    if not isinstance(value, dict) or value.get("bytes") is None:
        return value
    try:
        with Image.open(io.BytesIO(value["bytes"])) as img:
            img = img.convert("RGB")
            if max(img.size) <= image_max_side:
                return value
            img.thumbnail((image_max_side, image_max_side), Image.Resampling.BICUBIC)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
    except Exception:
        return value
    resized = dict(value)
    resized["bytes"] = buf.getvalue()
    resized["path"] = None
    return resized


def _adapt_images(value):
    images = _as_list(value)
    if not images:
        return value
    return np.array([_resize_image_value(img) for img in images], dtype=object)


df["prompt"] = df.apply(_adapt_prompt, axis=1)
if "images" in df.columns:
    df["images"] = df["images"].apply(_adapt_images)
df.to_parquet(out, index=False)
print(out)
PY
}

if [[ "${TRAIN_FILE}" == "${VAL_FILE}" ]]; then
  TRAIN_FILE="$(adapt_mmsearch_parquet "${TRAIN_FILE}" train)"
  VAL_FILE="${TRAIN_FILE}"
else
  TRAIN_FILE="$(adapt_mmsearch_parquet "${TRAIN_FILE}" train)"
  VAL_FILE="$(adapt_mmsearch_parquet "${VAL_FILE}" val)"
fi

train_rows="$(
  python3 - <<PY
import pyarrow.parquet as pq
print(pq.read_table("${TRAIN_FILE}").num_rows)
PY
)"
if (( train_rows < TRAIN_BATCH_SIZE )); then
  echo "train rows=${train_rows} < TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}" >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}" "${PROFILE_ROLLOUT_DATA_DIR}"
if [[ "${CLEAN_PROFILE_LOGS:-1}" == "1" ]]; then
  rm -f \
    "${LOG_ROOT}/model_forward_log_${SUFFIX}.csv" \
    "${LOG_ROOT}/vision_encoder_log_${SUFFIX}.csv" \
    "${LOG_ROOT}/verl_sglang_generate_log_${SUFFIX}.csv" \
    "${LOG_ROOT}/cacheblend_barrier_log_${SUFFIX}.csv"
  rm -rf "${PROFILE_ROLLOUT_DATA_DIR}"
  mkdir -p "${PROFILE_ROLLOUT_DATA_DIR}"
fi

export SGLANG_VLM_CACHEBLEND_TARGET_TURNS="${SGLANG_VLM_CACHEBLEND_TARGET_TURNS:-1}"
export SGLANG_VLM_CACHEBLEND_TARGET_IMAGE_SLOTS="${SGLANG_VLM_CACHEBLEND_TARGET_IMAGE_SLOTS:-all}"
if env_flag_enabled "${SGLANG_VLM_CACHEBLEND:-0}"; then
  export SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_WAIT_POLICY="${SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_WAIT_POLICY:-bounded}"
  export SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_MAX_WAIT_S="${SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_MAX_WAIT_S:-0.05}"
fi

chunked_prefill_args=()
chunked_prefill_key="actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size"
if env_flag_enabled "${SGLANG_VLM_CACHEBLEND:-0}" && ! has_hydra_override "${chunked_prefill_key}" "$@"; then
  chunked_prefill_args=("+${chunked_prefill_key}=${CACHEBLEND_CHUNKED_PREFILL_SIZE:--1}")
fi

echo "[mmsearch-r1] train=${TRAIN_FILE} rows=${train_rows} batch=${TRAIN_BATCH_SIZE} n=${ROLLOUT_N} steps=${TOTAL_STEPS}"
echo "[mmsearch-r1] force_tool=${MMSEARCH_R1_PROFILE_FORCE_TOOL} candidates=${MMSEARCH_R1_IMAGE_SEARCH_TOPK} selected=${MMSEARCH_R1_CONTEXT_TOPK}"
echo "[mmsearch-r1] vit_cache=off log_root=${LOG_ROOT} suffix=${SUFFIX}"

INFER_BACKEND=sglang \
bash examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh \
  trainer.n_gpus_per_node="${NGPUS}" \
  trainer.nnodes=1 \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.image_key=images \
  'trainer.logger=["console"]' \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.project_name='verl_mmsearch_r1_profile' \
  trainer.experiment_name="${SUFFIX}" \
  trainer.rollout_data_dir="${PROFILE_ROLLOUT_DATA_DIR}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.35}" \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=mmsearch_r1_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=examples/profile/workloads/mmsearch_r1/mmsearch_r1_agent_loop.yaml \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=3 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts="${FILTER_OVERLONG_PROMPTS:-False}" \
  actor_rollout_ref.model.use_fused_kernels=False \
  "${chunked_prefill_args[@]}" \
  "$@"

echo ""
python3 examples/profile/workloads/mmsearch_r1/summarize_mmsearch_run.py \
  --rollout-data "${PROFILE_ROLLOUT_DATA_DIR}" \
  --log-dir "${LOG_ROOT}" \
  --suffix "${SUFFIX}"
echo ""
echo "Profiling report:"
echo "  python3 examples/profile/shared/analysis/analyze_profiling_logs.py \\"
echo "    --log-dir ${LOG_ROOT} --suffix ${SUFFIX} --report"
