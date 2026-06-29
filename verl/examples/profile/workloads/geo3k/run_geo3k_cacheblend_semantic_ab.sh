#!/usr/bin/env bash
# Semantic gate for Geo3K VLM-CacheBlend selectors.
#
# Runs CacheBlend off, kvdev, and cosine selector in sequence, then compares
# rollout semantics and timing from logs. Defaults use the minimum useful RL
# rollout profile size: TRAIN_BATCH_SIZE=64 and ROLLOUT_N=4. Smaller batches are
# only for debugging initialization or routing failures, not for reporting
# experiment progress.
#
# Usage:
#   bash examples/profile/workloads/geo3k/run_geo3k_cacheblend_semantic_ab.sh exact
#   CUDA_VISIBLE_DEVICES=4,5,6,7 SELECTORS="off kvdev cos" TRAIN_BATCH_SIZE=64 ROLLOUT_N=4 \
#     bash examples/profile/workloads/geo3k/run_geo3k_cacheblend_semantic_ab.sh diversified

set -euo pipefail

VARIANT="${1:-exact}"
if [[ "${VARIANT}" != "exact" && "${VARIANT}" != "diversified" && "${VARIANT}" != "stress" ]]; then
  echo "Usage: $0 [exact|diversified|stress]" >&2
  exit 1
fi
shift || true
EXTRA_OVERRIDES=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${VERL_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NGPUS="${NGPUS:-4}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export TOTAL_STEPS="${TOTAL_STEPS:-1}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.25}"
export CLEAN_PROFILE_LOGS="${CLEAN_PROFILE_LOGS:-1}"
export VERL_PROFILE_ROLLOUT_ONLY="${VERL_PROFILE_ROLLOUT_ONLY:-1}"
export GEO3K_REFOCUS_FIRST_TURN_MAX_NEW_TOKENS="${GEO3K_REFOCUS_FIRST_TURN_MAX_NEW_TOKENS:-16}"
export GEO3K_REFOCUS_FINAL_TURN_MAX_NEW_TOKENS="${GEO3K_REFOCUS_FINAL_TURN_MAX_NEW_TOKENS:-64}"
# Long-trajectory workloads can narrow this to "1" or "1,3" to reduce per-turn warmup barriers.
export SGLANG_VLM_CACHEBLEND_TARGET_TURNS="${SGLANG_VLM_CACHEBLEND_TARGET_TURNS:-all}"
export SGLANG_VLM_CACHEBLEND_TARGET_TURN="${SGLANG_VLM_CACHEBLEND_TARGET_TURN:-1}"
export SGLANG_VLM_CACHEBLEND_TARGET_IMAGE_SLOT="${SGLANG_VLM_CACHEBLEND_TARGET_IMAGE_SLOT:--1}"
export SGLANG_VLM_CACHEBLEND_POS_MODE="${SGLANG_VLM_CACHEBLEND_POS_MODE:-same}"
export SGLANG_VLM_CACHEBLEND_MAX_GROUPS="${SGLANG_VLM_CACHEBLEND_MAX_GROUPS:-16}"
export SGLANG_VLM_CACHEBLEND_DONOR_TO_CPU="${SGLANG_VLM_CACHEBLEND_DONOR_TO_CPU:-0}"
export SGLANG_VLM_CACHEBLEND_VERBOSE="${SGLANG_VLM_CACHEBLEND_VERBOSE:-0}"
export SGLANG_VLM_CACHEBLEND_RECOMPUTE_RATIO="${SGLANG_VLM_CACHEBLEND_RECOMPUTE_RATIO:-0.15}"
export SGLANG_VLM_CACHEBLEND_SIM_THRESHOLD="${SGLANG_VLM_CACHEBLEND_SIM_THRESHOLD:-0.90}"

if [[ "${VARIANT}" == "stress" && -z "${MAX_RESPONSE_LENGTH:-}" ]]; then
  export MAX_RESPONSE_LENGTH=2048
fi

SELECTORS="${SELECTORS:-off kvdev cos}"
RUN_TAG="${RUN_TAG:-semantic_2g}"
REPORT_DIR="${REPORT_DIR:-${VERL_ROOT}/profile_logs_geo3k_cacheblend_semantic_ab}"
mkdir -p "${REPORT_DIR}"

declare -A LOG_DIRS
declare -A SUFFIXES

run_selector() {
  local selector="$1"
  local enabled="1"
  local select_mode="${selector}"
  if [[ "${selector}" == "off" ]]; then
    enabled="0"
    select_mode="kvdev"
  elif [[ "${selector}" == "cos" ]]; then
    select_mode="sim"
  fi

  local suffix="geo3k_refocus_${VARIANT}_${RUN_TAG}_${selector}_bs${TRAIN_BATCH_SIZE}_n${ROLLOUT_N}"
  local log_root="${VERL_ROOT}/profile_logs_geo3k_refocus_${VARIANT}_${RUN_TAG}_${selector}"
  LOG_DIRS["${selector}"]="${log_root}"
  SUFFIXES["${selector}"]="${suffix}"

  echo "[semantic-ab] selector=${selector} enabled=${enabled} select_mode=${select_mode}"
  echo "[semantic-ab] log_root=${log_root} suffix=${suffix} gpus=${CUDA_VISIBLE_DEVICES}"

  SGLANG_VLM_CACHEBLEND="${enabled}" \
  SGLANG_VLM_CACHEBLEND_SELECT="${select_mode}" \
  LOG_ROOT="${log_root}" \
  SUFFIX="${suffix}" \
  PROFILE_ROLLOUT_DATA_DIR="${log_root}/rollout_data_${suffix}" \
  bash examples/profile/workloads/geo3k/run_geo3k_refocus_profile.sh "${VARIANT}" \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.forward_only=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-8192}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-8192}" \
    actor_rollout_ref.rollout.temperature=0 \
    actor_rollout_ref.rollout.top_p=1 \
    actor_rollout_ref.rollout.top_k=-1 \
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=${CACHEBLEND_CHUNKED_PREFILL_SIZE:--1}" \
    "${EXTRA_OVERRIDES[@]}"
}

for selector in ${SELECTORS}; do
  run_selector "${selector}"
done

if [[ -z "${LOG_DIRS[off]:-}" ]]; then
  echo "SELECTORS must include off so semantic gate has a baseline." >&2
  exit 1
fi

candidate_args=()
for selector in ${SELECTORS}; do
  if [[ "${selector}" == "off" ]]; then
    continue
  fi
  candidate_args+=(
    --candidate "${selector}:${LOG_DIRS[${selector}]}:${SUFFIXES[${selector}]}"
  )
done

report_json="${REPORT_DIR}/geo3k_refocus_${VARIANT}_${RUN_TAG}_semantic_gate.json"
gate_args=(
  --baseline-log-dir "${LOG_DIRS[off]}"
  --baseline-suffix "${SUFFIXES[off]}"
  "${candidate_args[@]}"
  --min-common "${SEMANTIC_MIN_COMMON:-1}"
  --max-correct-to-wrong "${SEMANTIC_MAX_CORRECT_TO_WRONG:-0}"
  --max-answer-correct-to-wrong "${SEMANTIC_MAX_ANSWER_CORRECT_TO_WRONG:-0}"
  --max-score-drop "${SEMANTIC_MAX_SCORE_DROP:-0.0}"
  --max-answer-changed-rate "${SEMANTIC_MAX_ANSWER_CHANGED_RATE:-0.0}"
  --write-json "${report_json}"
)
if [[ "${FAIL_ON_THRESHOLD:-1}" == "1" ]]; then
  gate_args+=(--fail-on-threshold)
fi

python3 examples/profile/shared/analysis/semantic_cacheblend_gate.py \
  "${gate_args[@]}"

echo "[semantic-ab] report=${report_json}"
