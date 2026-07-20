#!/usr/bin/env bash
# One-line stress workload for VLM-CacheBlend LLM prefill reuse.
#
# Dataset shape:
#   Refocus Chart with image_1 and bbox metadata scaled by 2x. Oracle refocus then
#   produces a large turn1 refocus image. With TRAIN_BATCH_SIZE=1 and ROLLOUT_N=8,
#   one GRPO group should create one donor and several recipients for the same
#   agent_uid/turn/image_slot.
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash examples/profile/workloads/chart/run_cacheblend_stress.sh on
#   bash examples/profile/workloads/chart/run_cacheblend_stress.sh off
#   bash examples/profile/workloads/chart/run_cacheblend_stress.sh both

set -euo pipefail

MODE="${1:-on}"
if [[ "${MODE}" != "on" && "${MODE}" != "off" && "${MODE}" != "both" ]]; then
  echo "Usage: $0 [on|off|both]" >&2
  exit 1
fi

VERL_VISION_ROOT="${VERL_VISION_ROOT:-/workspace/repo/verl_vision}"
SRC_DATA_ROOT="${SRC_DATA_ROOT:-/data/refocus_chart_multiturn_oracle_changed}"
DATA_ROOT="${DATA_ROOT:-/data/refocus_chart_cacheblend_stress_s2}"
SCALE="${CACHEBLEND_STRESS_SCALE:-2.0}"

cd "${VERL_VISION_ROOT}"

if [[ ! -f "${SRC_DATA_ROOT}/train.parquet" || ! -f "${SRC_DATA_ROOT}/test.parquet" ]]; then
  echo "[stress] source Refocus Chart missing at ${SRC_DATA_ROOT}; preparing a small source split"
  DATA_ROOT="${SRC_DATA_ROOT}" \
  REFOCUS_MAX_TRAIN_ROWS="${REFOCUS_MAX_TRAIN_ROWS:-64}" \
  REFOCUS_MAX_TEST_ROWS="${REFOCUS_MAX_TEST_ROWS:-8}" \
    bash examples/profile/workloads/chart/prepare_data.sh
fi

if [[ ! -f "${DATA_ROOT}/train.parquet" || ! -f "${DATA_ROOT}/test.parquet" || "${FORCE_STRESS_DATA:-0}" == "1" ]]; then
  echo "[stress] building CacheBlend stress data -> ${DATA_ROOT} scale=${SCALE}"
  python3 examples/profile/data_preprocess/chart/refocus_chart_cacheblend_stress.py \
    --src_dir "${SRC_DATA_ROOT}" \
    --local_save_dir "${DATA_ROOT}" \
    --scale "${SCALE}" \
    --max_train_rows "${CACHEBLEND_STRESS_TRAIN_ROWS:-32}" \
    --max_test_rows "${CACHEBLEND_STRESS_TEST_ROWS:-8}"
fi

run_one() {
  local mode="$1"
  local enabled="0"
  if [[ "${mode}" == "on" ]]; then
    enabled="1"
  fi

  export TRAIN_FILE="${DATA_ROOT}/train.parquet"
  export VAL_FILE="${DATA_ROOT}/test.parquet"
  export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
  export ROLLOUT_N="${ROLLOUT_N:-8}"
  export TOTAL_STEPS="${TOTAL_STEPS:-1}"
  export NGPUS="${NGPUS:-4}"
  export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-12288}"
  export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
  export LOG_ROOT="${LOG_ROOT:-${VERL_VISION_ROOT}/profile_logs_cacheblend_stress_s2_${mode}_4g_n${ROLLOUT_N}}"
  export SUFFIX="${SUFFIX:-cacheblend_stress_s2_${mode}_4g_n${ROLLOUT_N}}"

  # Keep this workload focused on LLM-side visual KV reuse.
  export VTOOL_MODEL_REFOCUS=0
  export VTOOL_ORACLE_REFOCUS=1
  export VTOOL_ORACLE_DIVERSIFY=0
  export VTOOL_ORACLE_FIRST_TURN_MAX_NEW_TOKENS="${VTOOL_ORACLE_FIRST_TURN_MAX_NEW_TOKENS:-32}"
  export VTOOL_ORACLE_FINAL_TURN_MAX_NEW_TOKENS="${VTOOL_ORACLE_FINAL_TURN_MAX_NEW_TOKENS:-32}"
  export VERL_PROFILE_ROLLOUT_ONLY="${VERL_PROFILE_ROLLOUT_ONLY:-1}"

  export SGLANG_VLM_CACHEBLEND="${enabled}"
  export SGLANG_VLM_CACHEBLEND_TARGET_TURN="${SGLANG_VLM_CACHEBLEND_TARGET_TURN:-1}"
  export SGLANG_VLM_CACHEBLEND_TARGET_IMAGE_SLOT="${SGLANG_VLM_CACHEBLEND_TARGET_IMAGE_SLOT:--1}"
  export SGLANG_VLM_CACHEBLEND_SELECT="${SGLANG_VLM_CACHEBLEND_SELECT:-kvdev}"
  export SGLANG_VLM_CACHEBLEND_RECOMPUTE_RATIO="${SGLANG_VLM_CACHEBLEND_RECOMPUTE_RATIO:-0.15}"
  export SGLANG_VLM_CACHEBLEND_POS_MODE="${SGLANG_VLM_CACHEBLEND_POS_MODE:-same}"
  export SGLANG_VLM_CACHEBLEND_MAX_GROUPS="${SGLANG_VLM_CACHEBLEND_MAX_GROUPS:-8}"
  export SGLANG_VLM_CACHEBLEND_DONOR_TO_CPU="${SGLANG_VLM_CACHEBLEND_DONOR_TO_CPU:-0}"
  export SGLANG_VLM_CACHEBLEND_VERBOSE="${SGLANG_VLM_CACHEBLEND_VERBOSE:-0}"

  local -a EXTRA_OVERRIDES=(
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.35}"
  )
  # [VLM-CacheBlend] Single-chunk prefill so the donor captures the FULL refocus
  # image span. The turn1 prompt is ~10.5k tokens (chart + refocus image, each
  # ~5244 tokens). With chunked prefill on (sglang auto-picks 2k-8k), the refocus
  # span starts well past the chunk boundary and is split across forwards, so the
  # donor records only a fragment (e.g. 2918 of 5244) and every recipient falls
  # back with image_token_count_mismatch -> cacheblend_used=0. Note: verl's
  # rollout.enable_chunked_prefill is NOT wired into the sglang path; the only
  # effective lever is engine_kwargs.sglang.chunked_prefill_size (-1 == disable).
  if [[ "${CACHEBLEND_DISABLE_CHUNKED_PREFILL:-1}" == "1" ]]; then
    # `+` prefix is required: engine_kwargs.sglang is an empty dict in the config
    # and hydra runs in struct mode, so the key must be force-added (matches the
    # repo idiom, e.g. tests/special_npu/run_qwen3_8b_grpo_mindspeedllm.sh).
    EXTRA_OVERRIDES+=(
      "+actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=${CACHEBLEND_CHUNKED_PREFILL_SIZE:--1}"
    )
  fi

  echo "[stress] mode=${mode} enabled=${enabled} train=${TRAIN_FILE} batch=${TRAIN_BATCH_SIZE} n=${ROLLOUT_N}"
  echo "[stress] overrides: ${EXTRA_OVERRIDES[*]}"
  bash examples/profile/workloads/chart/run_rollout_profile.sh "${EXTRA_OVERRIDES[@]}"

  # [VLM-CacheBlend] Fail-fast: on an `on` run, refuse to proceed (or to be trusted
  # as a benefit measurement) unless donor KV was actually reused. used=0 means the
  # `on` path only paid overhead -> any timing delta is noise, not a CacheBlend result.
  if [[ "${enabled}" == "1" && "${CACHEBLEND_ASSERT_USED:-1}" == "1" ]]; then
    python3 examples/profile/shared/analysis/assert_cacheblend_used.py \
      --log-root "${LOG_ROOT}" --suffix "${SUFFIX}"
  fi
}

case "${MODE}" in
  on|off)
    run_one "${MODE}"
    ;;
  both)
    LOG_ROOT="${VERL_VISION_ROOT}/profile_logs_cacheblend_stress_s2_off_4g_n${ROLLOUT_N:-8}" \
    SUFFIX="cacheblend_stress_s2_off_4g_n${ROLLOUT_N:-8}" \
      run_one off
    LOG_ROOT="${VERL_VISION_ROOT}/profile_logs_cacheblend_stress_s2_on_4g_n${ROLLOUT_N:-8}" \
    SUFFIX="cacheblend_stress_s2_on_4g_n${ROLLOUT_N:-8}" \
      run_one on
    ;;
esac
