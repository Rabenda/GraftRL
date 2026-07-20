#!/usr/bin/env bash
# Geo3K rollout optimization AB (full training step, standard PPO recompute path).
#
# This script varies ONLY the SGLang / CacheBlend rollout stack (KV graft, warm
# barrier, decode-side sparse opts). Training always recomputes old_log_probs —
# no bypass / audit / logprob-sanitize shortcuts.
#
# Defaults: TRAIN_BATCH_SIZE=64, ROLLOUT_N=4 (64×4 floor).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2,3,6,7 \
#     bash examples/profile/workloads/geo3k/run_geo3k_rollout_ab.sh exact
#   CACHEBLEND_SELECTOR=kvdev CACHEBLEND_COMPACT_PREFILL=1 \
#     bash examples/profile/workloads/geo3k/run_geo3k_rollout_ab.sh exact

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
export TOTAL_STEPS="${TOTAL_STEPS:-2}"
export AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-4}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
export CLEAN_PROFILE_LOGS="${CLEAN_PROFILE_LOGS:-1}"
export VERL_PROFILE_ROLLOUT_ONLY=0
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
# This A/B targets merged visual-token work in the LLM prefill stack. Keep the
# custom similarity/per-item paths and SGLang's whole-bundle embedding cache off
# in every arm so ViT caching cannot contribute to (or regress) the result.
export SGLANG_VLM_CACHE_SIZE_MB=0
export GEO3K_REFOCUS_FIRST_TURN_MAX_NEW_TOKENS="${GEO3K_REFOCUS_FIRST_TURN_MAX_NEW_TOKENS:-16}"
export GEO3K_REFOCUS_FINAL_TURN_MAX_NEW_TOKENS="${GEO3K_REFOCUS_FINAL_TURN_MAX_NEW_TOKENS:-64}"

RUN_TAG="${RUN_TAG:-rollout_ab_4g}"
CACHEBLEND_SELECTOR="${CACHEBLEND_SELECTOR:-off}"
# Method A: which image slots to graft. "-1" = legacy last-slot (refocus only);
# "all" = every span (original + refocus); "0,-1" = explicit list.
CACHEBLEND_IMAGE_SLOTS="${CACHEBLEND_IMAGE_SLOTS:--1}"
SLOTS_TAG="$(echo "${CACHEBLEND_IMAGE_SLOTS}" | tr ',*' 'x_' | tr -cd '[:alnum:]x_-')"
case "${CACHEBLEND_IMAGE_SLOTS}" in
  -1|legacy|"") SLOTS_TAG="last" ;;
  all|"*") SLOTS_TAG="all" ;;
esac
# Method B (fast apply): memoize per-forward reuse/scatter indices.
CACHEBLEND_FAST_APPLY="${CACHEBLEND_FAST_APPLY:-0}"
FA_TAG="fa${CACHEBLEND_FAST_APPLY}"
# Experimental compact-prefill implementation. It is deliberately independent
# from sparse decoding and remains off in the standard sparse A/B launcher.
CACHEBLEND_COMPACT_PREFILL="${CACHEBLEND_COMPACT_PREFILL:-0}"
CP_TAG="cp${CACHEBLEND_COMPACT_PREFILL}"
REPORT_DIR="${REPORT_DIR:-${VERL_ROOT}/profile_logs_geo3k_rollout_ab}"
mkdir -p "${REPORT_DIR}"

case "${CACHEBLEND_SELECTOR}" in
  off)
    CACHEBLEND_ENABLED=0
    CACHEBLEND_SELECT=kvdev
    ;;
  kvdev)
    CACHEBLEND_ENABLED=1
    CACHEBLEND_SELECT=kvdev
    ;;
  cos|sim)
    CACHEBLEND_ENABLED=1
    CACHEBLEND_SELECT=sim
    ;;
  *)
    echo "CACHEBLEND_SELECTOR must be one of: off, kvdev, cos" >&2
    exit 1
    ;;
esac

suffix="geo3k_refocus_${VARIANT}_${RUN_TAG}_${CACHEBLEND_SELECTOR}_slots${SLOTS_TAG}_${FA_TAG}_${CP_TAG}_bs${TRAIN_BATCH_SIZE}_n${ROLLOUT_N}"
log_root="${VERL_ROOT}/profile_logs_geo3k_refocus_${VARIANT}_${RUN_TAG}_${CACHEBLEND_SELECTOR}_slots${SLOTS_TAG}_${FA_TAG}_${CP_TAG}"
console_log="${REPORT_DIR}/${suffix}.log"

echo "[rollout-ab] cacheblend=${CACHEBLEND_SELECTOR} image_slots=${CACHEBLEND_IMAGE_SLOTS} fast_apply=${CACHEBLEND_FAST_APPLY} compact_prefill=${CACHEBLEND_COMPACT_PREFILL} vit_cache=off"
echo "[rollout-ab] training=recompute (standard PPO old_log_prob path)"
echo "[rollout-ab] log_root=${log_root} suffix=${suffix} gpus=${CUDA_VISIBLE_DEVICES} steps=${TOTAL_STEPS}"

SGLANG_VLM_CACHEBLEND="${CACHEBLEND_ENABLED}" \
SGLANG_VLM_CACHEBLEND_SELECT="${CACHEBLEND_SELECT}" \
SGLANG_VLM_CACHEBLEND_TARGET_IMAGE_SLOTS="${CACHEBLEND_IMAGE_SLOTS}" \
SGLANG_VLM_CACHEBLEND_FAST_APPLY="${CACHEBLEND_FAST_APPLY}" \
SGLANG_VLM_CACHEBLEND_COMPACT_PREFILL="${CACHEBLEND_COMPACT_PREFILL}" \
LOG_ROOT="${log_root}" \
SUFFIX="${suffix}" \
PROFILE_ROLLOUT_DATA_DIR="${log_root}/rollout_data_${suffix}" \
bash examples/profile/workloads/geo3k/run_geo3k_refocus_profile.sh "${VARIANT}" \
  algorithm.use_kl_in_reward=False \
  data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.fsdp_config.forward_only=False \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-8192}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-8192}" \
  actor_rollout_ref.rollout.temperature=0 \
  actor_rollout_ref.rollout.top_p=1 \
  actor_rollout_ref.rollout.top_k=-1 \
  "${EXTRA_OVERRIDES[@]}" \
  2>&1 | tee "${console_log}"

echo "[rollout-ab] console_log=${console_log}"
echo "[rollout-ab] quick metrics:"
grep -E "timing_s/(step|gen|old_log_prob|update_actor|update_weights)|perf/mfu/actor_infer|actor/grad_norm" "${console_log}" || true
