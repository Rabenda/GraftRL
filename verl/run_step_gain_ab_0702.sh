#!/usr/bin/env bash
# Whole-step rollout-gain AB: baseline (off) vs best rollout stack (kvdev + warm barrier).
#
# Design principle (read before tuning):
#   We optimize rollout E+P (ViT encode + LLM prefill / KV graft), NOT decode length.
#   Do NOT inflate max_response_length to make rollout "look heavier" — that would not
#   validate CacheBlend / barrier benefits on Geo3K refocus.
#
#   Goal: fix ABNORMAL training inflation (clean_0702b ~387s/step) so gen vs training
#   ratio is sane, while keeping the native Geo3K workload:
#     - max_response_length=1024 (short geometry answers)
#     - multi-turn refocus images (natural E+P in rollout)
#     - param/optimizer offload ON (same as prior formal_log runs with SGLang)
#
#   Both arms share identical training + workload config. ONLY the rollout stack differs.
#
# ARM A: baseline (off) full step
# ARM B: best rollout (kvdev + warm barrier) full step
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate verl_vision

# ---- shared cluster / scale (both arms identical) ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NGPUS=4
export TRAIN_BATCH_SIZE=64
export ROLLOUT_N=4
export TOTAL_STEPS="${TOTAL_STEPS:-2}"          # step 1 warmup; analyze step 2
export AGENT_NUM_WORKERS=4

# ---- training config: standard PPO recompute; do NOT game rollout ratio via decode ----
# Same ballpark as prior formal_log runs with param/optimizer offload ON.
# Do not shorten response length; do not lengthen it either — inherited from refocus profile.
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.3}"
export ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-8192}"
export ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU="${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-8192}"

export RUN_TAG="${RUN_TAG:-stepgain_0702}"

# Ray on /dev/shm: clean_0702b ran on a 98%-full disk (/tmp/r4g) with object spilling;
# that likely inflated update_actor/update_weights, not a fair training baseline.
export RAY_TMPDIR="${RAY_TMPDIR:-/dev/shm/rsg}"
export RAY_raylet_start_wait_time_s=180
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
mkdir -p "${RAY_TMPDIR}"

SCRIPT=examples/profile/workloads/geo3k/run_geo3k_rollout_ab.sh

# No Hydra overrides that change workload shape or artificially shorten training.
# param_offload / optimizer_offload stay True (run_geo3k_refocus_profile.sh defaults).
TRAIN_OVERRIDES=()

echo "################ ARM A: BASELINE (off) — full step ################"
CACHEBLEND_SELECTOR=off \
CACHEBLEND_IMAGE_SLOTS=-1 \
CACHEBLEND_FAST_APPLY=0 \
CACHEBLEND_COMPACT_PREFILL=0 \
  bash "${SCRIPT}" exact "${TRAIN_OVERRIDES[@]}"
echo "################ ARM A DONE ################"

ray stop --force 2>/dev/null || true
sleep 8

echo "################ ARM B: BEST ROLLOUT (kvdev + warm barrier) — full step ################"
CACHEBLEND_SELECTOR=kvdev \
CACHEBLEND_IMAGE_SLOTS=-1 \
CACHEBLEND_FAST_APPLY=0 \
CACHEBLEND_COMPACT_PREFILL=0 \
SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER=1 \
SGLANG_VLM_CACHEBLEND_PREFIX_WARMUP_BARRIER=1 \
SGLANG_VLM_CACHEBLEND_TARGET_TURNS=1 \
SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_WAIT_POLICY=bounded \
SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_MAX_WAIT_S=0 \
  bash "${SCRIPT}" exact "${TRAIN_OVERRIDES[@]}"
echo "################ ARM B DONE ################"

ray stop --force 2>/dev/null || true

echo "################ ALL DONE ################"
echo ""
echo "Compare step-2 timing (rollout = gen; training = olp + update_actor + update_weights):"
echo "  grep 'step:2' profile_logs_geo3k_rollout_ab/geo3k_refocus_exact_${RUN_TAG}_off_slotslast_fa0_cp0_bs64_n4.log | grep -oE 'timing_s/[^ ]+'"
echo "  grep 'step:2' profile_logs_geo3k_rollout_ab/geo3k_refocus_exact_${RUN_TAG}_kvdev_slotslast_fa0_cp0_bs64_n4.log | grep -oE 'timing_s/[^ ]+'"
echo ""
echo "Rollout E+P per-request (turn1 focus):"
echo "  python3 examples/profile/shared/analysis/analyze_profiling_logs.py \\"
echo "    --log-dir profile_logs_geo3k_refocus_exact_${RUN_TAG}_kvdev_slotslast_fa0_cp0 \\"
echo "    --suffix geo3k_refocus_exact_${RUN_TAG}_kvdev_slotslast_fa0_cp0_bs64_n4 --report"
