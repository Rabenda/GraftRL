#!/usr/bin/env bash
# Clean baseline-vs-best rollout AB on idle GPUs 0,1,4,5.
# Both arms use identical standard-PPO training (old_log_prob recompute); the ONLY
# difference is the rollout optimization stack.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate verl_vision

export CUDA_VISIBLE_DEVICES=2,3,6,7
export NGPUS=4
export TRAIN_BATCH_SIZE=64
export ROLLOUT_N=4
# recompute mode (NO training bypass): the actor recomputes old_log_probs, i.e. the
# original PPO training path. Applied identically to BOTH arms, so the only variable is
# the rollout optimization stack.
export TOTAL_STEPS=5
export AGENT_NUM_WORKERS=4
# Lower SGLang util + smaller per-GPU token packing so the recompute old_log_prob /
# update_actor passes fit in 80GB (recompute OOM'd earlier at util 0.35 + 8192). Both
# arms share these, so the rollout comparison stays clean.
export GPU_MEMORY_UTILIZATION=0.3
export ACTOR_MAX_TOKEN_LEN_PER_GPU=4096
export ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU=4096
export RUN_TAG=clean_0702b_2367
export RAY_TMPDIR=/tmp/r4g
mkdir -p /tmp/r4g

SCRIPT=examples/profile/workloads/geo3k/run_geo3k_rollout_ab.sh

echo "################ ARM A: BASELINE (off) ################"
CACHEBLEND_SELECTOR=off bash "${SCRIPT}" exact
echo "################ ARM A DONE ################"

ray stop --force 2>/dev/null || true
sleep 8

echo "################ ARM B: BEST ROLLOUT (kvdev + slots=all + fast_apply + compact) ################"
CACHEBLEND_SELECTOR=kvdev \
CACHEBLEND_IMAGE_SLOTS=all \
CACHEBLEND_FAST_APPLY=1 \
CACHEBLEND_COMPACT_PREFILL=1 \
  bash "${SCRIPT}" exact
echo "################ ARM B DONE ################"

ray stop --force 2>/dev/null || true
echo "################ ALL DONE ################"
