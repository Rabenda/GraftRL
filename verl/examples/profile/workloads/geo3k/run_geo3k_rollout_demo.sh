#!/usr/bin/env bash
# Geo3K rollout demo: baseline (off) vs rollout E+P optimization (kvdev + warm barrier).
#
# Prerequisites (run from graftrl/verl):
#   - 4× GPU, conda env with verl + patched sglang (see repo README)
#   - export CUDA_VISIBLE_DEVICES=0,1,2,3   # adjust to your machine
#   - Geo3K refocus data under verl/data/geo3k_refocus_exact/ (auto-prepared if missing)
#
# Scale floor: TRAIN_BATCH_SIZE=64 × ROLLOUT_N=4 (64×4). Do not shrink for profiling conclusions.
# Training path: standard PPO recompute (old_log_prob recomputed after rollout).
#
# Usage — pick ONE arm:
#   bash examples/profile/workloads/geo3k/run_geo3k_rollout_demo.sh baseline
#   bash examples/profile/workloads/geo3k/run_geo3k_rollout_demo.sh optimized
#
# After both runs, compare step-2 (step 1 is warmup):
#   grep 'step:2' profile_logs_geo3k_rollout_ab/geo3k_refocus_exact_${RUN_TAG}_off_*.log | grep timing_s
#   grep 'step:2' profile_logs_geo3k_rollout_ab/geo3k_refocus_exact_${RUN_TAG}_kvdev_*.log | grep timing_s
#   python3 examples/profile/shared/analysis/analyze_profiling_logs.py \
#     --log-dir profile_logs_geo3k_refocus_exact_${RUN_TAG}_kvdev_slotslast_fa0_cp0 \
#     --suffix geo3k_refocus_exact_${RUN_TAG}_kvdev_slotslast_fa0_cp0_bs64_n4 --report

set -euo pipefail

MODE="${1:-}"
if [[ "${MODE}" != "baseline" && "${MODE}" != "optimized" ]]; then
  echo "Usage: $0 {baseline|optimized}" >&2
  echo "" >&2
  echo "  baseline   — CacheBlend off (vanilla rollout)" >&2
  echo "  optimized  — kvdev + warm barrier + bounded wait (rollout E+P / KV graft)" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${VERL_ROOT}"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate verl_vision 2>/dev/null || true
elif [[ -f "/workspace/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "/workspace/miniconda3/etc/profile.d/conda.sh"
  conda activate verl_vision 2>/dev/null || true
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NGPUS="${NGPUS:-4}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export TOTAL_STEPS="${TOTAL_STEPS:-2}"
export AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-4}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.3}"
export ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-8192}"
export ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU="${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-8192}"
export RUN_TAG="${RUN_TAG:-demo}"
export RAY_TMPDIR="${RAY_TMPDIR:-/dev/shm/rsg}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-180}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
mkdir -p "${RAY_TMPDIR}"

AB_SCRIPT=examples/profile/workloads/geo3k/run_geo3k_rollout_ab.sh

case "${MODE}" in
  baseline)
    echo "[demo] ARM A — baseline (CacheBlend off, no warm barrier)"
    CACHEBLEND_SELECTOR=off \
    CACHEBLEND_IMAGE_SLOTS=-1 \
    CACHEBLEND_FAST_APPLY=0 \
    CACHEBLEND_COMPACT_PREFILL=0 \
      bash "${AB_SCRIPT}" exact
    ;;
  optimized)
    echo "[demo] ARM B — rollout E+P optimized (kvdev + warm barrier + bounded wait)"
    CACHEBLEND_SELECTOR=kvdev \
    CACHEBLEND_IMAGE_SLOTS=-1 \
    CACHEBLEND_FAST_APPLY=0 \
    CACHEBLEND_COMPACT_PREFILL=0 \
    SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER=1 \
    SGLANG_VLM_CACHEBLEND_PREFIX_WARMUP_BARRIER=1 \
    SGLANG_VLM_CACHEBLEND_TARGET_TURNS=1 \
    SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_WAIT_POLICY=bounded \
    SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_MAX_WAIT_S="${SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_MAX_WAIT_S:-10}" \
      bash "${AB_SCRIPT}" exact
    ;;
esac

echo ""
echo "[demo] Done. Logs under profile_logs_geo3k_rollout_ab/ and profile_logs_geo3k_refocus_exact_${RUN_TAG}_*"
echo "[demo] Analyze turn1 E+P: see header of this script."
