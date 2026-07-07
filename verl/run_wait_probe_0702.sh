#!/usr/bin/env bash
# Warm-barrier MAX_WAIT_S probe.
#
# Purpose: canary_clean_tracked kvdev had donor_ready=0 for 100% of 384 recipients
# (MAX_WAIT_S=0 => recipients never wait => graft hit-rate ~2% => turn1 +85%).
# This probe raises MAX_WAIT_S and measures whether donor_ready flips to 1
# and whether turn1 e2e recovers.
#
# Discriminator:
#   donor_ready% jumps up + turn1 drops  => WAIT-POLICY problem (donor merely late)
#   donor_ready% still ~0                 => EVICTION problem   (donor KV gone; wait can't help)
#
# Scale is fixed at the project floor (64x4, recompute). Do NOT shrink.
#
# Usage:
#   bash run_wait_probe_0702.sh            # runs the single most-informative value: 3s
#   bash run_wait_probe_0702.sh 0.5 1 2 3  # full sweep (4x GPU cost)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate verl_vision

# ---- values to probe (default: just 3s, the most decisive single point) ----
WAITS=("$@")
if [ "${#WAITS[@]}" -eq 0 ]; then WAITS=(3); fi

# ---- shared cluster / scale (identical to canary_clean_tracked kvdev) ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,6,7}"
export NGPUS=4
export TRAIN_BATCH_SIZE=64
export ROLLOUT_N=4
export TOTAL_STEPS="${TOTAL_STEPS:-2}"        # step1 warmup; analyze step2
export AGENT_NUM_WORKERS=4
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"

# ---- Ray hygiene (short tmpdir; visible-devices flag in shell env, not Hydra) ----
export RAY_TMPDIR="${RAY_TMPDIR:-/dev/shm/rsg}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
mkdir -p "${RAY_TMPDIR}"

SCRIPT=examples/profile/workloads/geo3k/run_geo3k_rollout_ab.sh

for W in "${WAITS[@]}"; do
  # RUN_TAG must be filesystem-safe: 0.5 -> w0p5
  WSAFE="w$(echo "$W" | tr '.' 'p')"
  export RUN_TAG="waitprobe_${WSAFE}"
  LOGDIR="profile_logs_geo3k_refocus_exact_${RUN_TAG}_kvdev_slotslast_fa0_cp0"
  SUFFIX="geo3k_refocus_exact_${RUN_TAG}_kvdev_slotslast_fa0_cp0_bs64_n4"

  echo "################ WAIT PROBE: MAX_WAIT_S=${W} (RUN_TAG=${RUN_TAG}) ################"
  ray stop --force 2>/dev/null || true
  sleep 5

  CACHEBLEND_SELECTOR=kvdev \
  CACHEBLEND_IMAGE_SLOTS=-1 \
  CACHEBLEND_FAST_APPLY=0 \
  CACHEBLEND_COMPACT_PREFILL=0 \
  SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER=1 \
  SGLANG_VLM_CACHEBLEND_PREFIX_WARMUP_BARRIER=1 \
  SGLANG_VLM_CACHEBLEND_TARGET_TURNS=1 \
  SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_WAIT_POLICY=bounded \
  SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_MAX_WAIT_S="${W}" \
    bash "${SCRIPT}" exact 2>&1 | tee "waitprobe_${WSAFE}_$(date +%Y%m%d_%H%M%S).log"

  echo "---------------- ANALYSIS MAX_WAIT_S=${W} ----------------"
  BARRIER_CSV="${LOGDIR}/cacheblend_barrier_log_${SUFFIX}.csv"
  GEN_CSV="${LOGDIR}/verl_sglang_generate_log_${SUFFIX}.csv"
  MF_CSV="${LOGDIR}/model_forward_log_${SUFFIX}.csv"
  W="$W" BARRIER_CSV="$BARRIER_CSV" GEN_CSV="$GEN_CSV" MF_CSV="$MF_CSV" python3 - <<'PY'
import csv, os, statistics
from collections import Counter, defaultdict
W=os.environ["W"]; bp=os.environ["BARRIER_CSV"]; gp=os.environ["GEN_CSV"]; mp=os.environ["MF_CSV"]
def load(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []
bar, gen, mf = load(bp), load(gp), load(mp)
print(f"[MAX_WAIT_S={W}] step2 verdict")
for step in ["2"]:
    br=[r for r in bar if r.get("global_step")==step and r.get("barrier_role")=="recipient"]
    if br:
        dr=Counter(r.get("donor_ready","?") for r in br)
        ready=dr.get("1",0); tot=len(br)
        print(f"  donor_ready: 1={ready}/{tot} ({100*ready/tot:.1f}% hit)  raw={dict(dr)}")
        bw=[float(r["barrier_wait_ms"]) for r in br if r.get("barrier_wait_ms")]
        if bw: print(f"  barrier_wait_ms: mean={statistics.mean(bw):.0f} max={max(bw):.0f}")
    mfr=[r for r in mf if r.get("global_step")==step]
    used=sum(1 for r in mfr if r.get("cacheblend_used")=="1")
    print(f"  model_forward graft_used rows: {used}/{len(mfr)}")
    g=[r for r in gen if r.get("global_step")==step]
    if g:
        byturn=defaultdict(list)
        for r in g: byturn[r["agent_turn"]].append(float(r["generate_e2e_ms"]))
        for t in sorted(byturn):
            v=byturn[t]; print(f"  turn{t} e2e: mean={statistics.mean(v):.0f}ms n={len(v)}")
print("  ref baseline off@canary step2: turn0=3512 turn1=3327 ; kvdev MAX_WAIT_S=0: turn0=2757 turn1=6163")
PY
  echo "################ DONE MAX_WAIT_S=${W} ################"
  ray stop --force 2>/dev/null || true
  sleep 5
done

echo "################ ALL WAIT PROBES DONE ################"
