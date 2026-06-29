# VLM CacheBlend Donor-Ready Barrier

Rollout-side ordering guard for GRPO parallel branches using **VLM CacheBlend** (visual KV grafting).

## Problem

With `rollout.n > 1`, multiple branches of the same `agent_uid` can hit the same
agent turn concurrently. A **recipient** may reach SGLang before the **donor**
finishes warming the reusable state → `donor_not_ready` fallback on CacheBlend
turns, unstable `cacheblend_used`, and lost speedup.

## Behavior

For each `(global_step, agent_uid, agent_turn)` group:

1. The **first** eligible request runs alone (barrier **donor**).
2. After it completes, the key is marked warmed; sibling branches proceed in
   parallel (barrier **recipient**).
3. Missing metadata / disabled turn → **bypass** (no barrier).

Turn0 uses a `prefix:` warmup key to warm SGLang's prefix/RadixCache path. It does
not use donor KV grafting, kvdev, or cosine selection. Agent turns >= 1 use
`cacheblend:` warmup keys when selected by `SGLANG_VLM_CACHEBLEND_TARGET_TURNS`.

Requires **CacheBlend enabled** and sticky routing via `agent_uid` (already used by GraftRL).

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `SGLANG_VLM_CACHEBLEND` | `0` | Must be `1` for barrier to activate |
| `SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER` | `1` | `0` disables barrier (ablation off) |
| `SGLANG_VLM_CACHEBLEND_TARGET_TURN` | `1` | Legacy single CacheBlend target turn |
| `SGLANG_VLM_CACHEBLEND_TARGET_TURNS` | `SGLANG_VLM_CACHEBLEND_TARGET_TURN` | CacheBlend turns: `all` or comma-separated turns such as `1,2,3` |
| `SGLANG_VLM_CACHEBLEND_PREFIX_WARMUP_BARRIER` | `1` | Warm turn0 prefix/RadixCache before sibling branches |
| `SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_KEEP_STEPS` | `4` | Prune warmed keys older than the last N `global_step` values |
| `SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_TIMEOUT_S` | `300` | Max seconds a recipient waits for donor ready before proceeding |
| `SGLANG_INFERENCE_LOG_DIR` | — | If set, append `cacheblend_barrier_log_{suffix}.csv` |
| `SGLANG_INFERENCE_LOG_SUFFIX` | — | Optional CSV filename suffix |
| `VERL_LOGGING_LEVEL` | `WARN` | Set `INFO` to see `[VLMCacheBlendBarrier]` lines |

## Barrier log fields (CSV / log line)

- `barrier_enabled`, `barrier_role` (`donor` | `recipient` | `bypass`)
- `agent_uid`, `agent_turn`, `target_turn`, `rollout_idx`, `global_step`
- `wait_ms` — time blocked on the per-group asyncio lock
- `donor_ready` — `True` when proceeding after donor warmup for that key

## Ablation checklist

Compare runs with `SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER=0` vs `1` (CacheBlend on both):

- `cacheblend_barrier_log_*.csv`: `barrier_role=donor` count per group, `wait_ms` on recipients
- `model_forward_log_*.csv`: `cacheblend_used`, `cacheblend_fallback_reason=donor_not_ready`
- `verl_sglang_generate_log_*.csv`: per-turn `sglang_call_ms` / prefill latency

## Multi-worker support

- Barrier state is coordinated by a shared **`GlobalCacheBlendCoordinator` Ray actor**.
- With `SGLANG_VLM_CACHEBLEND_TARGET_TURNS=all`, the total number of rollout turns
  does not need to be known ahead of time; every observed turn >= 1 is eligible for
  CacheBlend.
- CacheBlend still stores donor K/V inside each SGLang replica; donor K/V is **not**
  copied across replicas.
- verl routes CacheBlend requests with a sticky key based on
  `(training_global_step, agent_uid)`, so branches from the same GRPO group are sent
  to the same SGLang replica when metadata is present.
- Supported scope: **multiple AgentLoopWorkers + sticky routing to one SGLang replica
  per GRPO group**.
- Out of scope: reusing donor K/V when the same GRPO group is intentionally split
  across multiple SGLang replicas.

Run Geo3K multi-worker profiling by setting:

```bash
AGENT_NUM_WORKERS=4 bash examples/profile/workloads/geo3k/run_geo3k_refocus_profile.sh exact
```

After a run, inspect the logs without GPUs:

```bash
python3 examples/profile/shared/analysis/validate_cacheblend_multiworker.py \
  --log-dir profile_logs_geo3k_refocus_exact \
  --suffix geo3k_refocus_exact_bs64_n4
```

Implementation: `verl/verl/workers/rollout/llm_server.py`.
