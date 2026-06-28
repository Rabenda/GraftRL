# VLM CacheBlend Donor-Ready Barrier

Rollout-side ordering guard for GRPO parallel branches using **VLM CacheBlend** (visual KV grafting).

## Problem

With `rollout.n > 1`, multiple branches of the same `agent_uid` can hit turn1 refocus prefill concurrently. A **recipient** may reach SGLang before the **donor** finishes capturing refocus image KV → `donor_not_ready` fallback → unstable `cacheblend_used` and lost speedup.

## Behavior

On the configured target turn (default **turn 1**), for each `(global_step, agent_uid)` group:

1. The **first** turn1 request runs alone (barrier **donor**).
2. After it completes, the key is marked warmed; later branches proceed in parallel (barrier **recipient**).
3. Other turns / missing metadata → **bypass** (no barrier).

Requires **CacheBlend enabled** and sticky routing via `agent_uid` (already used by GraftRL).

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `SGLANG_VLM_CACHEBLEND` | `0` | Must be `1` for barrier to activate |
| `SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER` | `1` | `0` disables barrier (ablation off) |
| `SGLANG_VLM_CACHEBLEND_TARGET_TURN` | `1` | Agent turn to serialize (refocus turn) |
| `SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_KEEP_STEPS` | `4` | Prune warmed keys older than the last N `global_step` values |
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
- `verl_sglang_generate_log_*.csv`: turn1 `sglang_call_ms` / prefill latency

## Scope / limitations

- Barrier state lives on each **`LLMServerClient` instance** (not a global Ray actor).
- Experiments in this repo use **`actor_rollout_ref.rollout.agent.num_workers=1`**.
- Multi-worker rollout needs a shared barrier (Ray actor or scheduler); not implemented here.

Implementation: `verl/verl/workers/rollout/llm_server.py`.
