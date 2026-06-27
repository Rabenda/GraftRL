# VLM-CacheBlend: why `on` showed no reuse, and the fix

**Status:** for review (codex). No GPU rerun done yet — changes are no-GPU prep.
**Scope of changes:** 2 files, no kernel / capture-logic edits.

---

## 1. Symptom

`scale=2.0`, `ROLLOUT_N=8`, one GRPO group. `on` turn1 was ~2x slower than `off`
(6537 vs 3066 ms avg), read as "CacheBlend has no benefit / is slower".

## 2. What the logs actually show

From `profile_logs_cacheblend_stress_s2_on_4g_n8/model_forward_log_*.csv` (81 rows):

- `cacheblend_used == 0` for **every** row. `cacheblend_reused_tokens == 0` everywhere.
  → **Reuse never fired.** So the 2x is pure overhead of an inactive feature
  (donor capture + per-layer hooks), **not** a CacheBlend-vs-baseline result.
- The non-`none` rows:

  | role | n_image_tokens | fallback_reason |
  |------|----------------|-----------------|
  | donor | **2918** | donor_captured |
  | recipient | 5244 | image_token_count_mismatch |
  | recipient | 5244 | image_token_count_mismatch |
  | recipient | 8086 | batch:image_token_count_mismatch=2 |
  | recipient | 10898 | batch:image_token_count_mismatch=3 |
  | recipient | 8551 | batch:image_token_count_mismatch=2 |
  | recipient | 10488 | batch:image_token_count_mismatch=2 |

  The donor and its **own** request (same uid `b729a77c`) appear as donor=2918 and
  recipient=5244.

## 3. Root cause (evidence, not inference)

The differing `n_image_tokens` are **not** different image sizes. Checked the dumped
PNGs directly:

```
all 8  *_t1_refocus_output.png  ->  1596 x 2576  (~5244 tokens), identical
all 8  *_t0_chart_input.png     ->  1596 x 2576, identical
rollout_data: refocus_source == "oracle" for all 8 samples
image dumps: single uid 83335223, 8 samples x 2 turns
```

So the 8 siblings produce a **byte-identical 5244-token refocus image** (oracle
refocus works; grouping by `agent_uid` works). The varying counts in the CSV are
**chunked-prefill fragments** of that one 5244-token span (and 10488 ≈ both image
spans counted together when the slot selector sees a fragmented prompt).

Mechanism: the turn1 prompt is ~10.5k tokens (chart 5244 + refocus 5244 + text).
verl defaults `enable_chunked_prefill=True`; sglang auto-picks `chunked_prefill_size`
2k–8k (`server_args.py:907-946`). The refocus span starts past the chunk boundary, so
it is split across forward passes. The donor-capture hook
(`models/qwen2.py:_maybe_cacheblend_after_full_prefill`, line ~531) runs **per
forward** and only guards on `is_extend()` — it has **no "last chunk of this request"
check**, so it records only the image tokens present in that chunk (2918 of 5244).
Recipients then see donor=2918 ≠ recipient=5244 and bail at the first check
(`vlm_cacheblend.build_recipient_kv_blend_plan`, the `image_token_count_mismatch`
guard, line ~542).

**Net:** the only blocker is chunked prefill fragmenting the donor capture. Sibling
images are already identical, so no data change is needed.

### Note on the alternative (scale sweep 1.1–1.6)

Lowering scale to fit the image under the chunk boundary would also produce hits, but
it shrinks the reused span (~5244 → ~1500 tokens), throwing away the reuse payoff —
i.e. it caps the very benefit we want to measure. Disabling chunking keeps the full
5244-token image. Same number of reruns, stronger signal.

### Note on verl's `enable_chunked_prefill`

`actor_rollout_ref.rollout.enable_chunked_prefill` is **not wired into the sglang
path** (only vllm/trtllm async servers read it). The only effective lever is
`engine_kwargs.sglang.chunked_prefill_size`, passed through verbatim to `ServerArgs`
(`async_sglang_server.py:402,445`). `chunked_prefill_size <= 0` (`-1`) disables
chunked prefill (`managers/scheduler.py:740`).

---

## 4. Changes

### 4.1 `examples/profile/workloads/chart/run_cacheblend_stress.sh`

Append a single-chunk-prefill override (default on, env-overridable), and a
fail-fast assertion after `on` runs:

```bash
# default ON: disable chunked prefill so the donor captures the full image span.
# `+` prefix is required (engine_kwargs.sglang is an empty dict + hydra struct mode);
# matches repo idiom (tests/special_npu/run_qwen3_8b_grpo_mindspeedllm.sh:165).
if [[ "${CACHEBLEND_DISABLE_CHUNKED_PREFILL:-1}" == "1" ]]; then
  EXTRA_OVERRIDES+=(
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=${CACHEBLEND_CHUNKED_PREFILL_SIZE:--1}"
  )
fi
...
# after the run, on `on` only:
if [[ "${enabled}" == "1" && "${CACHEBLEND_ASSERT_USED:-1}" == "1" ]]; then
  python3 examples/profile/shared/analysis/assert_cacheblend_used.py \
    --log-root "${LOG_ROOT}" --suffix "${SUFFIX}"
fi
```

Knobs: `CACHEBLEND_DISABLE_CHUNKED_PREFILL=0` to keep chunking;
`CACHEBLEND_CHUNKED_PREFILL_SIZE=16384` to cap chunk size instead of disabling;
`CACHEBLEND_ASSERT_USED=0` to skip the gate. `set -euo pipefail` makes the failed
assertion abort the run.

### 4.2 `examples/profile/shared/analysis/assert_cacheblend_used.py` (new)

Reads `model_forward_log_<suffix>.csv`. PASS iff `sum(reused_tokens) > 0` **and**
`used_rows > 0`; otherwise exits non-zero with a fragment hint. Self-tested against
the existing failing `on` log:

```
[assert-cacheblend][FAIL] CacheBlend reuse never fired (reused_tokens_sum=0)...
[assert-cacheblend][HINT] donor captured a FRAGMENT of the span (chunked prefill split it):
  donor=[2918] vs recipient=[5244, 8086, 8551, 10488, 10898].
  Disable chunking: engine_kwargs.sglang.chunked_prefill_size=-1
```

---

## 5. How to verify (needs GPU)

```bash
export CUDA_VISIBLE_DEVICES=0,1,5,7
bash examples/profile/workloads/chart/run_cacheblend_stress.sh on
```

Expected after the fix: donor and recipient `n_image_tokens` both **5244**,
`cacheblend_used=1`, `reused_tokens > 0`, assertion PASS. Then run `both` for the real
off/on turn1 timing.

Watch-outs:
- Single-chunk prefill raises activation memory (one ~10.5k-token prefill). If OOM,
  bump `GPU_MEMORY_UTILIZATION` (default 0.35) or use
  `CACHEBLEND_CHUNKED_PREFILL_SIZE=16384` instead of `-1`.

---

## 6. Open questions for review

1. Disable chunked prefill (this change) vs. make donor capture **chunk-aware**
   (accumulate the image span across a request's forwards, mark donor complete only on
   the last chunk). Chunk-aware is the production-correct fix if chunked prefill must
   stay on during real training. Defer until a benefit signal is confirmed?
2. With `chunked_prefill_size=-1`, confirm the recipient fast-path / active-query-range
   logic still behaves (it was exercised only under chunking so far).
3. Should the assertion also gate on `recompute_ratio_effective` being in range, not
   just `reused_tokens > 0`?
