# GraftRL release milestones

Staged tags document incremental research progress from upstream baseline to the
current GraftRL implementation. Pair **verl** and **sglang** subtrees at the same tag.

## Tags

### `v0.0-baseline`

- **verl** @ upstream `802256a7`
- **sglang** @ upstream `0189f41`
- No GraftRL / profiling / cacheblend code yet.
- Monorepo skeleton: `README.md`, `.gitignore`, this file.

### `v0.1-profiling` (planned)

- EPD-style profiling: `verl_sglang_generate_log`, `vision_encoder_log`, `model_forward_log`.
- Geo3K full-profile scripts and `analyze_profiling_logs.py`.

### `v0.2-motivation` (planned)

- Geo3K text-only padded baseline.
- Unified similarity analysis (`analyze_similarity_unified.py`).
- Refocus Chart agent / vtool profiling skeleton.

### `v0.3-vit-reuse` (planned)

- `SGLANG_GRPO_SIM_CACHE` ViT partial reuse (window / token / merged / token_sparse).
- Negative result: merged-token similarity high but too late; token_sparse too slow.
- Doc: `PARTIAL_WINDOW_REUSE_EXPERIMENT_SUMMARY.md`.

### `v0.4-graft-core` (planned)

- `vlm_cacheblend.py`, unit tests, donor KV store.
- `SGLANG_VLM_CACHEBLEND` macro, design doc `VLM_CACHEBLEND_DESIGN.md`.

### `v0.5-graft-e2e` (planned)

- Recipient fast path in attention backends + QKV/MLP skip.
- Rollout metadata (`agent_uid`, `agent_turn`), AB scripts, correctness gate.

### `v0.6-stress` (planned)

- `refocus_chart_cacheblend_stress.py`, `run_cacheblend_stress.sh`.
- Chunked-prefill fix + `assert_cacheblend_used.py`.
- Target: measurable reuse on enlarged image span with `rollout.n=8`.

## How to checkout a milestone

```bash
git checkout v0.4-graft-core   # example
```

After checkout, install/run per `verl/examples/profile/README.md` for that era.
