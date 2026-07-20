# MMSearch-R1 rollout-reuse prototype

This workload is the first vertical prototype for the library-wide
[group rollout reuse design](../../GROUP_ROLLOUT_REUSE_DESIGN.md).

Its main path tests artifact reuse and LLM-prefill reuse without a custom ViT
cache:

- Search results are assigned content identities for attribution and
  dependency-aware tool/tokenization reuse.
- Every request runs the normal vision encoder. Both the custom per-item ViT
  cache and SGLang's whole-bundle embedding cache are disabled.
- Optional CacheBlend experiments reuse work only after image tokens enter the
  LLM prefill stack; each request keeps its own prompt order, position ids and
  mRoPE.
- Retrieved artifacts and observation tokenization use a dependency-aware
  single-flight registry: one branch computes and concurrent recipients reuse.
- Equivalence identity is separate from group/policy sharing scope, and
  `EXACT`, `LOCAL`, and already-materialized `SKIP` are reported separately.
- Group branches stay in one agent owner and one SGLang replica so process-local
  and GPU-local artifacts are physically reachable.
- Weight updates, memory release, and explicit cache flush invalidate model-bound
  LLM donor K/V and sparse-decode plans.

The default keeps all five retrieved results. Fixed top5-to-top2 is available
only as an explicit `SKIP` experiment and is not treated as reuse or as the
paper's main algorithm.

LLM prefill CacheBlend and sparse decoding are both off by default.

## Quick run

The default uses the local five-row parquet, `n=4`, two rollout-only steps and all
five retrieved candidates. It is a wiring check, not a performance-scale result:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
  bash examples/profile/workloads/mmsearch_r1/run_mmsearch_r1_profile.sh
```

The runner prints latest-step artifact/cache counts, images actually sent
through the vision encoder, turn-1 tokens and E2E, and real MMSearch reward
fields.

## Paired LLM-prefill check

Keep context identical in both runs so the comparison attributes changes to LLM
prefill reuse rather than pruning:

```bash
# Control: all five results, no custom reuse.
SUFFIX=mmsearch_prefill_control \
MMSEARCH_R1_CONTEXT_TOPK=5 \
SGLANG_VLM_CACHEBLEND=0 \
bash examples/profile/workloads/mmsearch_r1/run_mmsearch_r1_profile.sh

# Reuse: all five results, LLM prefill CacheBlend only.
SUFFIX=mmsearch_prefill_reuse \
MMSEARCH_R1_CONTEXT_TOPK=5 \
SGLANG_VLM_CACHEBLEND=1 \
SGLANG_VLM_CACHEBLEND_SELECT=kvdev \
bash examples/profile/workloads/mmsearch_r1/run_mmsearch_r1_profile.sh
```

## Optional SKIP experiment

Run static pruning separately. Its token/E2E reduction must not be combined
with or reported as exact-reuse benefit:

```bash
SUFFIX=mmsearch_skip_top2 \
MMSEARCH_R1_CONTEXT_TOPK=2 \
bash examples/profile/workloads/mmsearch_r1/run_mmsearch_r1_profile.sh
```

## Data and controls

- `MMSEARCH_R1_IMAGE_SEARCH_TOPK` and `MMSEARCH_R1_TEXT_SEARCH_TOPK` control
  retrieved candidates.
- `MMSEARCH_R1_CONTEXT_TOPK` is the optional static `SKIP` policy. It defaults
  to the larger candidate count, so neither modality is pruned by default.
- `VERL_ROLLOUT_REUSE_GROUP_AFFINITY=1` prevents a GRPO group from being split
  across agent workers. `VERL_ROLLOUT_REUSE_SERVER_AFFINITY=1` keeps the same
  group on one SGLang replica without a donor barrier. Both remain enabled in
  control and reuse runs so routing is not an A/B confounder.
- `SGLANG_VLM_CACHE_SIZE_MB=0` is forced by the runner so the upstream
  whole-bundle embedding cache cannot confound LLM-prefill measurements.
- `MMSEARCH_R1_PROFILE_FORCE_TOOL=image|text|auto` chooses the deterministic
  profiling path or model-emitted search tags.
- The current search operator is synthetic and deterministic. The runtime and
  document-stage boundaries are real, but live fetch/parse/chunk adapters are
  the next integration rather than an already implemented claim.
- Set `TRAIN_FILE`/`VAL_FILE` to a full MMSearch parquet for quality work. The
  bundled `mini_data.pq` has five real examples; the 2k file in the sibling
  MMSearch repository may remain a Git LFS pointer until pulled.
