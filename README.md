# GraftRL

**GraftRL** grafts cross-branch visual KV from a donor GRPO rollout branch onto
recipient branches during VLM turn-level prefill. It targets the non-prefix refocus
image span that RadixCache cannot reuse, using selective KV blending inspired by
CacheBlend.

This repository is a monorepo:

| Path | Role |
|------|------|
| `verl/` | Customized [verl](https://github.com/verl-project/verl) rollout, profiling, and agent loops |
| `sglang/` | Customized [SGLang](https://github.com/sgl-project/sglang) engine hooks (`vlm_cacheblend`, GRPO similarity cache) |
| `docs/` | Design notes and experiment summaries |

## Relation to prior work

**GraftRL is not JigsawRL.** JigsawRL was our previous project name for an earlier
line of work. GraftRL focuses on **LLM prefill (step 6) visual KV reuse** across GRPO
branches when turn0 responses diverge and refocus images are similar but not identical.

## Upstream pins (v0.0 baseline)

| Component | Upstream | Commit |
|-----------|----------|--------|
| verl | `verl-project/verl` | `802256a7` |
| sglang | `sgl-project/sglang` | `0189f41` |

## Quick start (profiling)

See `verl/examples/profile/README.md` and `docs/RELEASES.md` for staged milestones.

Typical CacheBlend profiling (Chart refocus):

```bash
cd verl
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SGLANG_GRPO_SIM_CACHE=0
export SGLANG_VLM_CACHEBLEND=1
bash examples/profile/workloads/chart/run_model_refocus_profile.sh
```

Stress workload (large image span, `rollout.n=8`):

```bash
bash examples/profile/workloads/chart/run_cacheblend_stress.sh on
```

## License

`verl/` and `sglang/` retain their respective upstream licenses. See each subtree
and root `NOTICE` when present.
