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

## Documentation

**中文总览（从建仓到现在做了什么、算法步骤、代码位置）：**  
[`docs/GraftRL_项目全历程.md`](docs/GraftRL_项目全历程.md)

**Geo3K 实验复现（给合作者）：**  
[`docs/REPRODUCE_GEO3K.md`](docs/REPRODUCE_GEO3K.md) — 环境安装 + baseline/optimized 两条命令 + 结果判读

版本标签说明：[`docs/RELEASES.md`](docs/RELEASES.md)

## Quick start (profiling)

See `verl/examples/profile/README.md` for how to run workloads.

### Geo3K rollout demo — baseline vs E+P optimization

The fastest way to reproduce **CacheBlend off vs kvdev + warm barrier** on the Geo3K
refocus workload (**64×4**, standard PPO recompute — no training bypass):

**Prerequisites**

- 4 GPUs (80GB recommended) — see **[`docs/REPRODUCE_GEO3K.md`](docs/REPRODUCE_GEO3K.md)** for full env setup (`requirements-reproduce.txt` + editable `verl`/`sglang`)
- Run from `graftrl/verl`

```bash
cd verl
export CUDA_VISIBLE_DEVICES=0,1,2,3   # use your 4 GPUs
conda activate verl_vision            # or your env name
export RAY_TMPDIR=/dev/shm/rsg        # short path; avoids Ray socket issues
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
mkdir -p "${RAY_TMPDIR}"
```

**ARM A — baseline** (CacheBlend off):

```bash
bash examples/profile/workloads/geo3k/run_geo3k_rollout_demo.sh baseline
```

**ARM B — rollout E+P optimized** (kvdev + warm barrier + bounded wait):

```bash
bash examples/profile/workloads/geo3k/run_geo3k_rollout_demo.sh optimized
```

Each run executes **2 training steps**; **analyze step 2** (step 1 is warmup).

```bash
# Whole-step timing (rollout = timing_s/gen)
grep 'step:2' profile_logs_geo3k_rollout_ab/geo3k_refocus_exact_demo_off_*.log | grep timing_s
grep 'step:2' profile_logs_geo3k_rollout_ab/geo3k_refocus_exact_demo_kvdev_*.log | grep timing_s

# Per-request turn1 E+P report
python3 examples/profile/shared/analysis/analyze_profiling_logs.py \
  --log-dir profile_logs_geo3k_refocus_exact_demo_kvdev_slotslast_fa0_cp0 \
  --suffix geo3k_refocus_exact_demo_kvdev_slotslast_fa0_cp0_bs64_n4 --report
```

If optimized runs show `donor_ready=0%` in `cacheblend_barrier_log_*.csv`, raise wait
time: `SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_MAX_WAIT_S=10 bash ... optimized` (demo
defaults to 10s).

More context: [`docs/GraftRL_项目全历程.md`](docs/GraftRL_项目全历程.md) §11,
[`verl/docs/geo3k_profiling_archive_0702.md`](verl/docs/geo3k_profiling_archive_0702.md).

### Chart refocus (legacy quick start)

Typical CacheBlend profiling (Chart refocus):

```bash
cd verl
export CUDA_VISIBLE_DEVICES=0,1,2,3
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
