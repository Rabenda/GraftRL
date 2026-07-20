# MMDU benchmark profiling

MMDU prompts are multi-turn and multi-image. **Dataset parquet is unchanged**; rollout
shape is controlled by the ``mmdu_multiturn_agent`` loop in ``run_mmdu_profile.sh``.

## Recommended接法 (snowball multi-turn)

Unlike Geo3K refocus (synthetic 2-turn + artificial refocus image), MMDU keeps
dataset **user** questions/images and lets the **model generate** each assistant
reply.  That reply is appended into the live conversation before the next user
turn (true agent snowball — not replaying dataset assistant history).

| Runtime turn | Context | Typical effect |
|--------------|---------|----------------|
| 0 | first selected user (+ images) | prefill + model decode |
| 1..N-2 | prior model assistants + next dataset user (often new `<image>`) | longer prefill, reuse image KV |
| final | snowball history + final user question | main decode; reward uses dataset GT |

Returned to the trainer: **final-turn** ``prompt_ids`` / ``response_ids`` only
(so reward matches the last answer). Intermediate turns still call SGLang
``generate`` with ``agent_turn`` for CacheBlend / prefill-decode profiling.

Defaults: ``MMDU_RUNTIME_TURNS=4``, intermediate decode 128 tok, final 512 tok,
``VERL_PROFILE_ROLLOUT_ONLY=1`` (no ref/update_actor).

```bash
cd verl
export PATH=/data/conda_envs/verl_vision/bin:$PATH

CUDA_VISIBLE_DEVICES=0,1,4,7 \
DATA_ROOT=/workspace/repo/graftrl/verl/data/mmdu_benchmark_small_filtered_16384 \
TRAIN_BATCH_SIZE=64 ROLLOUT_N=4 \
MMDU_RUNTIME_TURNS=4 \
bash examples/profile/workloads/mmdu/run_mmdu_profile.sh
```

CacheBlend across turns:

```bash
SGLANG_VLM_CACHEBLEND=1 \
SGLANG_VLM_CACHEBLEND_SELECT=kvdev \
SGLANG_VLM_CACHEBLEND_TARGET_TURNS=all \
bash examples/profile/workloads/mmdu/run_mmdu_profile.sh
```

Legacy single-turn (one-shot full prompt):

```bash
MMDU_MULTITURN=0 VERL_PROFILE_ROLLOUT_ONLY=0 bash .../run_mmdu_profile.sh
```

## Prompt length distribution (offline filter)

### `mmdu_benchmark_small` (110-dialogue HF benchmark)

Measured with `analyze_prompt_lengths.py` (same path as training filter):

| cap | train kept | test kept | notes |
|-----|------------|-----------|-------|
| 8192 | 92 / 256 (36%) | 19 / 64 | enough for bs64, but drops most long multi-image context |
| 12288 | 146 / 256 (57%) | 35 / 64 | moderate trim |
| **16384** | **178 / 256 (70%)** | **52 / 64** | stable long-context baseline without fused kernels |
| 24576 | 235 / 256 (92%) | 61 / 64 | step2 ref log-prob OOM at bs64×n4 without fused kernels |

The 110-dialogue benchmark is intentionally hard (avg ~8.2k tokens, max ~24k). **You cannot reach 256 train rows at 8192 from benchmark alone** (cap ~92).

### `mmdu_45k_pool` (MMDU-45k instruct data, first N dialogues)

MMDU-45k dialogues are shorter on average (~5k tokens). Scanning the first 2500 dialogues (`turns=1`) yields:

| stat | value |
|------|-------|
| p50 prompt tokens | ~2470 |
| p95 | ~4024 |
| max | ~4393 |
| `<=8192` retention | **100%** (1500/1500 in our 1200 train + 300 test pool) |

Use this pool when you want **8192 context + 256 train rows** for stable 64×4 profiling.

## Build 8192 dataset from MMDU-45k

```bash
cd verl
export PATH=/data/conda_envs/verl_vision/bin:$PATH

# 1) Download mmdu-45k.json + mmdu-45k_pics.zip, convert 1200/300 row pool
bash examples/profile/workloads/mmdu/prepare_45k_pool.sh

# 2) Offline filter (all rows pass for first-2500-dialogue pool)
python3 examples/profile/workloads/mmdu/filter_overlong.py \
  --input-dir data/mmdu_45k_pool \
  --output-dir data/mmdu_45k_pool_filtered_8192 \
  --max-prompt-length 8192 \
  --num-workers 8

# 3) Slice standard 256/64 profiling subset
SRC_ROOT=data/mmdu_45k_pool_filtered_8192 \
DST_ROOT=data/mmdu_benchmark_8192 \
bash examples/profile/workloads/mmdu/slice_pool.sh
```

Pre-built: `data/mmdu_benchmark_8192` (256 train / 64 test, all `<=8192`).

## Offline filter (benchmark only)

```bash
cd verl
export PATH=/data/conda_envs/verl_vision/bin:$PATH

# inspect lengths
python3 examples/profile/workloads/mmdu/analyze_prompt_lengths.py \
  --input-dir data/mmdu_benchmark_small

# recommended dataset for stable 64×4 profiling
python3 examples/profile/workloads/mmdu/filter_overlong.py \
  --input-dir data/mmdu_benchmark_small \
  --output-dir data/mmdu_benchmark_small_filtered_16384 \
  --max-prompt-length 16384 \
  --num-workers 4
```

Pre-built dirs: `filtered_8192`, `filtered_12288`, `filtered_16384`, `filtered_24576`.

## Baseline run (64×4, 3 steps)

### 8192 + MMDU-45k pool (recommended for stable memory)

```bash
cd verl
export PATH=/data/conda_envs/verl_vision/bin:$PATH

CUDA_VISIBLE_DEVICES=0,1,4,7 \
DATA_ROOT=/workspace/repo/graftrl/verl/data/mmdu_benchmark_8192 \
TRAIN_BATCH_SIZE=64 \
ROLLOUT_N=4 \
TOTAL_STEPS=3 \
MAX_PROMPT_LENGTH=8192 \
FILTER_OVERLONG_PROMPTS=False \
SUFFIX=mmdu_benchmark_baseline_bs64_n4_s3_8192_45k \
LOG_ROOT=/workspace/repo/graftrl/verl/profile_logs_mmdu_baseline \
bash examples/profile/workloads/mmdu/run_mmdu_profile.sh
```

Auto `LOGPROB_MAX_TOKEN_LEN_PER_GPU=8704` (8192+1024+512). Expect much lower ref/old log-prob peak than 24k runs.

### 16384 + benchmark (longer context, benchmark-hard dialogues)

```bash
CUDA_VISIBLE_DEVICES=0,1,4,7 \
DATA_ROOT=/workspace/repo/graftrl/verl/data/mmdu_benchmark_small_filtered_16384 \
MAX_PROMPT_LENGTH=16384 \
FILTER_OVERLONG_PROMPTS=False \
bash examples/profile/workloads/mmdu/run_mmdu_profile.sh
```

If you must keep 24576 prompts, try fused log-prob to avoid materializing full vocab logits:

```bash
USE_FUSED_KERNELS=True \
LOGPROB_MAX_TOKEN_LEN_PER_GPU=25600 \
DATA_ROOT=.../mmdu_benchmark_small_filtered_24576 \
MAX_PROMPT_LENGTH=24576 \
bash examples/profile/workloads/mmdu/run_mmdu_profile.sh
```

## Why 24576 failed on step 2

- Step 1 passed with `LOGPROB_MAX_TOKEN_LEN_PER_GPU=25600` but `prompt_length/max=24091`.
- Step 2 OOM in `compute_ref_log_prob`: single-sequence logits ~ `seq × vocab × 2B` (~7–8GB at 25k tokens)
  plus SGLang KV reservation on the same GPUs.

Lowering prompt cap (16384) or enabling `USE_FUSED_KERNELS=True` addresses the logits peak.
