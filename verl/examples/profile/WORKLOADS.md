# Vision Token Reuse — Profile Workloads

GRPO + SGLang profiling for **image-token reuse** experiments (Phase 1 similarity, Phase 2 replacement).

## Directory layout

```
examples/profile/
  README.md                 # upstream verl profiler entrypoints
  WORKLOADS.md              # this file
  run_refocus_chart_multiturn_profile.sh   # → workloads/chart/ (compat)
  vtool_agent_loop.py       # shim → shared/agent/
  analyze_similarity_unified.py  # shim → shared/analysis/

  workloads/                # one folder per dataset experiment
    geo3k/                  # baseline single-turn
    sokoban/                # multi-turn env (legacy; low token count)
    chart/                  # Refocus / VTool bar-chart QA (~580 tokens)
    deepeyes/               # DeepEyes visual_toolbox_v2 zoom/crop QA

  shared/                   # reused across workloads
    agent/                  # vtool_agent_loop, refocus tools, deepeyes_tools
    analysis/               # similarity, EPD logs, phase1/2 scripts
    reports/                # request-flow walkthrough HTML

  data_preprocess/          # profile-only parquet builders (see README.md)
    chart/                  # Refocus / VTool download, filter, smoke converts
    deepeyes/               # DeepEyes visual_toolbox_v2 download
    geo3k/                  # text-only ablation parquet

  archive/                  # early experiments (dummy crop, docs)
```

## Quick start (bs64 × n4)

Chart workload is always **64 groups × 4 GRPO branches = 256 rollouts** per step. Do not use smaller `TRAIN_BATCH_SIZE` for profiling.

| Workload | Prepare data | Rollout | Similarity analysis |
|----------|--------------|---------|-------------------|
| **Geo3K** | `data/geo3k/*.parquet` | `workloads/geo3k/run_geo3k_full_profile.sh` | `shared/analysis/analyze_profiling_logs.py` |
| **Sokoban** | `workloads/sokoban/prepare_sokoban_data.sh` | `workloads/sokoban/run_sokoban_rollout_profile.sh` | `workloads/sokoban/run_sokoban_similarity.sh` |
| **Chart** | `workloads/chart/prepare_data.sh` (→ `data_preprocess/chart/`) | `workloads/chart/run_rollout_profile.sh` | `workloads/chart/run_similarity.sh` |
| **DeepEyes** | `workloads/deepeyes/prepare_data.sh` | `workloads/deepeyes/run_rollout_profile.sh` | `workloads/deepeyes/run_similarity.sh` |

All rollout scripts need patched `sglang_vision_profile` on `PYTHONPATH` and `verl.trainer.main_ppo` via `examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh`.

## Chart-specific notes

Three refocus modes (pick one per rollout):

| Mode | Command flag | turn1 source | branch 分化 |
|------|--------------|--------------|------------|
| **Oracle** (default) | — | 同一段 dataset oracle | 无（4 branch 同图） |
| **Diversified oracle** | `VTOOL_ORACLE_DIVERSIFY=1` | 人工按 branch 换 bbox | 有（可控） |
| **Model refocus** | `VTOOL_MODEL_REFOCUS=1` or `run_model_refocus_profile.sh` | 各 branch turn0 模型输出 | 有（真实语义采样） |

Recommended entry points (no manual `export` needed — each script sets data + LOG_ROOT):

```bash
# Clean + forced oracle → profile_logs_vtool_chart_clean/
bash examples/profile/workloads/chart/run_clean_oracle_profile.sh

# Clean + model refocus → profile_logs_vtool_chart_model_refocus/
bash examples/profile/workloads/chart/run_model_refocus_profile.sh

# Similarity (match the rollout you ran)
bash examples/profile/workloads/chart/run_similarity_clean_oracle.sh
bash examples/profile/workloads/chart/run_similarity_model_refocus.sh
```

- **Clean parquet** (valid turn0→turn1 gate for oracle path): `examples/profile/data_preprocess/chart/filter_refocus_chart_oracle.py`
- **Model refocus** 也建议用 clean parquet（bbox 可解析），但成功率取决于 base 模型能否写出合法 `focus_on_*` 代码
- **Do not use** raw `test.parquet` for oracle/refocus profiling (826/826 rows lack teacher `thoughts` / oracle code; normal for VTool-R1 eval split).

## DeepEyes-specific notes

Data: HuggingFace `ChenShawn/DeepEyes-Datasets-47k`, filtered to `env_name=visual_toolbox_v2`.

```bash
# Prepare (~2000 train + 200 test by default)
bash examples/profile/workloads/deepeyes/prepare_data.sh

# CPU-only validation (no GPU)
bash examples/profile/workloads/deepeyes/dry_run_checks.sh

# Rollout (bs64 × n4) → profile_logs_deepeyes/
bash examples/profile/workloads/deepeyes/run_rollout_profile.sh

# Similarity (after rollout, needs GPU)
bash examples/profile/workloads/deepeyes/run_similarity.sh
```

Agent: `deepeyes_agent` parses `<tool_call>` JSON with `image_zoom_in_tool` + `bbox_2d`, crops the **original** image (DeepEyes semantics), appends cropped image for next turn. Image dump roles: `deepeyes_input`, `zoom_output`.

Unlike Chart, there is no oracle/clean filter yet — success rate depends on base Qwen2.5-VL learning the DeepEyes XML+JSON tool format.

## Outputs (gitignored)

Each experiment line uses its **own** `profile_logs_*` tree — do not mix.

| Experiment | `LOG_ROOT` | Refocus mode |
|------------|------------|--------------|
| Origin (raw HF parquet) | `profile_logs_refocus_chart_origin/` | oracle if row has code, else model |
| Raw + forced oracle | `profile_logs_vtool_chart_raw/` | teacher oracle |
| **Clean + oracle** | `profile_logs_vtool_chart_clean/` | teacher oracle (4 branch 同图) |
| Clean + diversified | `profile_logs_vtool_chart_diversified/` | teacher, per-branch bbox |
| **Clean + model** | `profile_logs_vtool_chart_model_refocus/` | 各 branch 模型 turn0 代码 |
| **DeepEyes zoom** | `profile_logs_deepeyes/` | 各 branch 模型 `<tool_call>` bbox crop |

Under each `LOG_ROOT`:

- `image_dump_${SUFFIX}/manifest.jsonl`
- `verl_sglang_generate_log_${SUFFIX}.csv` (+ model/vision encoder logs)
- `similarity/` after `run_similarity.sh` or `run_similarity_model_refocus.sh`

Legacy names (`profile_logs_refocus_chart/`, `profile_logs_vtool_chart/`) are older runs; new work should use the table above.

## Core framework patches (under `verl/`)

Required for multimodal GRPO rollout + SGLang inference logging:

- `verl/experimental/agent_loop/agent_loop.py`
- `verl/workers/rollout/sglang_rollout/async_sglang_server.py`
- `verl/workers/rollout/llm_server.py`
