# Geo3K Refocus Profiling 归档总览（2026-07-02）

> **用途**：在删除远端 `profile_logs_*` 原始目录、仅保留 tar 包或本地副本后，用本文档快速恢复所有核心实验结论、配置差异、数字与脚本路径。  
> **归档包**：`/workspace/repo/graftrl/verl/profile_logs_bundle_0702.tar.gz`（约 4.2G，含 110 个 `profile_logs_*` + `profile_gallery`）  
> **代码根目录**：`/workspace/repo/graftrl/verl`  
> **最小有效实验规模**：`TRAIN_BATCH_SIZE=64` × `ROLLOUT_N=4`（64×4），4× `AgentLoopWorker`

---

## 目录

1. [优化栈一览](#1-优化栈一览)
2. [指标与口径（必读）](#2-指标与口径必读)
3. [MODES 说明（recompute / bypass / audit）](#3-modes-说明)
4. [标准实验配置](#4-标准实验配置)
5. [Baseline 与 Training 占比分析](#5-baseline-与-training-占比分析)
6. [Rollout 优化结论（2 卡 vs 4 卡）](#6-rollout-优化结论2-卡-vs-4-卡)
7. [各优化项逐项结论](#7-各优化项逐项结论)
8. [推荐配置与复测脚本](#8-推荐配置与复测脚本)
9. [负结果与踩坑记录](#9-负结果与踩坑记录)
10. [与 JigsawRL / oyhh 前作对比](#10-与-jigsawrl--oyhh-前作对比)
11. [日志文件结构](#11-日志文件结构)
12. [实验 Run 索引（按类别）](#12-实验-run-索引按类别)
13. [关键 Run 数字表](#13-关键-run-数字表)
14. [磁盘清理与归档](#14-磁盘清理与归档)
15. [分析工具命令](#15-分析工具命令)

---

## 1. 优化栈一览

| 层级 | 名称 | 环境变量 / 配置 | 作用 |
|------|------|-----------------|------|
| **① KV Graft** | CacheBlend 本体 | `CACHEBLEND_SELECTOR=kvdev/cos/off` | donor/recipient 配对，选择性重算 image-token K/V |
| **② Slot Reuse** | Method A | `CACHEBLEND_IMAGE_SLOTS=-1`（last）或 `all` | 控制 graft 哪些 image slot |
| **③ Apply 优化** | Method B | `CACHEBLEND_FAST_APPLY=0/1` | 缓存 scatter/reuse 索引 |
| **④ Compute Skip** | M2 等 | `skip_reuse_qkv_proj` 等（引擎内） | reused token 跳过 QKV/MLP/attention |
| **⑤ Compact Prefill** | Method C | `CACHEBLEND_COMPACT_PREFILL=0/1` | 物理缩短 decoder 活跃序列 |
| **⑥ Warm Barrier** | 调度层 | `SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER=1` 等 | turn0 prefix + turn≥1 graft 协调 |
| **⑦ Workload** | Geo3K refocus | `geo3k_refocus_agent` multi-turn | exact/diversified/stress 变体 |

**文档中 ~46% turn1 加速**：是 **barrier + prefix + KV graft + M2 skip 整包** 相对 vanilla `off`，不是单独 KV graft 的净收益。

---

## 2. 指标与口径（必读）

### 2.1 Trainer 计时（`timing_s/*`，来自 console log）

| Key | 含义 | 归属 |
|-----|------|------|
| `timing_s/step` | 整步墙钟 | 全部 |
| **`timing_s/gen`** | rollout 生成阶段墙钟（等待所有 AgentLoopWorker 完成 `generate_sequences`） | **Rollout** |
| `timing_s/old_log_prob` | actor 重算 old log prob（仅 `MODES=recompute`） | Training |
| `timing_s/update_actor` | actor 前向 + 反向 + optimizer | Training |
| `timing_s/update_weights` | 权重同步回 rollout engine | Training/衔接 |
| `timing_s/adv` | advantage 计算 | Training |
| `timing_s/reward` | reward 计算 | 通常 ≈0 |

**注意**：`gen` ≠ 论文 EPD 里的完整 rollout stage；不包含 reward/ref 等若未计入 timer 的部分。

### 2.2 Per-request Rollout（CSV `generate_e2e_ms`）

- 文件：`verl_sglang_generate_log_<suffix>.csv`
- 列：`generate_e2e_ms`, `agent_turn`, `queue_time_ms`, `prefill_launch_latency_ms` 等
- **单条请求**从进入到返回的墙钟；与 `gen` 关系：`gen` ≈ 并行调度后的整步 rollout 墙钟，**远小于** `n_requests × mean(e2e)`

### 2.3 其他 CSV

| 文件 | 内容 |
|------|------|
| `vision_encoder_log_*.csv` | ViT encode 耗时、`cached_image_features` |
| `model_forward_log_*.csv` | LLM EXTEND/DECODE forward |
| `cacheblend_barrier_log_*.csv` | barrier 等待、donor/recipient 事件 |
| `e2e_module_breakdown_*_summary.csv` | per-request 模块分解汇总 |

### 2.4 `gen` vs `e2e` 为何差很多（示例：clean_0702b off）

- `gen` ≈ **35.4s**（4 worker 并行后的整步 rollout 墙钟）
- `generate_e2e_ms` 均值 ≈ **5788ms**（单请求）
- 每 step 约 512 条请求（64×4×2 turns）；若串行 ≈ 1480s，实际并行后 ≈ 35s

**看 rollout 优化**：以 **`generate_e2e_ms`（尤其 turn1）为主**，`gen` 为辅。

---

## 3. MODES 说明

`run_geo3k_training_bypass_ab.sh` 中 `MODES` 只影响 **rollout 之后 training 如何处理 old_log_prob**，不是 rollout 优化。

| Mode | `calculate_log_probs` | `bypass_mode` | 说明 |
|------|----------------------|---------------|------|
| **recompute** | False | False | 标准 PPO：rollout 后 actor **再前向**算 `old_log_prob` |
| **bypass** | True | True | rollout 带出 log_prob，training **跳过重算**（省 ~8s training，非 rollout 优化） |
| **audit** | True | True | bypass + 额外 audit 一遍数值差异 |

**Rollout 对比实验**：两臂应统一 `MODES=recompute`，只改 `CACHEBLEND_SELECTOR` 等 rollout 栈。

---

## 4. 标准实验配置

### 4.1 Geo3K Refocus Exact（主 workload）

| 项 | 值 |
|----|-----|
| 模型 | `Qwen/Qwen2.5-VL-7B-Instruct` |
| GPU | 4× H100 80GB（常用 `CUDA_VISIBLE_DEVICES=2,3,6,7` 或 `4,5,6,7`） |
| batch | `train_batch_size=64`, `rollout.n=4` |
| agent | `geo3k_refocus_agent`, `num_workers=4`, multi-turn 2×2 |
| prompt/response | `max_prompt_length=8192`, `max_response_length=1024` |
| rollout 引擎 | SGLang, `tensor_model_parallel_size=1`, `enforce_eager=True` |
| 训练 | FSDP, `MODES=recompute`, GRPO |

### 4.2 clean_0702b baseline（历史重 run，training 极慢）

见 [§5](#5-baseline-与-training-占比分析)。关键：**`param_offload=True` + `optimizer_offload=True` + `ppo_max_token_len_per_gpu=4096`**，且磁盘/Ray spilling 可能放大耗时。

### 4.3 step_gain 复测脚本（`run_step_gain_ab_0702.sh`）

**原则（重要）**：

- 优化目标是 rollout 里的 **E+P**（ViT + LLM prefill / KV graft），**不是**拉长 decode。
- **禁止**通过增大 `max_response_length` 来“做漂亮”的 rollout 占比——那测不到 CacheBlend 价值。
- 接受 Geo3K 天然的短答案（`max_response_length=1024`）+ 多轮 refocus 图（E+P 重）。
- 要做的是：**消除异常的 training 膨胀**（如磁盘满导致 Ray spilling），让比例回到合理区间。

脚本配置：

| 项 | 值 | 说明 |
|----|-----|------|
| `TOTAL_STEPS` | 2 | step1 预热，看 step2 |
| `MODES` | recompute | 两臂相同 |
| `max_response_length` | **1024（默认，不改）** | 不人为拉长 decode |
| `param_offload` | **True（默认，不改）** | 与前作 formal_log 一致 |
| `ACTOR_MAX_TOKEN_LEN` | 8192 | 比 clean_0702b 的 4096 更正常 |
| `RAY_TMPDIR` | `/dev/shm/rsg` | 避免 98% 满盘 spilling |
| 变量 | 仅 rollout stack | off vs kvdev+barrier |

---

## 5. Baseline 与 Training 占比分析

### 5.1 clean_0702b off recompute（5 steps 均值）

**Log**：`profile_logs_geo3k_training_ab/geo3k_refocus_exact_clean_0702b_2367_off_slotslast_fa0_cp0_recompute_bs64_n4.log`

| 模块 | 时间 | 占 step |
|------|------|---------|
| **update_actor** | 183.4s | **47.4%** |
| **update_weights** | 84.7s | 21.9% |
| **old_log_prob** | 82.7s | 21.4% |
| **gen (rollout)** | 35.4s | **9.2%** |
| **step 总计** | 386.6s | 100% |

**Rollout bucket**（gen+reward+dump）≈ 9.2%；**Training bucket** ≈ 90.8%。

### 5.2 同配置 rollout e2e（CSV）

| | 均值 |
|--|------|
| 全部 | 5788ms |
| turn0 | 7818ms |
| turn1 | 3759ms |

### 5.3 为何 training 占比这么悬殊？（分清「结构」与「异常」）

**两类原因，不要混为一谈：**

| 类型 | 原因 | 能否通过拉长 decode「修复」？ |
|------|------|------------------------------|
| **结构性** | Geo3K：prompt 8192 + **response 1024 短**；update_actor 重算全序列；VLM 7B | **否** — 这是我们优化的 E+P workload，decode 短是任务本身 |
| **异常性** | clean_0702b step **387s** 远超同配置 bypass run（**37s**）；磁盘 98% 满 → Ray spilling；GPU 争抢 | **否** — 应修环境，不是改 workload |

**param_offload 是不是根因？——不是。**

前作 `formal_gsm8k_qwen3-4b-instruct`（SGLang）同样 `param_offload=True`、`optimizer_offload=True`：

- step **129s**，gen **59s（46%）**，update_actor **35.6s（28%）**

前作 rollout 重，是因为 **`max_response_length=8192`（长 decode）**，不是因为没有 offload。

我们 Geo3K 即使 offload 相同，decode 天然短 → rollout 占比会低于 GSM8K。**这是 workload 差异，不是配置错误。**

我们**不接受**靠把 decode 拉到 8192 来「对齐比例」——那优化的是 decode，不是 E+P。

**我们接受的做法**：

1. 修异常 training（`RAY_TMPDIR=/dev/shm`、空卡、避免 spilling）→ 参考 `training_bypass_4g` ~37s/step  
2. 保持 Geo3K refocus 原生形态（短 response + 多图 E+P）  
3. 用 CSV 的 **E/P/D 分解** + turn1 `generate_e2e_ms` 看 rollout 优化收益  
4. 整步看 `gen` 是否随 E+P 优化下降（即使占比仍 <50%，只要 gen 绝对值下降且 step 传导即可）

### 5.4 对比：training_bypass_4g recompute（同 repo，step ≈ 37s，更可信 baseline）

| 模块 | off | kvdev |
|------|-----|-------|
| step | 36.6s | 39.0s |
| gen | 10.5s | 10.1s |
| update_actor | 15.5s | 17.9s |
| old_log_prob | 8.0s | 8.4s |

说明 **387s 的 clean_0702b 含异常/重配置因素**，不是「同一 verl 必然 training-heavy」的唯一形态。

---

## 6. Rollout 优化结论（2 卡 vs 4 卡）

### 6.1 2 卡 — 仅 rollout smoke（`bs4×n4`，32 req）

| 配置 | e2e 均值 | turn0 | turn1 |
|------|----------|-------|-------|
| off | 1857ms | 2458ms | 1256ms |
| kvdev | 1025ms (**-45%**) | 912ms | 1138ms (**-9%**) |
| cos | 948ms (**-49%**) | 842ms | 1055ms |

**无 full-step timing**；barrier 未单独验证。turn0 降幅大（prefix/Radix 暖 cache），turn1 收益较小。

### 6.2 4 卡 — rollout only（`semantic_4g_bs64n4`，512 req）

| 配置 | e2e 均值 | turn0 | turn1 |
|------|----------|-------|-------|
| off | 4353ms | 5091ms | 3614ms |
| kvdev | 2074ms (**-52%**) | 2148ms | 2000ms (**-45%**) |
| cos | 2020ms (**-54%**) | 2056ms | 1984ms (**-45%**) |

**~46% turn1 加速**对应此口径（4 卡 rollout vs off）。

### 6.3 4 卡 — full step + recompute（`training_bypass_4g`，10 steps）

| | off | kvdev | rollout 变化 |
|--|-----|-------|-------------|
| gen | 10.6s | 10.1s | -5% |
| step | 28.5s | 28.8s | ≈0% |
| e2e 均值 | 2452ms | 1464ms | **-40%** |
| turn1 e2e | 2813ms | 1776ms | **-37%** |

**结论**：per-request e2e 收益大，但 `gen`/整 step 因 training 占主导而几乎不变。

---

## 7. 各优化项逐项结论

| 优化项 | 2 卡 | 4 卡 rollout | 4 卡 full-step | 建议 |
|--------|------|--------------|----------------|------|
| **① KV graft (kvdev)** | ✅ -45% e2e | ✅ turn1 -45% | gen -5%, step≈0 | **保留** |
| **① cos selector** | ✅ | ≈ kvdev | — | 可选 |
| **② slotsall** | 无数据 | turn1 常回退 | cp5 AB 更慢 | **用 slotslast (-1)** |
| **③ fast_apply (fa1)** | — | turn1 可能更慢 | step +29% vs fa0 | **关 (fa0)** |
| **④ compact prefill (cp1)** | — | 混合/负 | clean recompute 灾难 | **关 (cp0)** |
| **⑤ Warm barrier** | 未验证 | turn1 -35~45% | 嵌入 production path | **开** |
| **⑤ bounded wait 50ms** | — | 比 0ms 慢 ~40% | — | **MAX_WAIT_S=0** |
| **⑥ Geo3K exact on** | — | -24% vs off | — | workload 配套 |
| **Chart 正常尺度** | — | -9%, turn1≈0 | — | 几乎无 wall-clock 收益 |

### Barrier 专项（4 卡）

| 对比 | e2e | turn1 |
|------|-----|-------|
| warmbarrier off → on (stress bs32) | 4090→3179ms | 4078→2233ms (**-45%**) |
| nobarrier → turn1barrier (diag) | 1977→1506ms | 2556→1662ms (**-35%**) |

---

## 8. 推荐配置与复测脚本

### 8.1 当前最佳 Rollout 组合（`MODES=recompute`）

```bash
CACHEBLEND_SELECTOR=kvdev
CACHEBLEND_IMAGE_SLOTS=-1          # slotslast
CACHEBLEND_FAST_APPLY=0
CACHEBLEND_COMPACT_PREFILL=0
SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER=1
SGLANG_VLM_CACHEBLEND_PREFIX_WARMUP_BARRIER=1
SGLANG_VLM_CACHEBLEND_TARGET_TURNS=1
SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_WAIT_POLICY=bounded
SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_MAX_WAIT_S=0
MODES=recompute
```

### 8.2 Baseline 对照

```bash
CACHEBLEND_SELECTOR=off
CACHEBLEND_IMAGE_SLOTS=-1
CACHEBLEND_FAST_APPLY=0
CACHEBLEND_COMPACT_PREFILL=0
# 不设 barrier env
MODES=recompute
```

### 8.3 一键 AB 脚本

**路径**：`graftrl/verl/run_step_gain_ab_0702.sh`

```bash
cd /workspace/repo/graftrl/verl
bash run_step_gain_ab_0702.sh
```

- **不改** `max_response_length`（保持 Geo3K 短答案 + E+P 形态）
- **不关** offload（与前作 formal_log 一致）
- **修环境**：`RAY_TMPDIR=/dev/shm/rsg`，避免满盘 spilling
- 两臂之间自动 `ray stop` + sleep 8

**期望**：step 量级接近 `training_bypass_4g`（~30–40s），gen 占 step **~25–35%**（仍可能低于前作 gsm8k 的 46%，因为 decode 短——这是正常的）。  
**成功标准**：kvdev 臂的 **turn1 e2e / E+P / gen** 相对 off 下降，且 **step 有传导**（不必 gen 占 50%）。

### 8.4 其他相关脚本

| 脚本 | 用途 |
|------|------|
| `run_geo3k_training_bypass_ab.sh` | 单次/多 mode AB（recompute/bypass/audit） |
| `run_geo3k_refocus_profile.sh` | rollout profile 主入口 |
| `run_geo3k_cacheblend_semantic_ab.sh` | semantic gate off/kvdev/cos |
| `run_clean_ab_0702.sh` | 旧 clean AB（ARM B 为 slotsall+fa1+cp1，已证负） |
| `examples/profile/shared/analysis/analyze_profiling_logs.py` | CSV 汇总 + report |

---

## 9. 负结果与踩坑记录

| Run / 配置 | 现象 | 原因推测 |
|------------|------|----------|
| `warmbarrier_retry1` + slotsall+fa1+cp0 recompute | step 697s, e2e 22214ms | 整包配置/协调问题 |
| `clean_0702b` kvdev+slotsall+fa1+cp1 recompute | step 577s, e2e 10633ms | 同上 |
| kvdev+fa0 ab4g（无 barrier 协调） | turn1 比 off **慢 30%** | recipient 先到 |
| slotsall+kvdev cp0 bypass | turn1 4884ms vs off 3189ms | slot 扩大无 barrier |
| `MAX_WAIT_S=0.05` waitsweep | 比 0ms 慢 | bounded wait 过长 |
| `run_clean_ab_0702.sh` ARM B | fa1+cp1+slotsall | 非当前最佳组合 |
| 磁盘 98% 满 + `/tmp/r4g` Ray | spilling、step 暴涨 | 用 `/dev/shm` 短路径 |

---

## 10. 与前作 formal 实验对比（`verl_sglang/verl_customize/formal_*`）

> **正确定位**：前作数据在 `formal_log/`、`formal_log_2gpu/`、`formal_exp_data_mfu/`，不是仅 `run_*_oyhh.sh` 两个脚本。  
> 前作 **已使用 SGLang**（`actor_rollout_ref.rollout.name=sglang`），含完整 `timing_s/*`。

### 10.1 前作代表 run（4 卡，gsm8k qwen3-4b-instruct）

| 模块 | 时间 | 占 step |
|------|------|---------|
| **gen** | 59.0s | **46%** |
| update_actor | 35.6s | 28% |
| ref | 18.2s | — |
| old_log_prob | 12.2s | — |
| **step** | 129.1s | 100% |

启动参数摘录：`max_response_length=8192`, `max_prompt_length=512`, `rollout.n=4`, `rollout TP=4`, `gpu_mem_util=0.2`, **`param_offload=True`**, **`optimizer_offload=True`**, SGLang。

### 10.2 前作 vs Geo3K：为何 rollout 占比差很多

| 项 | 前作 formal（rollout-heavy） | Geo3K refocus（E+P 优化目标） |
|----|------------------------------|-------------------------------|
| 模型 | qwen3-4b 文本 | Qwen2.5-VL-7B |
| **max_response_length** | **8192** | **1024** ← 结构差异，不应对齐 |
| max_prompt_length | 512 | 8192 + 多图 |
| offload | True | True |
| gen 占 step | 26%~81%（多数 40%+） | ~9%（clean_0702b 异常 run）~28%（bypass 正常 run） |

**结论**：前作 rollout 重，主因是 **长 decode（8192）**；我们 Geo3K 刻意保持短答案 + 图像 E+P，**不应靠拉长 decode 来对齐比例**。

### 10.3 整步收益实验应对齐什么

| 应对齐 | 不应对齐 |
|--------|----------|
| SGLang + GRPO + recompute 路径 | 把 response 拉到 8192 |
| 消除 Ray spilling / 空卡环境 | 关 offload 仅为了做漂亮比例 |
| 两臂 training 配置完全一致 | 用 bypass mode 冒充 rollout 收益 |
| 看 E+P（vision_log + EXTEND）和 turn1 e2e | 只盯 decode token 数 |

### 10.4 前作数据路径

| 目录 | 内容 |
|------|------|
| `verl_sglang/verl_customize/formal_log/` | 68 条 console log，含 `timing_s/*` |
| `verl_sglang/verl_customize/formal_log_2gpu/` | 2 卡对照 |
| `verl_sglang/verl_customize/formal_exp_data_mfu/` | `inference_step_log_*.csv`（SGLang forward 粒度） |
| `verl_sglang/verl_customize/formal_exp_data_decoding_length/` | 实际 decode 长度分布 |

---

## 11. 日志文件结构

每个 `profile_logs_<name>/` 通常含：

```
profile_logs_<name>/
  verl_sglang_generate_log_<suffix>.csv    # per-request rollout e2e
  vision_encoder_log_<suffix>.csv
  model_forward_log_<suffix>.csv
  cacheblend_barrier_log_<suffix>.csv      # 若启用 barrier
  e2e_module_breakdown_<suffix>_summary.csv # 若有 analyze 导出
  image_dump_<suffix>/                      # PNG，占空间大，分析可不要
  rollout_data_<suffix>/                    # JSONL rollout 原文
```

Console 全量 log：`profile_logs_geo3k_training_ab/<suffix>.log`（含 `timing_s/*`）

**profile_gallery/**：数据集样本 HTML 预览（非 profiling 计时），1.6M。

---

## 12. 实验 Run 索引（按类别）

归档包内约 **110** 个 `profile_logs_*` 目录 + **48** 条带 `timing_s` 的 training console log。

### 12.1 Geo3K Refocus Exact（主实验）

- `profile_logs_geo3k_refocus_exact_*`：semantic、training_bypass、clean_0702、ab4g、cp5、waitsweep、rollout_diag、warmbarrier、research_* 等
- 命名模式：`geo3k_refocus_exact_<RUN_TAG>_<selector>_slots<last|all>_fa<N>_cp<N>_<recompute|bypass>_bs64_n4`

### 12.2 Geo3K 其他变体

- `profile_logs_geo3k_refocus_diversified_*`
- `profile_logs_geo3k_refocus_stress_*`
- `profile_logs_geo3k_refocus_exact_cacheblend_on`
- `profile_logs_geo3k_full_old`, `profile_logs_geo3k_text_only`

### 12.3 Chart / VTool / 其他 workload

- `profile_logs_vtool_chart_*`（~833M）
- `profile_logs_refocus_chart_*`
- `profile_logs_deepeyes_*`, `profile_logs_sokoban_*`
- `profile_logs_cacheblend_stress_*`

### 12.4 Training AB 控制台日志目录

`profile_logs_geo3k_training_ab/*.log` — 所有 `run_geo3k_training_bypass_ab.sh` 运行的 tee 输出。

---

## 13. 关键 Run 数字表

### 13.1 Full-step training（慢→快排序，节选）

| step | gen | update_actor | old_log_prob | Run（后缀） |
|------|-----|--------------|--------------|-------------|
| 697s | 82.9s | 323.9s | 148.3s | warmbarrier_retry1 kvdev slotsall fa1 cp0 recompute |
| 577s | 60.5s | 262.0s | 133.3s | clean_0702b kvdev slotsall fa1 cp1 recompute |
| 387s | 35.4s | 183.4s | 82.7s | **clean_0702b off slotslast recompute（baseline）** |
| 37s | 10.5s | 15.5s | 8.0s | training_bypass_4g off recompute |
| 29s | 10.6s | 15.4s | 0 | training_bypass_4g off bypass |
| 29s | 10.1s | 16.2s | 0 | **training_bypass_4g kvdev bypass（rollout e2e 最优之一）** |
| 36s | 10.1s | 17.9s | 8.4s | training_bypass_4g kvdev recompute |

### 13.2 Rollout e2e 关键对比

这里的 `CSV请求数` 是对应 generate log 里参与统计的请求行数，不是 `rollout.n=4`。

| e2e | turn0 | turn1 | CSV请求数 | 目录 |
|-----|-------|-------|-----------|------|
| 4353ms | 5091 | 3614 | 512 | semantic_4g off |
| 2074ms | 2148 | 2000 | 512 | semantic_4g kvdev |
| 2452ms | 2090 | 2813 | 5120 | training_bypass off bypass |
| 1464ms | 1152 | 1776 | 5120 | **training_bypass kvdev bypass** |
| 5788ms | 7818 | 3759 | 2560 | clean_0702b off recompute |
| 10633ms | 9934 | 11332 | 2560 | clean_0702b kvdev slotsall fa1 cp1（负） |
| 1977ms | 1397 | 2556 | 5120 | rollout_diag nobarrier |
| 1506ms | 1350 | 1662 | 5120 | **rollout_diag turn1barrier** |
| 4090ms | 4102 | 4078 | 256 | warmbarrier stress off |
| 3179ms | 4125 | 2233 | 256 | **warmbarrier stress on** |

---

## 14. 磁盘清理与归档

### 14.1 已打包

```text
/workspace/repo/graftrl/verl/profile_logs_bundle_0702.tar.gz  (~4.2G)
```

本地下载（用户 SSH）：

```bash
scp -P 2230 puzzrl_hehua@picasso-hopper.ucsd.edu:/workspace/repo/graftrl/verl/profile_logs_bundle_0702.tar.gz ~/Downloads/
```

### 14.2 下载后可删（约 9G+）

```bash
cd /workspace/repo/graftrl/verl
find . -maxdepth 1 -type d -name 'profile_logs_*' -exec rm -rf {} +
rm -rf profile_gallery
# 保留 profile_logs_bundle_0702.tar.gz；本地确认后再删 tar
```

### 14.3 其他可清理（不影响跑 Geo3K 脚本）

| 路径 | ~大小 |
|------|-------|
| `_archive_verl_vision/` | 2.3G |
| `/workspace/backup_verl_vision_20260627.tar.gz` | 2.0G |
| `code_generation_lite/test*.jsonl` | 4.2G |
| `/workspace/tmp`, `/tmp/r4g*` | ~1.4G |
| `data/refocus_chart_raw/`（不跑 chart 时） | 1.1G |

---

## 15. 分析工具命令

### 15.1 从归档解压

```bash
tar xzf profile_logs_bundle_0702.tar.gz -C /path/to/restore/
```

### 15.2 单 run 分析报告

```bash
cd graftrl/verl
python3 examples/profile/shared/analysis/analyze_profiling_logs.py \
  --log-dir profile_logs_geo3k_refocus_exact_<name> \
  --suffix <suffix> \
  --report
```

### 15.3 快速 grep step timing

```bash
grep -E 'timing_s/(step|gen|old_log_prob|update_actor|update_weights)' \
  profile_logs_geo3k_training_ab/<run>.log
```

### 15.4 Per-request e2e 均值（Python）

```python
import csv, statistics as st
from collections import defaultdict
path = "profile_logs_.../verl_sglang_generate_log_....csv"
bt = defaultdict(list)
with open(path) as f:
    for row in csv.DictReader(f):
        if row.get("generate_e2e_ms"):
            bt[int(row["agent_turn"])].append(float(row["generate_e2e_ms"]))
allv = [x for v in bt.values() for x in v]
print("ALL", st.mean(allv), {t: st.mean(v) for t,v in bt.items()})
```

---

## 附录 A：环境变量速查

| 变量 | 推荐值 | 说明 |
|------|--------|------|
| `CACHEBLEND_SELECTOR` | `off` / `kvdev` | graft 总开关+选择器 |
| `CACHEBLEND_IMAGE_SLOTS` | `-1` | last=refocus 图 only |
| `CACHEBLEND_FAST_APPLY` | `0` | Method B |
| `CACHEBLEND_COMPACT_PREFILL` | `0` | Method C |
| `SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER` | `1` | turn≥1 barrier |
| `SGLANG_VLM_CACHEBLEND_PREFIX_WARMUP_BARRIER` | `1` | turn0 prefix |
| `SGLANG_VLM_CACHEBLEND_TARGET_TURNS` | `1` | 仅 turn1 graft |
| `SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_MAX_WAIT_S` | `0` | 不等 donor |
| `RAY_TMPDIR` | `/dev/shm/rsg` 等短路径 | 避免满盘 spilling |
| `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` | `1` | shell env，非 Hydra |

---

## 附录 B：文档修订记录

| 日期 | 说明 |
|------|------|
| 2026-07-02 | 初版：汇总 profile_logs 归档、2卡/4卡结论、baseline 分析、推荐配置、`run_step_gain_ab_0702.sh` |
| 2026-07-02 | 修正：前作定位到 `formal_log/`（SGLang + 完整 timing）；明确不靠拉长 decode 做比例；step_gain 改为修异常 training、保持 offload |

---

*更完整的项目历程见：`graftrl/docs/GraftRL_项目全历程.md`*  
*Barrier 设计见：`graftrl/docs/VLM_CACHEBLEND_WARMUP_BARRIER.md`*
