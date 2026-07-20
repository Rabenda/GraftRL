# GraftRL 项目全历程（从建仓到现在）

> 本文档把原先分散在多份英文/技术 doc 里的内容，合并成**一份中文说明**。  
> 读者不需要先读代码；括号里给出**文件路径和函数名**，方便你或合作者定位。

---

## 0. 这个项目到底在解决什么问题

训练视觉语言模型（VLM）做强化学习（GRPO）时，同一条样本会采样 **4 条并行分支**（`rollout.n=4`）。  
在 **Chart Refocus** 任务里，每条分支会：

1. 先看一张图表原图，模型回答一轮（turn0）；
2. 调用工具在图上画框/高亮，得到一张 **refocus 新图**（和原图很像，但不完全一样）；
3. 带着原图 + 新图再问一轮（turn1）。

我们发现：**同组 4 条分支的 refocus 图往往非常相似**，但系统仍然每条分支都完整跑一遍视觉编码和语言模型预填充（prefill），很慢。

**GraftRL 的目标**：在「图很像、但不是完全相同的公共前缀」的情况下，让同组后面的分支**少算一点**，同时尽量不改变模型输出。

---

## 1. 仓库是怎么建起来的

### 1.1 上游起点（回溯用）

GraftRL 不是从零写的，而是在两个开源项目之上改出来的：

| 组件 | 上游地址 | 基于的 commit（完整 SHA） | 一句话 |
|------|----------|---------------------------|--------|
| verl（训练/rollout 框架） | github.com/verl-project/verl | `802256a79d676215740e9545cf79be816afea78c` | 2026-05-15 的 main |
| SGLang（推理引擎） | github.com/sgl-project/sglang | `0189f41c30ede088a040a711a384f3024b8d7af5` | 2026-01-23 的 main |

在 GraftRL 自己的 git 里，这个干净起点打了 tag：**`v0.0-baseline`**（commit `219b32b`）。  
之后所有改动都在此之上分批提交（v0.1～v0.6）。

### 1.2 目录结构

```
graftrl/
  verl/     ← 改过的 verl（agent、profiling 脚本、日志）
  sglang/   ← 改过的 SGLang（缓存与 KV 复用逻辑）
  docs/     ← 文档（本文档为主）
```

旧的 `verl_vision`、`sglang_vision_profile` 已改名为 `_archive_*` 归档，**以后只在 graftrl 里开发**。

---

## 2. 第一阶段：先把「慢在哪里」测清楚（v0.1）

### 2.1 做了什么

我们给一次 rollout 加了 **三类 CSV 日志**，把整条链路拆开：

| 日志文件 | 记录什么 |
|----------|----------|
| `verl_sglang_generate_log_*.csv` | 从发起请求到拿到回复的总时间、排队时间等 |
| `vision_encoder_log_*.csv` | 视觉编码器（ViT）跑了多久、几张图 |
| `model_forward_log_*.csv` | 语言模型每一轮 forward 算了多少 token、多久 |

并写了汇总脚本，把耗时拆成 **E（编码图）/ P（预填充）/ D（解码）** 三段。

### 2.2 代码在哪里

- 在 verl 侧写入 generate 日志：`verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py`（函数 `_append_verl_sglang_generate_log`、`_prompt_image_text_token_stats`）
- 在 SGLang 侧写入 forward 日志：`sglang/python/sglang/srt/model_executor/model_runner.py`（函数 `_append_inference_step_log`）
- 汇总分析：`verl/examples/profile/analyze_profiling_logs.py`
- Geo3K 单图测速脚本：`verl/examples/profile/workloads/geo3k/run_geo3k_full_profile.sh`
- 补充：分析 ViT 是「冷启动」还是「跟 batch 挤在一起」：`verl/examples/profile/shared/analysis/analyze_vision_cold_warm.py`

### 2.3 结论（动机）

- 带图的 rollout 里，**等待和同卡排队**往往占很大比例；
- ViT 单次只有几十毫秒，但会**挡住**后面的 LLM prefill 启动；
- 真正吃时间的是 **LLM 的 extend/prefill**（把新 token 算进 KV 缓存的那一步）。

---

## 3. 第二阶段：证明「图确实冗余、但现有缓存用不上」（v0.2）

### 3.1 做了什么

1. **纯文本对照实验**：同样长度的纯文本 prompt，对比带图 rollout，说明「图」带来了额外成本。  
   （脚本：`verl/examples/profile/workloads/geo3k/run_geo3k_text_only_profile.sh`）

2. **离线相似度分析**：对 refocus 前后、同组不同分支的图，算像素差异和 ViT 输出向量的相似度。  
   （脚本：`verl/examples/profile/analyze_similarity_unified.py`）

3. **Chart Refocus 在线 agent**：让模型真的走「看图 → 工具改图 → 再看图」多轮流程。  
   - Agent 逻辑：`verl/examples/profile/shared/agent/vtool_agent_loop.py`  
   - 工具实现：`verl/examples/profile/shared/agent/vtool_refocus_tools.py`  
   - 曾修过一个 bug：oracle 强制 refocus 开关没传到 Ray worker，导致 turn1 几乎不成功；改为通过 yaml 传 `force_oracle_refocus: true`。

4. **探索性 workload**（Sokoban、DeepEyes、早期 dummy 实验）：`verl/examples/profile/workloads/sokoban/`、`deepeyes/`、`archive/`

### 3.2 关键发现

| 发现 | 含义 |
|------|------|
| SGLang 自带的「整图一模一样才命中」缓存，在 GRPO 里几乎 **0 命中** | 主要因为 4 张卡、请求被散到不同进程（cross-replica） |
| refocus 后 **最终视觉 token** 约 **76%** 在相似度 ≥0.90 时几乎不变 | 说明冗余真实存在 |
| turn0 四分支原图 **字节级相同** | 理论上最容易复用，但在线仍命不中 |

---

## 4. 第三阶段：在 ViT（看图编码器）里试「局部复用」——未成功（v0.3）

### 4.1 我们想试什么

既然图只有一小部分变了，能不能在 **ViT 内部**只重算变了的块，别的块抄同组第一条分支的结果？

我们依次试了：

| 方案 | 白话 | 结果 |
|------|------|------|
| Window 复用 | 按「窗口」整块判断像不像 | 复用率太低 |
| Merged token 替换 | ViT **全部算完**后，用 donor 的 token 覆盖相似的 | 能证明像，但 **ViT 时间一点没省** |
| Token sparse | 在 ViT **前几层**真的跳过相似 token 的 MLP | 只复用约 **3.5%** token，反而 **慢一倍**（调度开销 > 收益） |

### 4.2 历史算法（ViT 路线，已删除）

**当前状态**：这条路径的实测均慢于 baseline，实现和
`SGLANG_GRPO_SIM_CACHE` 开关已于 2026-07-19 删除。下文仅保留为负结果记录。

**核心流程**（白话）：

1. 同组第一个编码这张图的 branch 当 **donor（供体）**，把 ViT 中间结果存进组内缓存（历史实现已删除）。

2. 同组后面的 branch 当 **recipient（受体）**：  
   - 原图 slot：必须像素几乎完全一样才跳过 ViT；  
   - refocus slot：算 raw patch 相似度，够高就整图 embedding 直接抄 donor。

3. 「真正少算」的 token_sparse 曾在 ViT 前 7 层里对标记为「可复用」的 token 跳过 FFN（历史实现已删除）。

### 4.3 为什么放弃 ViT 路线作为主方案

一句话：**相似度高的地方 ViT 已经算完了；能提前省算的地方相似度又太低。**

详细数据见下文「实验结论表」；这条线的价值是**负结果**，说明不能把「merged token 很像」直接等同于「ViT 能加速」。

---

## 5. 第四阶段：改到 LLM 预填充里做 KV 复用——GraftRL 主线（v0.4～v0.5）

师兄建议：**别在 ViT 里硬抠了，改在 LLM 已经拿到图 token 之后，复用 KV（键值缓存）**。

### 5.1 为什么 SGLang 自带的前缀缓存不够

turn1 的一条 prompt 可以想象成四段：

```
[ 系统 + turn0 原图 ]     ← 四分支相同，前缀缓存能命中
[ turn0 模型回复 ]       ← 四分支不同，从这里开始前缀对不上
[ refocus 图 ]           ← 在「不同回复」后面，不是公共前缀
[ turn1 文字说明 ]
```

SGLang 的 **RadixCache（前缀树缓存）** 只能从开头匹配**完全相同**的 token。  
refocus 图再像，也因为排在「不同的 turn0 回复」后面而 **无法被前缀缓存复用**。

我们要做的是：**内容很像、但不在前缀位置** 的那一段图 token 的 KV 复用（思路来自论文 CacheBlend，我们叫它 **GraftRL / visual KV grafting**）。

### 5.2 角色：donor 和 recipient

| 角色 | 谁 | 干什么 |
|------|-----|--------|
| **Donor** | 同组里**第一个**做完 turn1 prefill 的分支 | 正常全算；额外把 refocus 图那一段的 **每层 K、V** 存起来 |
| **Recipient** | 同组**后面**的分支 | 尽量用 donor 的 K/V；只对少量「不放心」的 token 重新算 |

分组键（同组判定）：训练 step + `agent_uid` + turn + 哪张图 + 图网格形状  
（`sglang/python/sglang/srt/mem_cache/vlm_cacheblend.py`，函数 `build_group_key`）

### 5.3 算法实现（按步骤，白话 + 代码定位）

**总开关**：`SGLANG_VLM_CACHEBLEND=1`。当前只保留 LLM prefill/decode 主线，不再有自定义 ViT cache 开关。

#### 步骤 1：rollout 时带上「我是谁、第几轮」

verl 发请求给 SGLang 时，登记：`agent_uid`（同组 id）、`agent_turn`（第几轮）、`global_step`。  
（`verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py`，`register_request_meta`）

#### 步骤 2：判断当前请求是 donor 还是 recipient

进入 turn1 时查组内是否已有完整 donor KV：  
- 没有 → 本次是 **donor**；  
- 有 → 本次是 **recipient**。  
（`vlm_cacheblend.py`，函数 `resolve_request_context`，约 1392 行：`role = "recipient" if donor.complete else "donor"`）

#### 步骤 3：找到 prompt 里「refocus 那张图」对应的 token 区间

在整段 input 里找**最后一张图**占用的 token 位置（默认 refocus 是 turn1 的第二张图）。  
用 image 占位符的 pad 值扫描，而不是死盯某个 token id。  
（`vlm_cacheblend.py`，函数 `_image_span_count_from_req`、`_token_span_count`；  
`sglang/python/sglang/srt/models/qwen2.py`，函数 `_cacheblend_locate_image_tokens`）

#### 步骤 4：Donor 在完整 prefill 后，把这段图的每层 K/V 拷出来存好

不改 attention 内核，只从 KV 池里**只读拷贝**。  
（`qwen2.py`，函数 `_maybe_cacheblend_after_full_prefill` → `capture_donor_kv`；  
存储：`vlm_cacheblend.py`，类 `DonorKVStore`、方法 `record_layer`）

#### 步骤 5：Recipient 预填充前，制定「哪些 token 抄 donor、哪些重算」

检查：grid 是否一致、图 token 个数是否一致、位置是否对得上（或用 rerotate 对齐）。  
然后对 refocus 图里约 **85%** token 标记为复用，约 **15%** 标记为重算（默认比例，可调环境变量）。  
（`vlm_cacheblend.py`，函数 `build_recipient_kv_blend_plan`、`select_recompute_tokens`）

#### 步骤 6：Recipient 真正少算——在 attention 之前换掉 KV，并跳过部分层计算

1. 每层 attention 算之前，把可复用位置的 KV 池内容换成 donor 的（并对 K 做位置旋转对齐，若两分支图 token 绝对位置不同）。  
   （`vlm_cacheblend.py`，`apply_recipient_kv_blend_for_layer`；  
   `sglang/.../flashattention_backend.py` 等在 extend 路径里调用）

2. 对标记为「复用」的图 token，跳过 QKV 投影、MLP、以及该 token 作为 query 的 attention 计算。  
   （`qwen2.py` 里 `Qwen2Attention.forward`、`Qwen2MLP.forward` 中的 skip 分支；  
   环境变量 `SGLANG_VLM_CACHEBLEND_SKIP_REUSE_QKV_PROJ` 等）

3. 每层 forward 开始前准备好 plan：  
   （`qwen2.py`，`_cacheblend_prepare_recipient_fast_path`）

#### 步骤 7：打日志，方便确认「到底有没有复用」

`model_forward_log_*.csv` 里会有：`cacheblend_role`、`cacheblend_reused_tokens`、`cacheblend_fallback_reason` 等。  
（`model_runner.py` 写入；汇总：`verl/examples/profile/shared/analysis/summarize_cacheblend_probe.py`）

### 5.4 质量验证（positive4602）

用 16 条 turn1 分支、温度 0，对比开关 on/off：  
- **输出文本：16/16 完全一致**；  
- 宽松答案准确率：on/off 都是 16/16。  
（脚本：`verl/examples/profile/shared/analysis/analyze_cacheblend_ab.py`）

### 5.5 速度结果（诚实）

在**正常 Chart 尺度**（单图约 580 个图 token）上：  
- turn1 端到端中位数约 **2844 ms → 2834 ms（约 10 ms）**；  
- 机制上 recipient 确实复用了约 **493/580** 个图 token，但只占整次 extend 的一小部分，**ViT 和排队时间不变**。

---

## 6. 第五阶段：放大图 + 修 chunked prefill 的坑（v0.6）

### 6.1 为什么要单独造 stress 数据集

正常 Chart 上图 token 太少，就算复用 85% 的 refocus 图，端到端也快不明显。  
我们写了脚本把图和 bbox **放大 2 倍**，让单张图约 **5244** 个图 token，并把 `rollout.n` 提到 **8**，让 7 条 recipient 都能蹭 donor。  
（`verl/examples/profile/data_preprocess/chart/refocus_chart_cacheblend_stress.py`；  
运行：`verl/examples/profile/workloads/chart/run_cacheblend_stress.sh`）

### 6.2 遇到的 chunk 问题（白话）

turn1 prompt 很长（约 1 万 token）。SGLang 默认会把 prefill **切成多块**（chunked prefill）分批算。  
refocus 图落在后面某一块里，结果：

- donor 在某一块 forward 里只「看到」图的前半段（例如 2918 个 token），就误以为整张图只有这么长；
- recipient 在完整 prompt 里看到 5244 个 token → **个数对不上** → **整组复用失败**，日志里 `cacheblend_used=0`。

**修复**：跑 stress 时关闭 chunked prefill（`chunked_prefill_size=-1`），让 donor 一次 capture 完整图 span。  
（`run_cacheblend_stress.sh`；说明：`docs/VLM_CACHEBLEND_CHUNKED_PREFILL_FIX.md`）  
跑完后用脚本断言必须 `reused_tokens > 0`：  
（`verl/examples/profile/shared/analysis/assert_cacheblend_used.py`）

> 注：stress 在修 chunk 之后需要 GPU 重跑验证；若你还没重跑，以日志断言为准。

---

## 7. 第六阶段：Geo3K Refocus 多轮 workload（2026-06）

Chart 正常尺度端到端加速不明显后，我们换了一个**更贴近 refocus 多轮 RL** 的 workload：**Geo3K 几何题 + 程序化 refocus**。

### 7.1 三种变体

| 变体 | 含义 | 用途 |
|------|------|------|
| **exact** | 全画布确定性 refocus，ROI 不变 | 最佳情况复用（同位置 graft） |
| **diversified** | 同 ROI、不同视觉风格 | 测「图很像但不 exact」时的复用 |
| **stress** | 放大图 + 更长回复 | 放大 E+P 信号 |

数据与脚本：

- 数据生成：`verl/examples/profile/workloads/geo3k/prepare_refocus_data.sh`
- Agent 逻辑：`verl/examples/profile/workloads/geo3k/geo3k_refocus_agent_loop.py`
- 配置：`verl/examples/profile/workloads/geo3k/geo3k_refocus_agent_loop.yaml`
- 跑 profile：`verl/examples/profile/workloads/geo3k/run_geo3k_refocus_profile.sh`
- **baseline vs 优化对比（推荐给别人）**：`verl/examples/profile/workloads/geo3k/run_geo3k_rollout_demo.sh`
- rollout AB 底层：`verl/examples/profile/workloads/geo3k/run_geo3k_rollout_ab.sh`

refocus 图在 rollout 时由 PIL **在线生成**（不在 parquet 里）；dump 在 `profile_logs_*/image_dump_*`（`*_t0_i0_input.png` + `*_t1_i1_refocus.png`）。

### 7.2 实验规模约定

本项目 rollout profiling 的**最小有意义规模**是 **`TRAIN_BATCH_SIZE=64` × `ROLLOUT_N=4`**（64 组 × 4 分支 = 256 条 rollout/步）。更小 batch 只用于排查初始化/路由，**不算实验结论**。

### 7.3 早期 Geo3K 结果（CacheBlend on/off，bs64×n4）

| 变体 | off e2e mean | on e2e mean | Δ e2e | E+P (GPU) off → on |
|------|-------------|-------------|-------|---------------------|
| exact | 4306 ms | 3293 ms | **−23%** | 86% → 78% |
| diversified | 4194 ms | 3210 ms | **−23%** | 85% → 77% |
| stress r4096 | 7213 ms | 6149 ms | **−15%** | 88% → 87% |

共性：ViT encode 行数约 **490 → 122**（~75% 减少）；`batch_sync_wait` / `extend_wall` 在 on 时接近 0。

日志目录示例：`profile_logs_geo3k_refocus_exact` / `_cacheblend_on` 等（均在 `.gitignore`，本地保留）。

---

## 8. 第七阶段：Rollout 侧 donor-ready barrier（v0.7）

### 8.1 问题

`rollout.n=4` 时，同 `agent_uid` 的多条分支会**并发**打到 turn1。若 **recipient 比 donor 先到** SGLang，会出现 `donor_not_ready` → `cacheblend_used=0`，加速不稳定。

### 8.2 行为（两层 warmup key）

对每个 `(global_step, agent_uid, agent_turn)` 组：

1. **第一个**到的请求单独跑（barrier **donor**）；
2. 完成后标记 warmed，同组 sibling 并行（barrier **recipient**）；
3. 缺元数据 / 非目标 turn → **bypass**。

**Turn0** 与 **Turn≥1** 分开：

| Turn | warmup key 前缀 | 作用 |
|------|-----------------|------|
| turn0 | `prefix:` | 暖 SGLang RadixCache / 前缀路径；**不做** KV graft |
| turn≥1 | `cacheblend:` | donor 捕获 refocus 图 KV，recipient 复用 |

实现与文档：

- 代码：`verl/verl/workers/rollout/llm_server.py`（`LLMServerClient.generate`、`_vlm_cacheblend_warmup_key`）
- 文档：`docs/VLM_CACHEBLEND_WARMUP_BARRIER.md`
- 日志：`cacheblend_barrier_log_{suffix}.csv`（字段含 `barrier_role`、`barrier_wait_ms`、`server_call_ms`、`wait_policy`、`donor_ready`）

环境变量要点：

| 变量 | 默认 | 含义 |
|------|------|------|
| `SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER` | `1` | 总开关（CacheBlend 也需为 `1`） |
| `SGLANG_VLM_CACHEBLEND_PREFIX_WARMUP_BARRIER` | `1` | turn0 前缀 barrier |
| `SGLANG_VLM_CACHEBLEND_TARGET_TURNS` | `1`（可设 `all`） | 哪些 agent turn 走 CacheBlend KV graft |
| `SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_WAIT_POLICY` | 代码默认 `strict`；profile 脚本在 kvdev 时设 `bounded` | `strict` = 等到超时再 fail/fallback；`bounded` = 超时后 **fail-open** 继续发请求 |
| `SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_MAX_WAIT_S` | profile 脚本默认 `0.05`；**慢簇建议 `10`** | bounded 模式下 recipient 最多等 donor 的秒数 |
| `SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_TIMEOUT_S` | `300` | **strict** 模式下 recipient 最长等待秒数 |
| `SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_TIMEOUT_ACTION` | `fail` | strict 超时默认 **抛错**；设 `fallback` 才静默继续 |

### 8.3 v1 → v2：单进程锁 → 全局 Ray Coordinator

| 版本 | commit | 能力 |
|------|--------|------|
| **v1** | `bd3da72` | 每个 `LLMServerClient` 内 `asyncio.Lock`；适合 `agent.num_workers=1` |
| **v2** | `8e52bfe` + `dbaf089` | **`GlobalCacheBlendCoordinator` Ray actor**，跨多个 `AgentLoopWorker` 共享 barrier 状态 |

v2 额外能力：

- Sticky 路由键：`cacheblend_group:{training_global_step}:{agent_uid}` → 同 GRPO 组落同一 SGLang replica；
- `training_global_step` 经 `agent_loop.py` → `geo3k_refocus_agent_loop.py` → `async_sglang_server.register_request_meta` 透传；
- `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` 时，`worker.py` 修正 `LOCAL_RANK` 与可见 GPU 列表的映射；
- 离线校验：`verl/examples/profile/shared/analysis/validate_cacheblend_multiworker.py`；
- CPU 契约测试：`verl/tests/experimental/agent_loop/test_cacheblend_contract_on_cpu.py`（`dbaf089`）。

**仍 out-of-scope**：donor KV **不跨 SGLang replica** 复制；多 replica 故意拆组时无法复用。

### 8.4 Warm barrier 消融（stress, bs32×n4）

| 指标 | barrier ON | barrier OFF |
|------|-----------|-------------|
| e2e mean | **3179 ms** | 4089 ms（约慢 29%） |
| decode_wall % e2e | 30.3% | 6.7% |

目录：`profile_logs_geo3k_refocus_stress_warmbarrier_{on,off}_bs32_n4`。  
Barrier 同步 donor/recipient 后，recipient 少踩 `donor_not_ready`，净 e2e 仍为正（尽管单条 recipient `wait_ms` 可达数秒）。

---

## 9. 第八阶段：选择性重算 selector 与语义门控（v0.8）

### 9.1 三种 selector（`SGLANG_VLM_CACHEBLEND_SELECT`）

| 模式 | 环境变量名 | 选哪些 token 重算 |
|------|-----------|------------------|
| **topr** | `topr` | 按 deviation 取 top-r%；无 deviation 时退化为前 r%（启发式） |
| **kvdev** | `kvdev` | 按 KV L2 deviation 取 top **15%**（`recompute_ratio=0.15`）— **默认主路径** |
| **cos** | `sim` | 视觉相似度 < `SIM_THRESHOLD`（默认 0.90）的 token 重算；脚本里 `cos` 映射为 `sim` |

实现：`vlm_cacheblend.py` 中 `select_recompute_tokens`、`finalize_recipient_plan_deviation` / `finalize_recipient_plan_similarity`。  
Cos 接线 commit：`1cdeebb`。

**重要澄清**：「85% 复用」= `recompute_ratio=0.15` + **kvdev**，不是 cosine top-85%。

### 9.2 `TARGET_TURNS`（commit `f0ac2c7`）

- `SGLANG_VLM_CACHEBLEND_TARGET_TURNS=all`：每个 **agent turn ≥ 1** 都可走 CacheBlend（turn0 仍只做 prefix warmup，不做 KV graft）。
- 也可设为 `1` 或 `1,3` 等逗号列表，减少长轨迹上的 barrier 次数。
- SGLang 与 verl 两侧均通过 `target_turn_enabled()` 解析，行为一致。

### 9.3 Semantic AB 流水线

脚本：`verl/examples/profile/workloads/geo3k/run_geo3k_cacheblend_semantic_ab.sh`

顺序跑 **off → kvdev → cos**，再用 CPU 脚本对比：

- 语义：`verl/examples/profile/shared/analysis/semantic_cacheblend_gate.py`  
  检查相对 off 是否有 answer/score 退化 + 汇总 turn1 时延、`reused_tokens`、`attention_skipped_tokens`。
- 路由/barrier：`validate_cacheblend_multiworker.py`  
  检查每组恰好 1 个 donor、sticky `server_id`、`donor_not_ready`。

默认正式配置（commit `55e8138`）：4 GPU、`bs64`、`n4`；尾部 Hydra override 通过 `EXTRA_OVERRIDES` 透传到内层 `run_geo3k_refocus_profile.sh`。

报告输出：`profile_logs_geo3k_cacheblend_semantic_ab/geo3k_refocus_*_semantic_gate.json`。

### 9.4 读日志时注意：batch 级 recipient 计数

SGLang 常把多个 recipient **批进一条 EXTEND forward**。`model_forward_log` 里可能只有十几行 `cacheblend_role=recipient`，但 `cacheblend_fallback_reason` 里的  
`batch:recipient_kv_blended=N` 展开后应为 **192**（64 组 × 3 recipient）。  
donor 对应 `batch:donor_captured=N` 展开应为 **64**。  
**不要**仅凭 recipient 行数判断覆盖率。

---

## 10. 第九阶段：正式规模端到端结果（2026-06-29）

### 10.1 配置

| 项 | 值 |
|----|-----|
| GPU | `CUDA_VISIBLE_DEVICES=2,3,6,7`（亦试过 4,5,6,7） |
| 规模 | `TRAIN_BATCH_SIZE=64`，`ROLLOUT_N=4`，`AGENT_NUM_WORKERS=4` |
| 变体 | exact |
| 模式 | `VERL_PROFILE_ROLLOUT_ONLY=1`（rollout-only，非 smoke） |
| RUN_TAG 示例 | `semantic_4g_aw4_bs64n4_2367` |

### 10.2 Semantic gate（相对 off）

| selector | 语义 | turn1 `sglang_call_ms` med | 相对 off 加速 | reused | recomputed | attention skipped |
|----------|------|---------------------------|--------------|--------|------------|-------------------|
| off | baseline | 3714 ms | — | 0 | 0 | 0 |
| kvdev | **PASS** | 1989 ms | **46.5%** | 31419 | 5553 | 7905 |
| cos | **PASS** | 2016 ms | **45.7%** | 31419 | 5553 | 24868 |

- answer / score：**零退化**（`answer_changed=0`，`correct_to_wrong=0`）。
- 回复文本仍有变化：kvdev **39.8%**，cos **33.2%**（answer 未变，需持续监控对 RL 训练信号的影响）。
- 报告：`profile_logs_geo3k_cacheblend_semantic_ab/geo3k_refocus_exact_semantic_4g_aw4_bs64n4_2367_semantic_gate.json`

### 10.3 Multi-worker 校验

kvdev / cos 均 **PASS**：

- `warmup_groups=128`（64 组 × turn0 prefix + turn1 cacheblend）
- donor / recipient barrier 行 = **128 / 384**
- `donor_not_ready=0`，`recipient_used_rate=1.0`（在 logged recipient 行上）
- batch 展开：donor **64**，recipient **192**

JSON：`..._2367_kvdev_multiworker.json`、`..._2367_cos_multiworker.json`。

### 10.4 kvdev vs cos 结论（exact 上）

| 观察 | 含义 |
|------|------|
| turn1 e2e：**kvdev 略快 ~27 ms** | 端到端仍以 `sglang_call_ms` / `prefill_launch_latency` 为主 |
| recipient forward med：**cos 更快**（156 vs 252 ms） | GPU forward 上 cos 的 attention skip 更激进 |
| attention skipped：**24868 vs 7905** | M2（FA3 query skip）在 cos 路径更活跃 |
| reused / recomputed **总和相同** | **exact refocus 图**上，两 selector 选出同一批 recompute token；差异主要在 attention 路径，不在 KV graft 集合 |

**设计含义**：exact 变体上 **kvdev 作为默认 selector 合理**；cos 的优势应在 **diversified**（相似但不 exact 的图）上验证，该规模 semantic AB 尚未跑完。

### 10.5 与 off baseline 对照时的注意点

semantic AB 的 **off** 跑：`SGLANG_VLM_CACHEBLEND=0` → **barrier 也关闭**（512 行全 `bypass`）。  
kvdev/cos 跑：barrier + prefix warmup **均开启**。  

因此表内 ~46% 加速是 **「barrier + prefix 排序 + KV graft + M2 skip」整包** 相对 vanilla，不是单独 KV graft 的净收益。若需拆因，应补跑 `CACHEBLEND=0` + `WARMUP_BARRIER=1` 对照。

---

## 11. 第十阶段：Rollout 收敛与 bounded barrier（2026-07）

6 月 semantic AB 证明机制有效后，7 月工作聚焦 **只做 rollout 优化**（E+P / KV graft），不再靠 training bypass 等方式缩短 wall-clock。

### 11.1 代码变更（main 已合入）

| commit | 内容 |
|--------|------|
| `242b1d0` | verl：**bounded warm-barrier**、`cacheblend_barrier_log_*.csv` 分字段日志、`run_geo3k_rollout_ab.sh`（rollout-only AB，标准 PPO recompute） |
| `307076e` | SGLang：**Method B/C**（`fast_apply` / `compact_prefill`）、multi-slot graft、`min_reuse` gate；**全部 env 默认关**，Geo3K 生产仍 fa0/cp0/slotslast |

**明确不做的事**：

- training bypass / audit / logprob sanitize（已回退，不在 rollout 优化栈里）
- 靠拉长 `max_response_length` 人为提高 rollout 占比

### 11.2 当前推荐 rollout 栈（Geo3K exact，64×4）

| 层 | 配置 | 说明 |
|----|------|------|
| ① KV graft | `CACHEBLEND_SELECTOR=kvdev` | donor/recipient + kvdev 15% 重算 |
| ② Slot | `CACHEBLEND_IMAGE_SLOTS=-1` | refocus 最后一张图（slotslast） |
| ③ Apply | `CACHEBLEND_FAST_APPLY=0` | Geo3K 上 fa1 未赢，默认关 |
| ④ Compact | `CACHEBLEND_COMPACT_PREFILL=0` | Geo3K decode 太短，cp1 证负；**decode-heavy workload 再开** |
| ⑤ Warm barrier | `WARMUP_BARRIER=1`, `PREFIX_WARMUP=1`, `TARGET_TURNS=1` | turn0 前缀 + turn1 graft |
| ⑥ Bounded wait | `WAIT_POLICY=bounded`, **`MAX_WAIT_S=10`**（慢簇） | 见下节 |
| ⑦ Sparse decode | `SGLANG_VLM_CACHEBLEND_SPARSE_DECODE=0` | **context 侧**稀疏：decode 时跳过 attend reused image KV；GPU mask/gather 构造短 page table；默认关，长 decode 数据集再开 |

**Prefill vs Decode 优化对照**：

| 功能 | 阶段 | 稀疏对象 | 默认 |
|------|------|----------|------|
| Compute skip (M2) | prefill/extend | reused **image query** 的 QKV/MLP/attn | 开 |
| Compact prefill (cp1) | prefill/extend | 物理缩短 active sequence | 关 |
| **Sparse decode** | **decode** | 缩短 **context page table**（少看 reused image KV）；`KEEP_RECENT/KEEP_FIRST/MIN_DROP` 可控 | 关 |

### 11.3 Bounded wait 关键结论（2367 同环境 AB）

| Run | `MAX_WAIT_S` | turn1 e2e | `donor_ready` | 备注 |
|-----|-------------|-----------|---------------|------|
| off baseline | — | 3383 ms | — | 同环境对照 |
| kvdev canary | `0` | 6163 ms (+85%) | **0%** | recipient 不等 donor → graft 命中率 ~43% |
| kvdev wait10 | **`10`** | **2899 ms (−14%)** | **100%** | 推荐慢簇配置 |

**读 graft 覆盖率**：不要数 `model_forward` 里 `cacheblend_used` 行数；应看 `batch:recipient_kv_blended=192`（64 组 × 3 recipient = **75%** request 覆盖率，GRPO n=4 结构上限）。

### 11.4 整步 profiling 口径

- **`TOTAL_STEPS=2`**：step1 含冷启动，**看 step2**
- **`timing_s/gen`** = rollout 墙钟；training = `old_log_prob` + `update_actor` + `update_weights`
- 单请求 E+P 看 **`generate_e2e_ms` turn1** + `vision_encoder_log` / `model_forward_log`
- 详细归档：`verl/docs/geo3k_profiling_archive_0702.md`

### 11.5 一键复现（给别人跑）

**前置**：4 GPU、`cd graftrl/verl`、conda 环境就绪、`export CUDA_VISIBLE_DEVICES=0,1,2,3`。

```bash
# ARM A — baseline（CacheBlend 全关）
bash examples/profile/workloads/geo3k/run_geo3k_rollout_demo.sh baseline

# ARM B — rollout E+P 优化（kvdev + warm barrier + bounded wait）
bash examples/profile/workloads/geo3k/run_geo3k_rollout_demo.sh optimized
```

底层脚本：`run_geo3k_rollout_ab.sh` → `run_geo3k_refocus_profile.sh`。  
日志：`profile_logs_geo3k_rollout_ab/` + `profile_logs_geo3k_refocus_exact_demo_*`。

跑完后对比 step2：

```bash
grep 'step:2' profile_logs_geo3k_rollout_ab/geo3k_refocus_exact_demo_off_*.log | grep timing_s
grep 'step:2' profile_logs_geo3k_rollout_ab/geo3k_refocus_exact_demo_kvdev_*.log | grep timing_s
python3 examples/profile/shared/analysis/analyze_profiling_logs.py \
  --log-dir profile_logs_geo3k_refocus_exact_demo_kvdev_slotslast_fa0_cp0 \
  --suffix geo3k_refocus_exact_demo_kvdev_slotslast_fa0_cp0_bs64_n4 --report
```

若 `donor_ready` 仍为 0%，把 optimized 臂的 `SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_MAX_WAIT_S` 提高到 `10`（demo 脚本默认已是 10）。

---

## 12. 实验结论总表（写论文/汇报可直接用）

| 阶段 | 做了什么 | 结果 |
|------|----------|------|
| Profiling | 三路日志 + EPD | 瓶颈在 LLM prefill 与调度，不在单次 ViT 微秒级耗时 |
| Motivation | 相似度 + 现有 exact cache | 冗余存在；SGLang 整图缓存在 GRPO 里几乎无用 |
| ViT 局部复用 | window / merged / token_sparse | **负结果**：merged 高相似但太晚；sparse 真省但太慢 |
| GraftRL (LLM KV) | donor/recipient + 选择性重算 | **机制通、质量无损**；Chart 正常尺度 **端到端几乎不加速** |
| Stress + chunk fix | 大图 + n=8 + 关 chunk | 设计用于放大可观测收益 |
| **Geo3K refocus** | exact/diversified/stress × bs64×n4 | on 相对 off e2e **−15%～−23%**；ViT encode **~75%↓** |
| **Warm barrier** | stress bs32 on/off | on 相对 off e2e **~−22%** |
| **Multi-worker barrier v2** | 4 AgentLoopWorkers + Coordinator | 128 groups 校验通过；batch 级 64 donor + 192 recipient |
| **Semantic AB (exact)** | off/kvdev/cos @ 64×4×aw4 | kvdev/cos 均 PASS；turn1 **~46%** 加速；exact 上 kvdev≈cos 选 token |
| **Cos selector** | `sim` + attention skip | 语义无损；e2e 未优于 kvdev（exact）；skip 更激进但不转化为 med 延迟收益 |
| **Bounded barrier (2026-07)** | wait10 vs canary @ 2367 | `MAX_WAIT_S=10` → donor_ready **100%**，turn1 **−14%** vs off；`MAX_WAIT_S=0` 证负 |
| **Decode sparse (optional)** | fa1/cp1 env-gated | Geo3K 默认关；decode-heavy workload 再验证 |

### 一些常问的数字

**Chart turn1，正常尺度**

| 指标 | 约数 |
|------|------|
| 完整 turn1 prompt | 4676 token（2 图 1160 + 文 3516） |
| recipient 复用的 refocus 图 token | ~493 / 580 |
| rollout.n=4 时受益比例 | 3/4 分支是 recipient |

**Geo3K exact，正式 64×4×aw4（2026-06-29）**

| 指标 | off | kvdev |
|------|-----|-------|
| turn1 sglang_call med | 3714 ms | 1989 ms |
| turn0 sglang_call med | 4797 ms | 1856 ms |
| barrier recipient wait med | — | ~3182 ms |
| 语义退化 | — | 0 |

---

## 13. Git 版本与文档对照

| Tag / commit | 内容 |
|--------------|------|
| `v0.0-baseline` | 上游 verl + sglang，无 GraftRL |
| `v0.1-profiling` | 测速日志与 Geo3K profile |
| `v0.2-motivation` | Refocus agent、相似度、text-only 对照 |
| `v0.3-vit-reuse` | ViT 组内相似度缓存 + partial 实验 |
| `v0.4-graft-core` | `vlm_cacheblend.py` 核心 + donor 捕获 |
| `v0.5-graft-e2e` | attention 集成 + AB 脚本 |
| `v0.6-stress` | stress 数据 + chunk 修复 + 断言脚本 |
| `acdb6d6` | Geo3K refocus multiturn profiling workload |
| `bd3da72` | Donor-ready barrier v1（单 client 锁） |
| `1cdeebb` | Cosine (`sim`) selector 接线 |
| `f0ac2c7` | `TARGET_TURNS` 多 turn 选择（SGLang） |
| `8e52bfe` | Global CacheBlend coordinator（multi-worker barrier v2） |
| `55e8138` | Geo3K 脚本 64×4 默认 + Hydra override 透传 |
| `d137c56` | README 指向本文档与 RELEASES |
| `dbaf089` | Barrier 超时 fail-fast；`training_global_step` 契约测试；profile 脚本补 `chunked_prefill_size=-1` |
| `242b1d0` | Bounded warm-barrier + barrier CSV 日志 + `run_geo3k_rollout_ab.sh` |
| `307076e` | SGLang fast_apply / compact_prefill / multi-slot graft（env 默认关） |

技术附录（英文，**以本文为准**）：

- `docs/VLM_CACHEBLEND_WARMUP_BARRIER.md` — barrier 设计与 multi-worker 范围  
- `verl/docs/geo3k_profiling_archive_0702.md` — Geo3K 7 月 profiling 归档与数字表  
- `VLM_CACHEBLEND_DESIGN.md` — LLM KV 设计  
- `VLM_CACHEBLEND_CHUNKED_PREFILL_FIX.md` — chunk 问题  
- `PARTIAL_WINDOW_REUSE_EXPERIMENT_SUMMARY.md` — ViT 线细节  

---

## 14. 本地 `profile_logs*` 要不要都留

这些目录是**跑实验生成的 CSV/图**，已在 `.gitignore` 里，**不会上传 GitHub**。

建议保留能支撑结论的 run：

| 目录 | 用途 |
|------|------|
| `profile_logs_vtool_chart_positive4602_on_4g` / `_off_4g` | Chart 质量 AB |
| `profile_logs_cacheblend_stress_s2_on_4g_n8` / `_off_4g_n8` | stress（若已重跑） |
| `profile_logs_geo3k_refocus_exact` / `_cacheblend_on` | Geo3K exact on/off |
| `profile_logs_geo3k_refocus_stress_warmbarrier_{on,off}_bs32_n4` | barrier 消融 |
| `profile_logs_geo3k_cacheblend_semantic_ab/` | semantic gate JSON 汇总 |
| `profile_logs_geo3k_refocus_exact_semantic_4g_aw4_bs64n4_2367_{off,kvdev,cos}` | 正式 64×4×aw4 三轮 |
| `profile_logs_geo3k_refocus_exact_demo_{off,kvdev}_*` | demo 脚本 baseline vs optimized |

其余早期 sweep 可删或打 tar 冷备份。

---

## 15. 当前状态与后续（可选）

### 15.1 Rollout 阶段已闭合的能力

- Geo3K refocus 三变体 workload + 正式 **64×4** profiling 流水线  
- Donor/recipient KV graft + kvdev selector + FA3 attention skip（M2）  
- Turn0 prefix barrier + turn≥1 cacheblend barrier + **multi-worker** Ray coordinator  
- **Bounded wait** + barrier CSV 分字段日志（`barrier_wait_ms` / `server_call_ms` / `donor_ready`）  
- Semantic gate + multi-worker 离线校验（2026-06）  
- exact 上 **~46% turn1 加速**（rollout-only AB）、语义零退化、batch 级 **64+192** 全覆盖  
- **一键 demo**：`run_geo3k_rollout_demo.sh baseline|optimized`

### 15.2 已知局限（诚实写进论文/汇报）

| 项 | 说明 |
|----|------|
| off 对照无 barrier | 6 月 semantic AB 加速含 barrier 收益，非纯 CacheBlend |
| `MAX_WAIT_S` 环境敏感 | 慢簇需 **10s**；`MAX_WAIT_S=0` 会导致 donor_ready=0%、graft 失效 |
| M2 仅 FA3 完整 | FlashInfer 等后端无 attention query skip |
| cp1 / fa1 在 Geo3K 默认关 | decode 占比低；decode-heavy 数据集再开 Method B/C |
| cos 在 exact 上不优于 kvdev | 选 token 集合相同；diversified 待测 |
| `response_changed` ~30–40% | answer 未变，RL 训练影响待观察 |
| 跨 replica KV | 未实现；sticky 失效时整组复用失败 |
| `TOTAL_STEPS=1` profile | 墙钟 EPD 分解含冷启动；稳态需 ≥2 step |
| profile 模式 `score_mean=0` | 只看相对退化，不看绝对答题率 |

### 15.3 若继续推进（非必须）

1. **diversified × 64×4 × aw4** semantic AB — 验证 cos 在相似图上的价值。  
2. 补对照 **`CACHEBLEND=0` + `WARMUP_BARRIER=1`**，拆清 barrier 净收益。  
3. decode-heavy workload 上开 **sparse decode**（必要时配 `KEEP_FIRST` / `MIN_DROPPED_TOKENS`），验证 context attention 稀疏 decode 收益。
4. `TOTAL_STEPS=2` 整步 profiling，拿稳态 gen vs training 占比。  

---

## 16. 当前架构的可采信收益快照（2026-07-20）

> 本节是对关闭/删除自定义 ViT 优化后的当前架构重新核算。前文的 `14%～46%`
> 等数字保留为历史实验记录，其中部分捆绑了 barrier、routing、prefix ordering
> 或旧 ViT cache，不应归因为当前纯 LLM prefill 的单项收益。

### 16.1 总结论

| 优化方向 | 当前实测 | 结论 |
|----------|----------|------|
| LLM prefill CacheBlend 复用 | rollout `50.380s → 43.895s`，**-12.9%** | 已证明端到端正收益 |
| 同两个 PPO step 整步 | `124.503s → 117.747s`，**-5.4%** | 已证明整步正收益 |
| CUDA Graph dense decode | 相对 forced eager dense 约 **-3.6%** | 已证明正收益 |
| 旧 reuse-mask sparse 局部 forward | 长上下文下 **-2.1%～-2.7%** | 局部计算已变快 |
| 旧 reuse-mask sparse rollout | `127.797s → 128.007s`，**+0.16%** | 测量噪声内持平，且策略假设已废弃 |
| query-aware sparse v1 | MMDU rollout `315.004s → 295.028s`，**-6.34%** | 单组 `64×4` 性能/质量 gate 已通过 |
| mass-budget sparse v2 | MMDU `287.068s → 295.671s`，墙钟 **+3.00%**；token/s **+0.83%** | 质量/长度门通过，但端到端门未过；已继续优化 v2.1 |
| 自定义 ViT cache / token sparse | 历史实验均慢于 baseline | 已删除，不保留 |
| `fast_apply` / `compact_prefill` | 整理、拷贝开销无法被省下的计算覆盖 | 默认关闭 |
| MMSearch 跨分支 KV 复用 | 新架构可 exact retrieval，但仍有大量 position mismatch | 无正式性能收益 |

因此，当前不能宣称「所有 workload 都等价或优于 baseline」。可采信的收益边界是：
**Geo3K/Refocus 这类组内重复视觉内容、且 recipient 能命中 donor 的场景，当前纯 LLM
prefill 复用可带来约 13% rollout 和约 5% 完整 PPO step 收益；在 decode-heavy MMDU
单组正式 gate 中，query-aware sparse v1 带来 6.34% rollout 收益。**

### 16.2 纯 prefill `64×4` 正式结果

配置：Geo3K exact，2 GPU，`train_batch_size=64`，`rollout.n=4`，2 个 PPO step，
temperature=0，所有自定义 ViT 优化关闭，`fast_apply=0`，`compact_prefill=0`。

| 指标 | baseline | prefill reuse | 变化 |
|------|----------|---------------|------|
| step 1 `timing_s/gen` | 33.007s | 30.339s | -8.1% |
| step 2 `timing_s/gen` | 17.372s | 13.555s | -22.0% |
| 两步 rollout 总计 | 50.380s | 43.895s | **-12.9%** |
| 两步 `timing_s/step` 总计 | 124.503s | 117.747s | **-5.4%** |

两臂每个 step 的 global sequence length、response length 和输出 token 数一致，因此不是靠
少生成 token 获得的速度收益。正式对比应以 trainer 的 `timing_s/gen` 为主；单请求
CSV 在 cache arm 中包含额外内部记录，两臂行数不同（512 vs 576），不能直接用其
per-request mean 宣称约 60% 收益。

当前 prefill 收益来自完整复用链路：group/replica affinity、donor-ready barrier、视觉
token span 匹配、donor K/V graft 以及被复用 merged visual token 的 LLM prefill 少算。
它不包含 ViT 层加速。

### 16.3 Sparse decoding 的当前边界

截至 2026-07-19 的正式数据来自旧 `reuse` 策略：把 CacheBlend 在 prefill 判为可复用的
image KV 当作 decode 可忽略的 token。该策略已实现 semantic position、fused Triton
compact、稳定 buffer/cache、CUDA Graph、append-only page table、profitability floor，
以及从 `req_to_token` 直接 gather/compact。

- 上下文丢弃约 28% 时，long/high-batch decoder forward 快 **2.1%～2.7%**。
- 加入 4096-token profitability floor 后，稀疏 rollout 从原先比 dense 慢 1.43%
  收敛到只慢 **0.16%**，但仍不能计为正收益。
- direct-source 路径用于移除每个 decode token 的完整 dense page-table gather/
  allocation；旧 `reuse` 结果尚未覆盖它，本次 query-aware 正式 gate 的 7,086 条
  `SKIP` forward 已全部实际走过该路径。

这说明旧实现的 attention FLOPs 确实减少，但省下的计算仍被稀疏索引、page-table
整理和调度开销吃掉。更关键的是，`prefill K/V 可复用` 并不推出 `future decode query
可以不 attend 该 K/V`；因此 `reuse → drop` 规则即便跑快也缺少语义依据。
旧环境变量别名现在也默认进入 `query_blocks`；历史 `reuse` 策略仅在显式设置
`SPARSE_DECODE_MODE=reuse` 时用于复现实验，不再因兼容别名被隐式选中。

通过正式 gate 的 v1 是独立的 query-aware block 策略：

- 不依赖 CacheBlend reuse mask，`SGLANG_VLM_CACHEBLEND=0` 时也能工作；
- 在 Qwen2.5-VL prefill 第 20 层，用新 prompt 末尾 32 token 中的 4 个 query landmark，对每个
  64-token 历史 block 的 4 个 K landmark 打分；
- 首 128 token 和最近 512 token 永远保留，候选 block 至少保留 50%；
- block 排序只做一次，decode 继续使用固定 position plan、direct-source fused compact、
  append-only page table 和 CUDA Graph，没有逐 token top-k；
- 这是近似 `SKIP`，不是 `EXACT/PARTIAL` 复用；CPU/静态回归和下述单组 GPU gate 均已通过。

正式验收只跑了一组 MMDU dense/query-sparse A/B：真实 `64×4`、两个 rollout-only
step、temperature=1 且固定 data seed=42、CUDA Graph；无 smoke、ABBA 或隐藏重复。

| 指标 | dense | query sparse | 变化 / 门槛 |
|------|-------|--------------|-------------|
| step 1 rollout | 165.910s | 159.241s | **-4.02%** |
| step 2 rollout | 149.094s | 135.786s | **-8.92%** |
| 两步 rollout 总计 | 315.004s | 295.028s | **-6.34%**（要求 ≥5%） |
| reward mean | 0.441203 | 0.437968 | -0.003235（允许下降 ≤0.01） |
| response length mean | 475.904 | 460.902 | -3.15%（允许漂移 ≤5%） |
| abort ratio max | 0 | 0 | 通过 |

稀疏端共有 7,086 条实际 `SKIP` forward、2,233 条 fail-closed `LOCAL` forward；全部
`SKIP` 使用 direct-source compact，其中 4,216 条命中 incremental append。累计保留
1,346,761,602、丢弃 387,485,698 个 decode attention context-token visit，实际丢弃率
22.34%，不是 dense bypass。该单组 gate 已通过；但 sparse attention 仍是近似策略，且
输出长度相差 3.15%，所以结论限定为“这组固定 seed 的端到端正收益与质量代理门通过”，
不外推为严格无损或多 seed 稳健性证明，默认仍保持关闭、由 workload 显式 opt-in。
按 `64×4×2=512` 条轨迹的实际平均输出长度折算，生成吞吐约从 773.5 提升到
799.9 response token/s（+3.41%）；因此 6.34% 墙钟收益不只是少生成 3.15% token，
但墙钟收益中确实包含了输出长度差异，不能把全部 6.34% 都归为 kernel 加速。

#### Gate 后继续开发与一次正式 v2 验证

按“先做确定的执行优化，再改变近似强度”的顺序继续开发，并没有停在 6.34%：

- 重新解析正式 CSV 后，`LOCAL=2233/9319` 的说法被纠正：其中 789 条是 EXTEND，
  真正 decode `LOCAL` 是 `1444/8530=16.9%`。624 条发生在 batch=1，另有大量短于
  4096 的上下文；这些正是历史上强行 sparse 会负收益的尾部，不能为了降低 LOCAL
  计数而取消 profitability fallback。
- 7,086 条 sparse forward 中有 2,870 条重新 full compact。根因之一是 plan store 的
  全局 revision：无关请求注册/释放 plan 也会使稳定 batch 的 workspace 失效。v2 改为
  按当前 batch 的 request + plan object identity 做 key；同一 pool slot 的 plan 替换仍
  必然失效，但无关 store churn 不再打断 incremental append。workspace/static cache
  working set 也从 8/16 提高到 32/64。按 v1 同 batch-size 的 forward 差值估算，所有
  可避免 full compact 的理论上限约 3.19s；这是上限分析，不是新的实测收益。
- Sparse selector 不再使用“固定保留 top-k 比例”。v2 将结构上限放宽到最多丢 70%，
  但只有累计 query-score softmax mass 不超过 5% 的低分 block 才允许删除；分数均匀、
  判断不确定时自动接近 dense。prompt-tail query landmark 从 4 个增加到 8 个，仍只在
  prefill 排一次，decode 没有逐 token top-k。策略标识为
  `query-block-mass-prefill-v2`。
- 该 5% 是 landmark 近似得到的 proxy error bound，不是完整 attention 的数学误差界；
  它现在随 `rollout-reuse-action-v1` 的 approximate `SKIP` 事件写入
  `reuse_error_bound`，避免只报“丢了多少”而不报近似风险。
- 多 turn 重打分现在先撤销旧 turn plan；只有本轮所有 guard 和 score-mass 门通过才
  安装新 plan。若本轮分数均匀或打分失败，请求会 dense fail closed，不会继续沿用
  与当前 query 不一致的旧 drop 位置。
- full compact Triton kernel 不再清零 FA3 永远不会读取的 page-table 矩形尾部，并合并
  重复的 keep-count reduction；这只减少内存写和 reduction，不改变有效前缀或
  `cache_seqlens`。
- 所有 dense fallback 现在区分 `empty_plan_store`、`no_active_sparse_plan`、
  `drop_upper_bound_below_floor`、实际 drop/ratio 不足等原因。下一次单组运行即可判断
  哪类可优化，不需要为归因另跑实验。
- 验收除 wall time、reward、length、abort 外，新增 token-throughput 必须为正；防止
  把“少生成”误报为核心执行收益。

上述 v2 随后只跑了一组新的正式 `64×4` dense→sparse pair，没有 smoke、ABBA、扫参
或隐藏重复：

| 指标 | dense | mass-budget v2 | 变化 |
|------|-------|----------------|------|
| 两步 rollout | 287.068s | 295.671s | **+3.00%（未过端到端门）** |
| response token/s | 819.06 | 825.84 | **+0.83%** |
| response length mean | 459.229 | 476.908 | +3.85%（门内） |
| reward mean | 0.442712 | 0.443376 | +0.000665 |
| abort ratio max | 0 | 0 | 通过 |

v2 的实际 context-token 丢弃率提高到 30.4%，7,411 条 `SKIP` 中 5,846 条
incremental append（78.9%）；因此策略、direct-source 和增量路径均真实生效。但输出
更长使墙钟变慢，且归一化吞吐只快 0.83%，不能宣称端到端收益。剩余 1,565 条 full
compact 的同 bucket 乐观可避免上限约 3.11s，其中两次首次 Triton JIT 约 1.16s 落在
正式 rollout 内。

测量后直接继续开发 v2.1，没有再占 GPU：

- direct position plan 不再为了遗留物理位置诊断把整条 `req_to_token` 从 GPU 拷回
  CPU；decode 只使用稳定 semantic positions。
- direct compact 的相关 power-of-two Triton 变体移到 CUDA Graph/server 初始化阶段
  预热，避免首次编译污染正式 rollout latency；真实 kernel 执行仍保留在计时路径。
- 正式 MMDU pair 在 temperature=1 下为每个 `global step × sample × GRPO branch × turn`
  生成稳定 SGLang sampling seed。两臂仍按各自 logits 采样，不强行令输出相同，但不再
  把并发调度导致的 RNG 流差异误认为 sparse 语义差异。
- v2.1 只通过 86 项相关 CPU 回归和静态检查，尚无新的 GPU 数字；当前可引用的正收益
  仍只有 v1 的 6.34%，而 v2 的正式结论是墙钟负收益、token throughput 小幅为正。

### 16.4 删除、保留与默认开关

- 已删除：自定义 ViT embedding cache、ViT merged-token replacement、ViT token sparse。
- 默认关闭：`fast_apply`、`compact_prefill`；目前没有可采信正收益。
- 继续保留：LLM prefill CacheBlend（包括 exact/kvdev 与 similarity selector）。删除的
  `grpo_similarity_cache.py` 属于旧 ViT 线，不是 LLM prefill 的 similarity selector。
- 继续保留：query-aware sparse v1 已取得一组真实 `64×4` 端到端正收益；v2 正式 pair
  墙钟未过门后已继续优化为 v2.1，尚未再次占卡。该方向因 approximate 语义保持显式
  opt-in。

### 16.5 正确性与质量口径

- Geo3K profile 的两臂生成长度一致，能证明性能可比，但 profile reward 为 0，
  不能作为任务质量验收。
- 旧 MMDU 单次结果：baseline reward `0.4601`，prefill `0.4522`，
  prefill+sparse `0.4514`。新 query-aware 正式 pair 的 reward 为 `0.441203 → 0.437968`
  （`-0.003235`）、abort 为 0，已通过预设代理质量门；仍不能据单 seed 宣称严格质量无损。
- MMSearch 已完成 exact retrieval/tokenization reuse 和 group/replica 归因，但不同分支的
  position/prefix hidden state 仍会使 K/V graft fail closed；在新架构完成正式实验前，
  不宣称性能收益。

### 16.6 统一 `EXACT/PARTIAL/SKIP/LOCAL` 运行时

此前 `rollout_reuse` 只提供 identity/registry/routing，MMSearch、CacheBlend 和 sparse
decode 各自使用不同字段，确实还不能称为公共运行时。当前已补齐：

- `ExecutionDecision` 把 action、operator、representation stage、reason、eligible/applied
  units、approximate/error bound 和 policy 放入同一个强校验对象；
- `RolloutReuseRuntime` 统一执行 exact registry、记录各 action、聚合计数与工作量，并能
  读取远端 backend 的 versioned wire event；
- MMSearch retrieval/tokenization 的 `EXACT/LOCAL`、去重/已物化的精确 `SKIP`、rank
  budget 的近似 `SKIP` 已接入该 runtime；
- SGLang CacheBlend 的 `PARTIAL/LOCAL` 与 sparse decode 的近似 `SKIP` 都输出同一
  `rollout-reuse-action-v1` schema；backend 先通过 `rollout_reuse_backend.py` 校验，
  agent runtime 可读取并合并归因；
- query-aware selector 已从 CacheBlend 主模块中独立到
  `query_aware_sparse_decode.py`，Sparse Decode 不再由视觉复用语义驱动。
- v2 的 query-score mass 作为 approximate `SKIP` 的 proxy `error_bound` 进入同一事件，
  不再用无误差口径的固定 top-k 数量代表策略质量。
- 正式 pair 暴露出一个纯归因问题：未命中 sparse plan 的 `LOCAL` decode 行仍沿用
  `vlm_cacheblend.prefill` operator 名。它不影响执行或上述计时，现已改为
  `sparse_decode.context_attention / decode_kv_blocks / no_active_sparse_plan`；后续 trace
  可以按同一 operator 同时统计 `SKIP` 与 fail-closed `LOCAL`。

边界也需说清：`vlm_cacheblend.py` 内仍保留大量 prefill KV backend 与 sparse page-table
executor 的机械实现，尚未完成文件级拆分；但它不再拥有另一套 action 语义。公共运行时
已经形成控制契约和统一 trace，后续重构是 backend 拆包，不是重新设计论文抽象。

---

*文档版本：当前工作区快照，2026-07-20。*
