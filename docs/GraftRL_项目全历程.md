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

### 4.2 算法与代码（ViT 路线）

**开关**：环境变量 `SGLANG_GRPO_SIM_CACHE=1` 才走这条路径；论文对比 baseline 时设为 **0**。

**核心流程**（白话）：

1. 同组第一个编码这张图的 branch 当 **donor（供体）**，把 ViT 中间结果存进组内缓存。  
   （`sglang/python/sglang/srt/mem_cache/grpo_similarity_cache.py`，类 `DonorKVStore` / 函数 `encode_with_grpo_similarity_cache`）

2. 同组后面的 branch 当 **recipient（受体）**：  
   - 原图 slot：必须像素几乎完全一样才跳过 ViT；  
   - refocus slot：算 raw patch 相似度，够高就整图 embedding 直接抄 donor。

3. 「真正少算」的 token_sparse 在 ViT 前 7 层里，对标记为「可复用」的 token 跳过 FFN。  
   （`sglang/python/sglang/srt/models/qwen2_5_vl.py`，函数 `forward_with_partial_window_reuse`）

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

**总开关**：`SGLANG_VLM_CACHEBLEND=1`（与 ViT 那条线的 `SGLANG_GRPO_SIM_CACHE` 独立）。

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

## 7. 实验结论总表（写论文/汇报可直接用）

| 阶段 | 做了什么 | 结果 |
|------|----------|------|
| Profiling | 三路日志 + EPD | 瓶颈在 LLM prefill 与调度，不在单次 ViT 微秒级耗时 |
| Motivation | 相似度 + 现有 exact cache | 冗余存在；SGLang 整图缓存在 GRPO 里几乎无用 |
| ViT 局部复用 | window / merged / token_sparse | **负结果**：merged 高相似但太晚；sparse 真省但太慢 |
| GraftRL (LLM KV) | donor/recipient + 选择性重算 | **机制通、质量无损**；Chart 正常尺度 **端到端几乎不加速** |
| Stress + chunk fix | 大图 + n=8 + 关 chunk | 设计用于**放大可观测收益**；需重跑确认 |

### 一些常问的数字（Chart turn1，正常尺度）

| 指标 | 约数 |
|------|------|
| 完整 turn1 prompt | 4676 token（2 图 1160 + 文 3516） |
| 本次实际要算的 extend（前缀缓存后） | ~1406 / 分支 |
| recipient 复用的 refocus 图 token | ~493 / 580 |
| rollout.n=4 时受益比例 | 3/4 分支是 recipient |

---

## 8. Git 版本与文档对照

| Tag | 内容 |
|-----|------|
| `v0.0-baseline` | 上游 verl + sglang，无 GraftRL |
| `v0.1-profiling` | 测速日志与 Geo3K profile |
| `v0.2-motivation` | Refocus agent、相似度、text-only 对照 |
| `v0.2.1-exploratory-workloads` | Sokoban、DeepEyes、archive |
| `v0.1.1-vision-cold-warm` | ViT 冷/热路径分析脚本 |
| `v0.3-vit-reuse` | ViT 组内相似度缓存 + partial 实验 |
| `v0.4-graft-core` | `vlm_cacheblend.py` 核心 + donor 捕获 |
| `v0.5-graft-e2e` | attention 集成 + AB 脚本 |
| `v0.6-stress` | stress 数据 + chunk 修复 + 断言脚本 |

旧的分文档（仍可作英文技术附录，**以本文为准**）：

- `PARTIAL_WINDOW_REUSE_EXPERIMENT_SUMMARY.md` — ViT 线细节  
- `GRPO_SIMILARITY_CACHE.md` — ViT 组内缓存 API  
- `VLM_CACHEBLEND_DESIGN.md` — LLM KV 设计（英文）  
- `VLM_CACHEBLEND_CHUNKED_PREFILL_FIX.md` — chunk 问题  
- `VLM_CACHEBLEND_VERIFICATION.md` — 代码核对清单  

---

## 9. 本地 `profile_logs*` 要不要都留

这些目录是**跑实验生成的 CSV/图**，已在 `.gitignore` 里，**不会上传 GitHub**（约 1.1GB、31 个文件夹）。

建议只留能支撑结论的几次 run，例如：

- `profile_logs_vtool_chart_positive4602_on_4g` / `_off_4g`（质量 AB）  
- `profile_logs_cacheblend_stress_s2_on_4g_n8` / `_off_4g_n8`（stress）  

其余早期 sweep、smoke 可删或打 tar 冷备份。

---

## 10. 接下来建议做什么

1. **重跑 stress**（关 chunk），确认 `assert_cacheblend_used.py` 通过。  
2. 若仍要更大加速：考虑 **同 agent_uid 固定到同一 GPU**（提高 turn0 同图 ViT 命中），或换 **图占比更高** 的 workload。  
3. 论文叙事：**机制 + 质量** 可写；**Chart 正常尺度端到端加速** 需诚实写 bound，或靠 stress 证明「放大后有效」。

---

*文档版本：与 GraftRL 仓库 main 同步，2026-06。*
