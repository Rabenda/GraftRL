# GRPO Cross-Branch Turn1 Image-Token Similarity Cache

本文档描述在 `verl_vision` + `sglang_vision_profile` 上新增的 **GRPO 组内 cross-branch turn1 image token 复用 cache**，供 Codex / 合作者 code review 使用。

---

## 1. 背景与动机

### 1.1 要解决的问题

在 VTool Chart / Refocus 的 GRPO rollout 中：

- 每个样本采样 `rollout.n=4`，4 个 branch 共享同一个 `agent_uid`（GRPO group）。
- **Turn0**：4 个 branch 输入图 byte 级完全相同。
- **Turn1**：prompt 含 2 张图 —— turn0 原图（slot 0）+ refocus 图（slot 1）。slot 0 仍相同；slot 1 各 branch 不同但 ViT token cosine 很高（相似度分析 + Phase2 token replacement 已验证）。

现有 SGLang `MultiModalStaticCache`（exact hash）：

- key = 图片 raw bytes hash，要求 **逐字节相同** 才复用。
- Refocus 图 cross-branch 必然 miss。
- 即使 turn0 图相同，RL rollout 跨 replica / 调度顺序也会导致 exact cache 命中率接近 0（`vision_encoder_log` 中 `cached=0`）。

### 1.2 本改动的目标

**不是** demo，也不是渐进式 V0/V1；而是一次性接入可开关的完整机制：

> 在同一 GRPO group（`agent_uid`）内，对 **turn1** 的多图 ViT encode 做 **group-aware 相似度复用**，跳过后续 branch 的冗余 ViT，从 `vision_encoder_time_ms` 上 measurable 降耗时。

功能正确性（高 sim token 替换后 ΔNLL≈0）已由离线 Phase2 支持；本改动聚焦 **在线 encode 路径 + 计时/profile**。

---

## 2. 设计概览

### 2.1 与现有 cache 的区别

| 维度 | SGLang `MultiModalStaticCache` | 本改动 `GrpoSimilarityCache` |
|------|-------------------------------|------------------------------|
| Key | 图片 content hash | `(agent_uid, turn, image_slot, grid_shape)` |
| 粒度 | 整 request 的多模态 embedding | 单张图（按 `image_grid_thw` 切分） |
| 相似性 | 仅 exact match | exact 复用 + 在线 raw-patch 相似度验证 |
| 作用 scope | server 级 LRU | GRPO group 级（进程内） |
| 目标场景 | serving 重复图 | RL rollout cross-branch turn1 |

### 2.2 Cache Key

```python
(global_step, agent_uid, agent_turn, image_slot, grid_sig)
# grid_sig = (T, H, W) 来自 image_grid_thw 的单行
```

- `agent_uid`：GRPO group id（verl agent loop 传入）
- `agent_turn`：固定只对 `1` 生效
- `image_slot`：turn1 两图切分后的 index（0=原图，1=refocus）
- `grid_sig`：防止不同分辨率/ grid 误复用
- `global_step`：训练 step；step 变化时清空旧 cache，避免长跑累计显存/内存

### 2.3 编码流程（turn1）

```
get_image_feature(items)
  ├─ 解析 item.hash → rid → agent_uid / agent_turn（request meta registry）
  ├─ 若 turn != 1 或无 meta → 走原始整段 ViT（不启用 group cache）
  └─ turn == 1：
       split_images_by_grid(pixel_values, image_grid_thw)
         ├─ slot 0（turn0 原图）
         │    ├─ 无 donor → ViT → 写入 cache
         │    └─ 有 donor 且 patch 相同率 ≥ 99.9% → skip ViT，直接用 donor embedding
         └─ slot 1（refocus）
              ├─ 无 donor → ViT → 写入 cache
              └─ 有 donor → 先查 exact；不 exact 时计算 raw patch cosine
                   通过阈值 → skip ViT，直接用 donor embedding
                   未通过 → 正常 ViT，不覆盖 donor
       concat(slot0_emb, slot1_emb) → 返回
```

**First branch in group** 总是 populate cache；**后续 branch** 统一走同一套策略：

- slot0 原图：必须 exact，一致才 skip；
- slot1 refocus：先检查 exact；不 exact 时计算原始 patch 向量 cosine，达到阈值才复用 donor embedding。

这不是复用“生成出来的文字 token”，而是复用“图片经过 ViT 后得到的 image embedding/token”。LLM 后面的采样仍然独立进行。

### 2.4 为何按 image slot 切分

Turn1 的 `pixel_values` / `image_grid_thw` 在 Qwen2.5-VL 里是多图 concat 一次 ViT forward。不切分则：

- 无法只对 refocus 图做 group 复用；
- 也无法在 slot0 已相同时跳过其 ViT 部分。

切分后每张图独立 `visual(patches, grid_row=...)`，与 Qwen ViT 的 `cu_seqlens` 按 grid 行分段一致。

---

## 3. 代码改动清单

### 3.1 新增文件

| 文件 | 作用 |
|------|------|
| `sglang_vision_profile/python/sglang/srt/mem_cache/grpo_similarity_cache.py` | 核心 cache、meta registry、按 grid 切图、skip 策略 |
| `verl_vision/examples/profile/shared/analysis/compare_grpo_cache_timing.py` | 对比 baseline vs cache 的 `vision_encoder_log` 耗时 |
| `verl_vision/examples/profile/shared/docs/GRPO_SIMILARITY_CACHE.md` | 本文档 |

### 3.2 修改文件

| 文件 | 改动 |
|------|------|
| `sglang_vision_profile/python/sglang/srt/models/qwen2_5_vl.py` | 在 `get_image_feature()` 中 hook `encode_with_grpo_similarity_cache`；`vision_encoder_log` 增加 `grpo_sim_*` 字段 |
| `verl_vision/verl/workers/rollout/sglang_rollout/async_sglang_server.py` | `generate()` 调用 SGLang 前 `register_request_meta(rid, agent_uid, agent_turn, ...)` |
| `verl_vision/examples/profile/workloads/chart/run_rollout_profile.sh` | 导出 `SGLANG_GRPO_SIM_*` 环境变量（默认关闭） |

### 3.3 未改动的部分（刻意不做）

- 未改 SGLang `MultiModalStaticCache` / exact hash 路径
- 未改 `Req` / scheduler / `GenerateReqInput` 结构（agent 元数据走进程内 registry，避免大范围 SGLang 侵入）
- 未实现 per-token partial ViT（需改 ViT 内部 attention mask；当前是 **整图 skip** 或 **整图 ViT**）
- 未做 partial token merge；当前只做整图 slot skip

---

## 4. 数据流与依赖

```
vtool_agent_loop
  └─ generate(agent_uid=uid, agent_turn=0|1, request_id=branch_id)
       └─ llm_server.generate
            └─ sglang_request_id = f"{request_id}_t{turn}"
                 └─ async_sglang_server.generate
                      ├─ register_request_meta(sglang_rid, agent_uid, agent_turn)  ← 写入 _REQUEST_META
                      └─ tokenizer_manager.generate_request
                           └─ model_runner forward
                                ├─ item_hash_to_rid（已有 profile 逻辑）
                                └─ qwen2_5_vl.get_image_feature
                                     └─ resolve agent_uid from hash_to_rid + _REQUEST_META
                                     └─ encode_with_grpo_similarity_cache
```

**前提**：

1. verl rollout worker 与 SGLang model runner **同一进程**（当前 hybrid 部署满足）；registry 非跨进程 RPC。
2. `item_hash_to_rid` 会在 `SGLANG_LOG_INFERENCE_STEP=1` 或 `SGLANG_GRPO_SIM_CACHE=1` 时填充（`model_runner._forward_batch_item_hash_to_rid`），否则无法从 `MultimodalDataItem.hash` 反查 `rid`。
3. Agent loop 必须传 `agent_uid`（`vtool_agent_loop` / `deepeyes_agent_loop` 已有）。

---

## 5. 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `SGLANG_GRPO_SIM_CACHE` | `0` | 总开关 |
| `SGLANG_GRPO_SIM_RAW_COSINE_THRESH` | `0.995` | slot1 单个 patch 的 cosine 阈值 |
| `SGLANG_GRPO_SIM_RAW_COSINE_RATIO` | `0.90` | slot1 达到 cosine 阈值的 patch 比例 |
| `SGLANG_GRPO_SIM_MAX_GROUPS` | `512` | group cache entry LRU 上限 |
| `SGLANG_GRPO_SIM_MAX_REQUEST_META` | `65536` | request meta LRU 上限 |

---

## 6. Profile / 日志字段

开启 `SGLANG_LOG_INFERENCE_STEP=1` 时，`vision_encoder_log_*.csv` 每行 encode 可能包含：

| 字段 | 含义 |
|------|------|
| `grpo_sim_cache_enabled` | 是否开启 |
| `grpo_sim_policy` | 固定策略名：slot0 exact + slot1 similarity |
| `grpo_sim_agent_uid` | GRPO group id |
| `grpo_sim_agent_turn` | turn |
| `grpo_sim_vit_calls` | 本次 encode 实际 ViT 调用次数（0/1/2） |
| `grpo_sim_vit_skipped` | 本次 skip 的 ViT 次数 |
| `grpo_sim_slot0_skipped` | turn0 原图 slot skip 次数 |
| `grpo_sim_slot1_skipped` | refocus slot skip 次数 |
| `grpo_sim_cache_hits` / `grpo_sim_cache_misses` | slot 级 donor 命中/未命中 |
| `grpo_sim_exact_reuse` | exact 复用次数 |
| `grpo_sim_similarity_reuse` | raw patch cosine 通过后的相似复用次数 |
| `grpo_sim_similarity_checked` | 做过 raw patch cosine 检查的 slot 数 |
| `grpo_sim_raw_cosine_mean_min` | 本次 encode 中通过检查 slot 的最小平均 raw cosine |
| `grpo_sim_raw_cosine_ratio_min` | 本次 encode 中通过检查 slot 的最小高 cosine patch 比例 |
| `vision_encoder_time_ms` | 仍计整个 encode 路径 wall time（skip 时应接近 0） |
| `cached_image_features` | 当 `vit_calls==0` 且 `vit_skipped>0` 时置 1 |

---

## 7. 实验方法（before / after）

推荐 workload：**diversified oracle** 或 **model refocus**（保证 turn1 cross-branch 有分化且有 similarity）。

### 7.1 Baseline（关 cache）

```bash
VTOOL_ORACLE_DIVERSIFY=1 \
SGLANG_GRPO_SIM_CACHE=0 \
bash verl_vision/examples/profile/workloads/chart/run_rollout_profile.sh
```

### 7.2 Treatment（开 cache）

```bash
VTOOL_ORACLE_DIVERSIFY=1 \
SGLANG_GRPO_SIM_CACHE=1 \
SUFFIX=vtool_chart_bs64_n4_diversified_grpo_cache \
bash verl_vision/examples/profile/workloads/chart/run_rollout_profile.sh
```

### 7.3 对比脚本

```bash
python3 verl_vision/examples/profile/shared/analysis/compare_grpo_cache_timing.py \
  --baseline-vision profile_logs_vtool_chart_diversified/vision_encoder_log_*_diversified.csv \
  --cached-vision profile_logs_vtool_chart_diversified/vision_encoder_log_*_grpo_cache.csv \
  --baseline-generate profile_logs_vtool_chart_diversified/verl_sglang_generate_log_*_diversified.csv \
  --cached-generate profile_logs_vtool_chart_diversified/verl_sglang_generate_log_*_grpo_cache.csv
```

输出：总 vision 耗时、turn1 vision 耗时、总 `generate_e2e_ms`、turn1 `generate_e2e_ms`、`sglang_call_ms`、节省比例、`grpo_sim_vit_skipped` / exact / similarity 复用累计。

### 7.4 预期现象（n=4）

每个 GRPO group：

- 第 1 个 branch turn1：`vit_calls=2`，populate cache
- 第 2–4 个 branch turn1：slot0 exact 时 skip；slot1 通过 raw patch cosine 时 skip

粗算 turn1 vision 工作量：

- 若只复用 slot0：从 8 个 image slot ViT → 5 个 image slot ViT，turn1 vision 上限节省约 37.5%；
- 若 slot1 也通过检查：从 8 个 image slot ViT → 2 个 image slot ViT，turn1 vision 上限节省约 75%；
- 端到端收益还会被文本 prefill、decode、调度和日志 IO 稀释，需要看 `generate_log` / `vision_encoder_log` 的总耗时对比。

---

## 8. 已知限制与 review 要点

请 Codex 重点检查：

1. **正确性 vs 速度权衡**
   - slot0 是严格 exact 复用；
   - slot1 是在线检查 raw patch cosine 后复用，仍是近似 rollout；
   - 如果 hit rate 或质量不达预期，优先调阈值或回看 offline ViT token cosine，而不是切换另一套机制。

2. **Donor 选择**
   - 当前是 **first-write-wins**（谁先 populate 谁当 donor），后续 miss 后重算 ViT 也不会覆盖 donor；非 “组内最优” 或 “最高 similarity” donor。

3. **进程内 state**
   - `_REQUEST_META` / `_GROUP_CACHE` 会在 `global_step` 变化时清空；同时有 LRU 上限。
   - 多 replica 之间 **不共享** cache（与 exact cache 相同局限；同一 batch 同 worker 内有效）。

4. **切分边界**
   - `split_images_by_grid` 用 `T*H*W` patch 计数；若 `pixel_values` 行数对不齐则 fallback 为整段单图（cache 粒度退化，但不 crash）。

5. **Meta 缺失 fallback**
   - 无 `agent_uid` 或无 `hash_to_rid` 时 fallback 到原始 ViT，`_STATS["no_meta"]++`。
   - 无有效 `global_step` 时 fallback 到原始 ViT，避免跨 step 误复用。
   - `hash_to_rid` 现在不再依赖 `SGLANG_LOG_INFERENCE_STEP=1`；只开 `SGLANG_GRPO_SIM_CACHE=1` 也能生效。

6. **未覆盖场景**
   - turn0 cross-branch 复用（本应 exact dedup，但交给现有 hash cache + 本机制未专门处理）
   - 同 branch turn0→turn1 partial reuse（multi-turn 时间维）
   - DeepEyes / 其他 workload（机制通用，但只在 chart rollout 脚本里接了 env）

---

## 9. 相关 prior work（仓库内）

| 路径 | 关系 |
|------|------|
| `examples/profile/shared/analysis/analyze_similarity_unified.py` | 证明 cross-branch turn1 token 高相似 |
| `examples/profile/shared/analysis/run_phase2_token_replacement.py` | 证明高 sim token 替换 functional OK |
| `sglang/srt/mem_cache/multimodal_cache.py` | 现有 exact hash cache（本改动不替换） |
| `examples/profile/shared/reports/generate_multiturn_request_flow_report.py` | turn / request_id 对齐说明 |

---

## 10. 一句话总结

> 在 Qwen2.5-VL `get_image_feature` 层，按 GRPO `agent_uid` 缓存 turn1 各 image slot 的 ViT embedding；同组后续 branch 对 exact 图直接复用，对 refocus 图先做 raw patch cosine 检查再复用，用 profile log 量化 before/after vision 和端到端耗时。

---

*文档版本：与当前 workspace 实现同步；如有代码 drift 以 `grpo_similarity_cache.py` 为准。*
