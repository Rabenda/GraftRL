# VLM-CacheBlend：第 6 步 LLM Prefill 阶段的视觉 KV 选择性复用

> 目标：把 CacheBlend（RAG / 纯文本场景的非前缀 KV 复用 + 选择性重算）迁移到
> Qwen2.5-VL 的 GRPO rollout，针对 **turn1 prefill** 阶段复用 group 内 donor 分支
> 已经算好的 **image-token 逐层 KV**，只重算少量发生变化的视觉 token，从而降低
> prefill / TTFT，同时通过重算恢复 cross-attention 以保住生成质量。
>
> 本文档 = 系统设计 + 最小代码接口规划。实现见
> `sglang_vision_profile/python/sglang/srt/mem_cache/vlm_cacheblend.py`
> 与 `models/qwen2.py` 中宏开关 `SGLANG_VLM_CACHEBLEND` 包裹的 hook。

---

## 0. 与 ViT 1–5 步加速的关系（边界）

- 1–5 步（ViT 内部 partial reuse：window / token / token_sparse / merged）全部挂在
  总开关 `SGLANG_GRPO_SIM_CACHE` 后面（`qwen2_5_vl.py:1563`）。该开关=0 → 走原始
  `_visual_encode`，**即论文 baseline 的「no cache」版本**。
- 本设计是**独立的第二条线**，开关 `SGLANG_VLM_CACHEBLEND`，作用在 §6（LLM），
  与 ViT 复用互不依赖。论文主实验：`SGLANG_GRPO_SIM_CACHE=0` + `SGLANG_VLM_CACHEBLEND=1`
  对比纯 baseline（两者都 0）。

---

## 1. turn1 序列结构分析：为什么 RadixCache 不够

GRPO 一个 group 内 4 个 branch 共享 `agent_uid`。turn1 的 prompt 形如：

```
turn1 prompt (branch b) =
  [ system + turn0 原图 ]      ← 段 A：4 个 branch 完全相同（真前缀）
  [ turn0 response_b ]         ← 段 B：各 branch 采样不同 → 前缀在此断裂
  [ refocus image_b (slot1) ]  ← 段 C：各 branch 相似但不同，且是"非前缀"
  [ turn1 instruction text ]   ← 段 D
```

### RadixCache 能复用什么、不能复用什么

SGLang 的 `RadixCache`（`mem_cache/radix_cache.py`）是**前缀树**：只能命中从序列
开头开始、**逐 token 完全相同**的最长公共前缀。

- 段 A：所有 branch 相同 → RadixCache **能**命中，KV 直接复用。✅
- 段 B：`turn0 response_b` 各 branch 采样不同 → 第一个不同 token 处前缀树分裂。
  从这里往后，RadixCache **完全失效**。❌
- 段 C（refocus image tokens）：位于段 B 之后，所以**即使 branch 间视觉内容高度相似**
  （离线已证 merged image-token cosine mean≈0.888、76% token 过 0.90 阈值），
  RadixCache 也**无法复用**——因为它不是前缀，且要求 byte 级相同。❌

这正是 CacheBlend 的动机：**非前缀、内容近似的 chunk，前缀缓存无能为力，需要
"内容近似即可复用 KV + 选择性重算修正" 的机制。** 我们把 CacheBlend 的 "text chunk"
换成 "refocus image-token 段"。

> 注：段 C 的视觉相似性来自 GRPO 同组同源图像（turn0 同一张图 → refocus 到相近区域），
> 这是 paper 的核心 redundancy 来源，已由 Part-1 的 merged token similarity + relaxed
> accuracy sweep 支撑，本阶段不再做大规模 feasibility probing。

---

## 2. donor / recipient 机制

### 2.1 角色划分（复用现有 GRPO group 基础设施）

- group key = `(step, agent_uid, agent_turn=1)`，由
  `resolve_agent_meta_for_items(items, hash_to_rid)` 解析（已存在于
  `grpo_similarity_cache.py:347`）。
- **donor**：group 内第一个进入 turn1 prefill 的 branch（branch0）。走**完整 prefill**，
  额外把 refocus image-token 段的 **逐层 K/V** 落到 `DonorKVStore`。
- **recipient**：同 group 后续 branch（branch1-3）。prefill 时查询 donor KV，
  对 image-token 段做 **selective KV reuse**：大部分 token 直接用 donor KV，
  少量 HKVD（high-KV-deviation）token 重算。

### 2.2 donor 捕获：从 KV pool 读，而不是改 kernel

donor 分支正常 prefill 后，K/V 已由 attention backend 写入
`MHATokenToKVPool`，token 槽位 = `forward_batch.out_cache_loc`。因此 donor 捕获
**不需要改 attention kernel**，只需在每层 attention 之后，按 image-token 的
out_cache_loc 槽位 `get_key_buffer(layer)/get_value_buffer(layer)` 取出切片存起来：

```
DonorKVStore.record_layer(group_key, layer_id,
    k = key_buffer[image_slot_locs],     # [n_img_tok, n_kv_head, head_dim]
    v = value_buffer[image_slot_locs],
    positions = mrope_positions[:, image_token_index])  # 段 C 的绝对位置
```

低风险、对 baseline 零侵入（仅在 donor 角色 + 宏开启时执行只读拷贝）。

### 2.3 recipient 复用：选择性重算

recipient 的 image-token 段，逐层执行（CacheBlend 算法的 VLM 版）：

1. 取 donor 该层 K/V（按 §3 做位置对齐）。
2. 选出本层需要重算的 HKVD token 子集（§4）。
3. 非 HKVD token：直接采用 donor K/V（跳过其 qkv_proj / MLP / query attention）。
4. HKVD token：用 recipient 真实 hidden 重算 Q/K/V，attention 时 query 看到
   **完整 K/V = donor(非 HKVD) ⊕ 重算(HKVD)**，cross-attention 得以恢复。
5. 该层最终写入 pool 的 image-token KV = donor(非 HKVD) ⊕ 重算(HKVD)，
   保证后续 decode 的 KV 完整正确。

> 计算节省 ≈ 段 C 上 `(1 - r)` 比例 token 的 qkv_proj + MLP + 注意力 query 开销，
> 其中 `r` = 重算比例。refocus 改动局部 → `r` 期望落在 CacheBlend 的 10–20%。

---

## 3. position alignment（VLM 的核心 nuance）

段 C 的绝对位置 = 段 A+B 长度。**各 branch 的 `turn0 response_b` 长度不同
→ 段 C 起始位置不同 → 跨位置复用。** 这是 image-token KV 复用受位置影响的根因。

Qwen2.5-VL 用 **mRoPE**（`forward_batch.mrope_positions`，shape `(3, seq_len)`，
`qwen2_5_vl.py:1677`）。RoPE/mRoPE 的关键性质（CacheBlend 附录 A）：**attention 分数
只依赖 query/key 的相对位置**。已写入 K 的旋转可以"撤销 donor 位置、重打 recipient 位置"：

```
k_unrot   = R(donor_pos)^{-1} · k_donor      # 抵消 donor 的旋转
k_aligned = R(recipient_pos) · k_unrot       # 打上 recipient 的位置
         = R(recipient_pos - donor_pos) · k_donor   # 合并为一次相对旋转
```

对 mRoPE，三个 section（t/h/w）各自按对应 `Δpos` 旋转。V 不带位置，无需处理。

### 实现分两档（宏 `SGLANG_VLM_CACHEBLEND_POS_MODE`）

- `same`（v1，默认）：要求 donor/recipient 的段 C **绝对位置一致**（即 turn0
  response 等长，或 prompt 对齐到同长）。此时 `Δpos=0`，**完全不需要旋转**，
  K 直接复用。最稳，先打通端到端。
- `rerotate`（v2）：donor/recipient 段 C 位置不同时，按上式对 donor K 做 mRoPE
  相对旋转。接口已在 `vlm_cacheblend.py:rerotate_keys_mrope()` 预留。

> v1 的 `same` 模式落地建议：rollout 侧让同 group 的 turn0 response 走相同
> max_new_tokens 且右侧 pad，使段 C 对齐；或仅在检测到位置一致的 (donor,recipient)
> 对上启用复用，不一致则 fallback 到完整 prefill（正确性优先）。

---

## 4. selective recompute：选哪些 token 重算

入口：`select_recompute_tokens(...) -> recompute_mask`（`vlm_cacheblend.py`）。
支持三种 `SGLANG_VLM_CACHEBLEND_SELECT` 模式，可叠加：

1. `topr`（fixed top-r%）：按某个 deviation 分数排序，取前 `r%`（`...RECOMPUTE_RATIO`）。
   最可控，便于做 r–质量 sweep。
2. `kvdev`（KV deviation，CacheBlend 原法）：先在**第 0/1 层**用 recipient 真实
   hidden 算出 image-token 的 K，与 donor 对齐后求逐 token 偏差
   `‖k_recipient − k_donor_aligned‖`，取偏差最大的一批；后续层沿用并逐层收窄
   （CacheBlend 的 gradual filtering / Insight 2）。
3. `sim`（visual similarity 先验）：直接用 Part-1 已有的 merged/patch token 相似度
   作为 deviation 代理——相似度低的 token 即认为变化大、需重算。最便宜（无需第 0 层
   bootstrap），但依赖 ViT 侧相似度信号可得。

固定保护项：
- slot0（turn0 原图，若也纳入复用范围）byte 相同 → 重算 0%。
- recompute_ratio 上限 / 下限 clamp；`r=1.0` 等价完整 prefill（用于 A/B 校验）。

融合：`blend_kv(donor_k, donor_v, recomp_k, recomp_v, recompute_mask)`：
recompute_mask 处取重算值，其余取 donor（已位置对齐）值。

---

## 5. metrics（不再看 vision_encoder_time_ms）

prefill 阶段指标（写入 profile CSV / stats）：

| 指标 | 含义 | 采集点 |
|---|---|---|
| `extend_wall_ms` | turn1 prefill（extend）墙钟 | model forward 计时 |
| `ttft_ms` | time-to-first-token | scheduler 侧已有/补桩 |
| `queue_ms` | 请求排队时间 | scheduler |
| `cacheblend_recompute_ratio` | 实际重算 token 占段 C 比例 `r` | vlm_cacheblend stats |
| `cacheblend_reused_tokens` | donor KV 直接复用的 token 数 | 同上 |
| `cacheblend_pos_mode` / `select_mode` | same/rerotate；topr/kvdev/sim | 同上 |
| `relaxed_acc` | 答案正确率（ChartQA relaxed） | 下游 eval |
| `answer_change_rate` | 复用 vs baseline 答案变化比例 | 离线 diff |
| `correct_to_wrong_rate` | 复用导致由对变错的比例（**质量红线**） | 离线 diff |

质量验收口径：在某个 `r` 下，`extend_wall/TTFT` 显著下降，且
`correct_to_wrong_rate` ≈ 0、`relaxed_acc` 不下降。

---

## 6. implementation plan：SGLang 接入点

### 6.1 调用栈与 hook 位置

```
Qwen2_5_VLForConditionalGeneration.forward            (qwen2_5_vl.py:1656)
  └─ general_mm_embed_routine  → 把视觉 embedding 填入序列
  └─ self.model = Qwen2Model.forward                  (qwen2.py:335)  ← 【donor/recipient 调度入口】
       for layer in layers:
         Qwen2DecoderLayer.forward                    (qwen2.py:235)
           └─ Qwen2Attention.forward                  (qwen2.py:178)  ← 【捕获/复用 K/V 的最小粒度】
                qkv_proj → split → rotary_emb(pos,q,k)
                self.attn(q,k,v,fb)  = RadixAttention  (radix_attention.py:96)
                  └─ unified_attention_with_output → backend.forward_extend
                       → MHATokenToKVPool.set_kv_buffer(layer, out_cache_loc, k, v)  (memory_pool.py:904)
       o_proj
```

接入策略（宏 `SGLANG_VLM_CACHEBLEND`，默认 0 → baseline 完全不变）：

- **donor 捕获（v1，低风险，已可正确实现）**：在 `Qwen2Model.forward` 末尾（或每层
  attention 之后），对 donor 角色读取 `token_to_kv_pool.get_key/value_buffer(layer)`
  的 image-token 槽位切片，存入 `DonorKVStore`。只读拷贝，不改 kernel。
- **recipient 复用（核心，需 GPU 在环验证）**：在 `Qwen2Model.forward` 用一个
  `cacheblend_forward()` 替代默认逐层循环（仅 recipient + 宏开 + 命中 donor 时）：
  - same 模式 v1：先实现 "donor KV 全量写回 + 仅段 C 的 HKVD token 重算" 的正确版本；
    非 HKVD token 不再过 qkv/MLP/attention，其 KV 取 donor。
  - 需要 attention backend 支持"变长 query 子集 + 外部 K/V 拼接"。FlashAttention/Triton
    backend 可通过自定义 metadata 实现；首版可先用 torch_native_backend 做正确性参照。

### 6.2 复用 LMCache / CacheBlend 思路 vs VLM-specific

| 部分 | 复用 CacheBlend/LMCache | VLM-specific（本项目新写） |
|---|---|---|
| 非前缀 KV 复用 + 选择性重算骨架 | ✅ 直接借鉴（LMCache `blend` 模块） | — |
| RoPE 相对旋转修正位置 | ✅ 公式与 LMCache 一致 | mRoPE 三段 (t/h/w) 旋转适配 |
| HKVD / gradual filtering 选择 | ✅ kvdev 模式 | `sim` 模式用 ViT 视觉相似度先验 |
| chunk = ？ | text chunk | **image-token 段**（按 grid 切分） |
| group / donor 身份 | LMCache 按 hash | 复用 GRPO `agent_uid`+turn+`hash_to_rid` |
| 触发时机 | RAG 请求 | GRPO turn1、同 group 跨 branch |
| 质量指标 | F1/Rouge | relaxed_acc / correct_to_wrong |

### 6.3 里程碑

- **M1（本次）**：设计文档 + `vlm_cacheblend.py` 核心算法（store / 位置对齐 /
  selective select / blend，纯张量、可单测）+ 宏开关 + donor 捕获 hook +
  recipient same-position 调度骨架（OFF by default）。
- **M2（需 GPU 在环）**：recipient selective-recompute 的 attention backend 集成，
  打通 same 模式端到端正确性（对拍完整 prefill 的 logits）。
- **M3**：cross-position（rerotate）+ kvdev/sim 选择 + r–质量 sweep + TTFT 实测。

### 6.4 当前代码状态（2026-06-24）

已落地：

- `model_runner.py` 在 `SGLANG_VLM_CACHEBLEND=1` 时为**单请求 extend batch**设置
  VLM-CacheBlend request context；多请求 batch 明确回退为 `multi_request_batch`，避免
  在 flatten 后的 `input_ids` 上误判 image span。
- 默认复用目标为 `SGLANG_VLM_CACHEBLEND_TARGET_IMAGE_SLOT=-1`，即 turn1 prompt 里的
  最后一个 image span（refocus 图）。这比捕获所有 `<image>` token 更贴近实验问题。
- donor 分支完成 full prefill 后，从 KV pool 捕获目标 image span 的逐层 K/V。
- recipient 分支暂时仍走 full prefill，但会检查 donor 是否命中、grid/token 数是否一致、
  `pos_mode=same` 下位置是否一致，并在 `model_forward_log*.csv` 输出：
  `cacheblend_role/cacheblend_eligible/cacheblend_fallback_reason/cacheblend_reused_tokens/
  cacheblend_recomputed_tokens`。

尚未落地：

- recipient fast path 还没有替代默认 layer loop；因此当前 `SGLANG_VLM_CACHEBLEND=1`
  **不会降低 TTFT/extend wall time**，只能验证 donor/recipient 配对与复用前置条件。
- 真正跳过非 HKVD image token 需要额外保存/复用层间 hidden/residual，不能只靠逐层 K/V；
  否则下一层无法为 skipped token 产生正确 hidden state。这是 M2 需要解决的核心接口。

---

## 7. 正确性与回退原则

- 宏关闭（默认）：代码路径与原始 SGLang 完全一致。
- 任一前置条件不满足（group 未命中 / grid 不一致 / 位置不一致且 pos_mode=same /
  donor 未就绪）→ **fallback 到完整 prefill**，绝不产出错误 KV。
- `r=1.0` 必须与完整 prefill **数值一致**（实现自检用）。
