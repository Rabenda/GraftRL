# VLM-CacheBlend / Visual KV Reuse —— 实现核对报告

> 目的：核对 "VLM 版 CacheBlend / visual KV reuse"（GRPO rollout 中 turn1 refocus image 的 LLM KV 复用）是否按设计实现。
> 范围：`/workspace/repo/sglang_vision_profile`（推理引擎）+ `/workspace/repo/verl_vision`（GRPO 训练 / rollout）。
> 核对方式：直接读码（人工）+ 3 个并行 Explore agent 交叉验证。所有结论都附 `file:line`，可独立复核。
>
> **给复核者（codex）的提示**：请逐条打开下面引用的 `file:line` 验证。重点复核 §四 的"实质落差 1"（高风险 token 选择是否真退化成按位置前 15%）和 §五（对第三个 agent verdict 的纠正是否成立）。

---

## 一、总判断

1. 你设计的 6 步**结构性地全部落地了**，而且是**真正接到了 SGLang 的 model forward / prefill 路径**——不是纸面代码、不是纯日志、不是只在单测里跑。KV pool 会被真实改写，attention kernel 会真实跳过被复用 token 的 query。
2. 两层（verl rollout → sglang engine）通过 `agent_uid / agent_turn / global_step` 元数据**真实打通**，挂在**线上训练 rollout 路径**上，不是 `examples/profile` 的离线分析脚本。
3. 但有 **1 个实质性落差** + **2 个需要知道的现实约束**：
   - **实质落差**：第 5 步"只对一部分**高风险** token 做 selective recompute"——默认配置下退化为"**按位置重算前 15%**"，并没有用 KV 偏差挑高风险 token。真正的高偏差选择（`kvdev` / `sim`）写好了但**没被喂数据**。
   - 约束 A：默认 `pos_mode="same"` 要求 donor/recipient 的 refocus image 绝对位置完全一致（Δ=0），turn0 response 长度分叉会让位置错位，命中率可能很低；处理位置差的 `rerotate` 默认不开、标注 experimental。
   - 约束 B：整套默认 `SGLANG_VLM_CACHEBLEND=0` 关闭；且核心文件 `vlm_cacheblend.py` **未提交 git**（untracked），`sglang_vision_profile.patch` 里**只有 profiling 日志**部分，没有 cacheblend 逻辑。

---

## 二、逐条对照 6 步需求

| # | 你的需求 | 状态 | 关键证据（file:line） |
|---|---|---|---|
| 1 | turn1 第一支判 **donor**，定位 refocus image 的 LLM image-token span | ✅ 已实现 | 角色判定 `role = "recipient" if donor.complete else "donor"`；span 通过 image pad_value / image_token_id 扫描定位：`vlm_cacheblend.py:1022-1050`（`_token_span_count` / `_image_span_count_from_req`）；forward 内调用 `qwen2.py:514`（`_cacheblend_locate_image_tokens`） |
| 2 | 抓该 span 的 **per-layer K/V**，按 `(global_step, agent_uid, agent_turn, image_slot, grid_sig)` 存 | ✅ 完全吻合 | 键构造 `(int(global_step), str(agent_uid), int(agent_turn), int(image_slot), tuple(grid_sig))` → `vlm_cacheblend.py:954-963`；捕获 `qwen2.py:704`（`capture_donor_kv`）；存储 `DonorEntry.record_layer` / `DonorKVStore`（LRU）→ `vlm_cacheblend.py:168-185, 220-275` |
| 3 | 后续支判 **recipient**，定位自己的 refocus span | ✅ 已实现 | `qwen2.py:507-525`（遍历 ctx，`role=="recipient"` 时定位 span 并建 plan） |
| 4 | 比较 donor/recipient 的 **grid / image-token count / mRoPE position** | ✅ 三项都在 | `build_recipient_kv_blend_plan` → `vlm_cacheblend.py:503`（`grid_mismatch`）、`:506`（`image_token_count_mismatch`）、`:509`（`position_mismatch`，`positions_match` 精确比对 mRoPE） |
| 5 | 条件满足则**复用 donor KV**，只对**一部分高风险 token** selective recompute | ⚠️ **部分实现 / 默认路径降级** | 复用真实：`apply_recipient_kv_blend_for_layer` 写 `k_buf[dst]=donor_k_reuse`（`vlm_cacheblend.py:566-628`）；attention/QKV/MLP 真跳过（见 §三）。**但"高风险"选择默认取前 15% 按位置**（见 §四 落差 1） |
| 6 | 不改语义、减少 turn1 refocus 的 **LLM prefill 成本** | ✅ 机制存在；语义安全性取决于配置 | `recipient_active_query_ranges` 真把复用 token 的 query 从 prefill kernel 剔除（`vlm_cacheblend.py:822-914`）；默认关闭 `SGLANG_VLM_CACHEBLEND=0` |

---

## 三、调用链（确认是真 forward 路径，不是死代码）

verl 侧：
- `agent_uid`（每条 trajectory 一个 UUID）、`agent_turn`（`assistant_turns` 计数）：`examples/profile/shared/agent/vtool_agent_loop.py:169, 206-249`
- turn0 response 分叉后，turn1 各支共享 refocus image：`vtool_agent_loop.py:291-308`（工具产出 `edited_image` 后 `images.append(edited_image)`）
- 把 `agent_uid / agent_turn / training_global_step` 写进 sglang：`verl/workers/rollout/sglang_rollout/async_sglang_server.py:726-757, 764-772`（`register_request_meta(...)`）

sglang 侧（forward 内）：
- forward 前设 context：`model_executor/model_runner.py:416`（`set_request_context(ctx)`），forward 后清空 `:421, :430`
- recipient 快路在 **decoder layer loop 之前** 建 plan：`models/qwen2.py:444`（`_cacheblend_prepare_recipient_fast_path`），plan 定义见 `:487-529`
- 每层 attention backend 消费 plan 改 KV：
  - `layers/attention/flashattention_backend.py:753`（`apply_recipient_kv_blend_for_layer`）、`:884-885`（`recipient_active_query_ranges` 跳 query）
  - `layers/attention/flashinfer_backend.py:774, 835`
  - `layers/attention/triton_backend.py:815`
  - `layers/attention/torch_native_backend.py:217`
- donor 捕获：`models/qwen2.py:704`（`capture_donor_kv`）
- 计算跳过（默认开）：
  - QKV/O projection skip：`qwen2.py:190-210, 245`（`skip_reuse_qkv_proj`）
  - MLP skip：`qwen2.py:317-341`（`skip_reuse_mlp`）
  - attention query skip：`flashattention_backend.py:884-927`（只对 active query 跑 `flash_attn_with_kvcache`）

开关（默认值）：
- `SGLANG_VLM_CACHEBLEND=0`（总开关，默认**关**）→ `vlm_cacheblend.py:107, 140`
- `SGLANG_VLM_CACHEBLEND_POS_MODE=same`（默认；`rerotate` 为 experimental）→ `:66, 108`
- `SGLANG_VLM_CACHEBLEND_SELECT=topr`（默认）→ `:68, 109`
- `SGLANG_VLM_CACHEBLEND_RECOMPUTE_RATIO=0.15` → `:69, 110`
- `SKIP_REUSE_MLP / SKIP_REUSE_QKV_PROJ / SKIP_REUSE_ATTENTION` 默认 `1`（开）→ `:117-123`

---

## 四、实质落差（请重点复核）

### 落差 1（核心）：默认配置下，"高风险 token"选择退化为"按位置取前 15%"

- 选择函数 `select_recompute_tokens`：`vlm_cacheblend.py:393-442`
  - `topr` 模式（默认）：**有 `deviation` 时**取偏差 top-r%；**没有 `deviation` 时**走 fallback `mask[:k]=True`，即**重算前 k 个（前 15%）token**（`:430-435`）。
  - `kvdev` 模式：按 KV 偏差 top-r%（需 `deviation`，由 `kv_deviation()` 算，`:445-456`）。
  - `sim` 模式：按相似度阈值（需 `similarity`）。
- **问题**：唯一构建 plan 的地方 `build_recipient_kv_blend_plan`（`vlm_cacheblend.py:516-520`）调用
  ```python
  recompute_mask = select_recompute_tokens(int(img_locs.numel()), cfg, device=img_locs.device)
  ```
  **没有传 `deviation` 也没有传 `similarity`**。donor 捕获阶段（`capture_donor_kv`, `qwen2.py:704-711`）也**不算 deviation**。
- **结论**：默认 `select_mode="topr"` + 无 deviation → 落到 `mask[:k]=True` → **重算的是"开头 15%"，不是"最该重算的 15%"**。CacheBlend 原意的 HKVD（高 KV 偏差集）`kv_deviation()` / `kvdev` / `sim` **作为函数写好了，但 wired 路径从未喂给它所需张量**，因此默认不生效。
- **影响**：默认行为 = "复用 85% donor KV + 重算前 15% token"，这与你 step 5 "只对一部分**高风险** token recompute" 的语义不符，可能影响语义正确性（step 6）。
- **修复方向**：在 `build_recipient_kv_blend_plan` 引入一个 bootstrap 层先算 recipient 的 K，用 `kv_deviation(recipient_k, donor_k_aligned)` 得到偏差，再传给 `select_recompute_tokens(..., deviation=...)`，并把默认 `select_mode` 切到 `kvdev`。

### 落差 2：`pos_mode="same"` 默认命中条件苛刻

- `build_recipient_kv_blend_plan` 在 `pos_mode=="same"` 时要求 `positions_match(donor.positions, img_positions)` 为真（`vlm_cacheblend.py:509`；`positions_match` 用 `torch.equal` 精确比较，`:285-295`）。
- turn0 response 各支长度不同 → refocus image 绝对 token 位置通常**不一致** → 命中 `position_mismatch` fallback → **不复用**。
- 处理位置差的 `rerotate`（mRoPE 重旋转，数学：inverse-donor ∘ recipient 复合旋转，`vlm_cacheblend.py:308-364`）**默认不开、标 experimental**（`:331-332`）。
- **需确认**：实际 prompt 是否把 refocus image 对齐到固定位置；否则默认配置命中率可能很低。

---

## 五、对"第三个 Explore agent 顶层 verdict"的纠正（请一并复核）

第三个 agent 的 headline 结论为 *"recipient requests run full prefill / 初始 PREFILL 不 blend / 仅 EXTEND 模式"*。**这是误读**，理由（已读码确认）：

1. **注释挂错了方法**。`qwen2.py` 有两个 hook：
   - `_cacheblend_prepare_recipient_fast_path`（定义 `:487-529`，在 **layer loop 之前** 的 `:444` 调用）——**这才是真正在跑的 recipient 路径**，它在层循环前就 `build_recipient_kv_blend_plan` + `set_recipient_blend_plans`，供各层 attention backend 消费。
   - `_maybe_cacheblend_after_full_prefill`（`:464`，layer loop **之后**）——注释 `:460-463` "recipient fast path intentionally not enabled **here** yet" 的 **"here" 指这个事后 hook**，意思是"事后 hook 只做 donor capture + 资格探测，不在这里 blend"（blend 已在 `:444` 提前做）。agent 把它读成"整个 recipient 复用没开"，错。

2. **"EXTEND-only" 不等于"排除 turn1"**。`_cacheblend_prepare_recipient_fast_path:494` 有 `if not forward_batch.forward_mode.is_extend(): return`。但在 **SGLang 语义里，新请求的 prefill 本身就是 EXTEND mode**（EXTEND = 在 radix 前缀上 prefill 新 token；DECODE 才是逐 token 解码）。turn1 refocus prompt 的 prefill 正是一次 extend → `is_extend()` 为 True → **recipient 复用会触发**。该 gating 的作用是"只在 prefill 触发、跳过 decode"，**不是把 turn1 排除掉**。agent 的 "Initial PREFILL ≠ EXTEND 所以不 blend" 对 SGLang 不成立。

> 注：第三个 agent 在**机制细节**（attention/QKV/MLP skip、`blend_kv`、`rerotate` 均为真实、默认开）上与人工读码一致，仅**顶层 verdict 误判**。

---

## 六、结论与建议

- "是否实现了"：**基本实现了，且是可运行的实验路径**。donor 捕获、5-tuple 键、grid/token-count/mRoPE 三重校验、KV 真改写、attention query 真跳过、mRoPE 重旋转——都是真代码、真接 forward，turn1 prefill 上 recipient 复用会真触发（开关打开时）。
- 若以"严格符合 CacheBlend 语义 + 默认稳定命中"为标准，仍差两块：
  1. **selective recompute 的 token 选择没接高偏差逻辑**（默认退化前 15%）——**优先补**（落差 1）。
  2. **默认 `pos_mode="same"` 命中条件太苛刻**——需验证 prompt 是否做了位置对齐，或转用并验证 `rerotate`（落差 2）。
- 次要：整套默认关闭，且 `vlm_cacheblend.py` 为未提交 working-tree 代码（`sglang_vision_profile.patch` 仅含 profiling 日志）。

### 建议下一步（二选一或都做）
- **(A)** 把 `kvdev`（真·高偏差选择，含 bootstrap 层算 `kv_deviation`）接进 `build_recipient_kv_blend_plan`，默认 `select_mode=kvdev`——让实现符合"只重算高风险 token"原意。
- **(B)** 拉 `profile_logs_*_kvblend*` 日志，核对实际 `cacheblend_used / fallback_reason / reused_tokens / recompute_ratio`，量化当前命中率与 prefill 节省。

---

## 附：核对者快速验证清单（codex 可直接跑）

```bash
cd /workspace/repo/sglang_vision_profile
# 1. 5-tuple 键
sed -n '954,963p' python/sglang/srt/mem_cache/vlm_cacheblend.py
# 2. 四重资格校验
sed -n '499,514p' python/sglang/srt/mem_cache/vlm_cacheblend.py
# 3. 落差1：select_recompute_tokens 的 fallback 前 r%
sed -n '393,442p' python/sglang/srt/mem_cache/vlm_cacheblend.py
# 4. 落差1：build plan 调用时未传 deviation/similarity
sed -n '516,520p' python/sglang/srt/mem_cache/vlm_cacheblend.py
# 5. recipient 快路（layer loop 之前）+ 被误读的注释
sed -n '444,529p' python/sglang/srt/models/qwen2.py
# 6. donor 捕获
sed -n '704,711p' python/sglang/srt/models/qwen2.py
# 7. attention query skip
sed -n '880,927p' python/sglang/srt/layers/attention/flashattention_backend.py
# 8. 核心文件未提交 + patch 仅 profiling
git status --short python/sglang/srt/mem_cache/vlm_cacheblend.py
grep -n 'training_global_step\|cacheblend' ../verl_vision/sglang_vision_profile.patch | head
```
