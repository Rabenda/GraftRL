# 局部视觉 Token 复用实验总结

## 1. 我们想证明什么

这个实验的目标是验证：

> 在多轮视觉推理中，图像经过 refocus / crop / 高亮等工具处理后，是否还能复用一部分视觉计算，而不是每次都重新跑完整个视觉编码器。

具体场景是：

1. 模型先看到原始图像。
2. 模型调用视觉工具，得到一个 refocus 后的新图像。
3. 新图像和原图不完全一样，但很多局部区域可能仍然相似。
4. 如果这些局部区域的视觉表示足够相似，理论上可以复用之前的视觉计算。

这件事的核心不是“整张图完全一样就跳过”，而是：

> 图像局部发生变化时，只重算变化区域，稳定区域尽量复用。

因此我们真正关心的是 **partial / local / token-level reuse**，不是 whole-image reuse。

一个成功的实验应该同时满足：

| 目标 | 含义 |
|---|---|
| 局部相似性成立 | refocus 后仍有很多局部视觉 token 相似 |
| 系统能识别可复用区域 | 能产生 token/window 级别的复用 mask |
| 系统真的执行了复用 | 不是只打日志，而是真的少算 |
| 语义不坏 | 答案质量不能明显下降 |
| 性能变快 | 节省的计算要大于复用带来的额外开销 |

## 2. Token、Window、整图分别是什么

这里有三个层级，不能混在一起：

| 层级 | 含义 | 和实验的关系 |
|---|---|---|
| Token | 视觉表示里的小单位，可以理解成局部图像特征 | 最贴近论文里“局部复用”的想法 |
| Window | ViT 内部把一组 token 放在一起做 window attention | 当前工程里更容易执行跳过计算 |
| 整图 / 整个 image slot | 整张图的最终视觉 embedding | 如果图像完全一样会很快，但不是本文主张 |

论文想讲的是 token 级局部冗余：

```text
refocus 后不是整张图都一样，但有些局部视觉 token 仍然相似，
这些局部 token 的视觉计算可能可以复用。
```

但工程实现会遇到一个问题：

```text
ViT 不是天然按单个 token 独立计算。
很多地方是按 window 或整段序列一起算。
```

所以我们先尝试了 window 级复用，后面又继续尝试 token 级复用。

## 3. 判断依据是什么

### 3.1 语义指标

复用不能只看速度。视觉输入已经变了，如果复用了错误的旧视觉特征，模型可能会更快但答错。

语义上主要看：

| 指标 | 含义 |
|---|---|
| token similarity | 局部 token 是否相似 |
| window similarity | window 内整体是否相似 |
| reuse mask | 哪些 token/window 被复用了 |
| reward / accuracy | 最终答案是否变差 |
| case study | 复用区域是否确实是稳定区域 |

这次 Chart 任务的原始 reward 有一个问题：它只认 `<answer>...</answer>` 或 `\boxed{...}`，但 rollout 里模型经常输出 `FINAL ANSWER: xxx`。所以原始 `score_mean=0` 不能直接说明答案全错。

为此我们在汇总脚本里加了宽松答案匹配：

```text
relaxed_acc_mean
relaxed_positive_rate
```

它会从 `FINAL ANSWER:` / `ANSWER:` / `<answer>` 中抽最终答案，再和 ground truth 做归一化匹配。

### 3.2 性能指标

性能上主要看：

| 指标 | 含义 |
|---|---|
| `partial_vit_used` | partial reuse 路径是否真的跑了 |
| `reused_windows` | 复用了多少 window |
| `reused_tokens` | 复用了多少 token |
| `reuse_ratio` / `token_reuse_ratio` | 复用比例 |
| `vision_encoder_time_ms` | 视觉编码器实际耗时 |
| `computed_token_layer_tokens` | 理论上还需要计算多少 token-layer |

一个真正有用的结果应该长这样：

```text
partial_vit_used > 0
reused_tokens 或 reused_windows 明显大于 0
vision_encoder_time_ms 下降
relaxed_acc_mean 不明显下降
```

## 4. 实验数据和路径

主要 Chart refocus 数据：

```text
/data/refocus_chart_multiturn_oracle_changed/train.parquet
/data/refocus_chart_multiturn_oracle_changed/test.parquet
```

DeepEyes 数据：

```text
/data/deepeyes_visual_toolbox_v2/train.parquet
/data/deepeyes_visual_toolbox_v2/test.parquet
```

主要代码路径：

```text
/workspace/repo/sglang_vision_profile/python/sglang/srt/models/qwen2_5_vl.py
/workspace/repo/sglang_vision_profile/python/sglang/srt/mem_cache/grpo_similarity_cache.py
```

主要脚本：

```text
/workspace/repo/verl_vision/examples/profile/workloads/chart/run_token_reuse_probe.sh
/workspace/repo/verl_vision/examples/profile/workloads/chart/run_merged_token_reuse_sweep.sh
/workspace/repo/verl_vision/examples/profile/workloads/chart/run_merged_window_e2e_sweep.sh
```

主要汇总脚本：

```text
/workspace/repo/verl_vision/examples/profile/shared/analysis/summarize_token_reuse_probe.py
/workspace/repo/verl_vision/examples/profile/shared/analysis/summarize_merged_token_reuse_sweep.py
/workspace/repo/verl_vision/examples/profile/shared/analysis/compare_token_sparse_timing.py
```

## 5. 第一阶段：Window 级复用

### 5.1 为什么先做 Window

Qwen2.5-VL 的 ViT 里有 window attention。工程上，如果一个 window 足够相似，就可以尝试跳过这个 window 在前面几层的计算，直接复用 donor 的中间 hidden states。

所以第一阶段做的是：

```text
判断 refocus 图和 donor 图在 window 级是否相似；
如果相似，就复用这个 window 的 prefull hidden states。
```

### 5.2 实验配置

典型配置：

```text
SGLANG_GRPO_SIM_CACHE=1
SGLANG_GRPO_REUSE_MODE=token_or_window_partial_reuse
SGLANG_GRPO_ENABLE_PARTIAL_VIT_REUSE=1
SGLANG_GRPO_PARTIAL_REUSE_GRANULARITY=window
SGLANG_GRPO_PARTIAL_REUSE_THRESHOLD=0.90
SGLANG_GRPO_SIM_RAW_COSINE_THRESH=0.90
SGLANG_GRPO_SIM_RAW_COSINE_RATIO=0.0
VTOOL_ORACLE_DIVERSIFY=1
```

输出目录示例：

```text
/workspace/repo/verl_vision/profile_logs_vtool_chart_window_positive/
```

### 5.3 Window 结果

观察到的 window 相似度大概是：

| 指标 | 数值 |
|---|---:|
| window cosine mean | 约 0.74 |
| window cosine max | 约 0.86 |
| threshold | 0.90 |

结果：

| 指标 | 结果 |
|---|---:|
| `partial_vit_used` | 0 |
| `reused_windows` | 0 |
| `total_windows` | 约 16004 |
| `reuse_ratio` | 0% |

解释：

```text
window 粒度太粗。
一个 window 里只要有局部变化，整个 window 的相似度就会被拉低。
所以 threshold=0.90 时没有 window 能通过复用条件。
```

降低阈值到 0.84 左右可以作为机制探针，证明代码路径能跑通，但这不代表 0.84 是一个安全或有效的最终阈值。

## 6. 为什么不把整图复用当主结论

整图复用确实可能很快，尤其是两张图 byte-identical 的时候。

但它不是这个工作的核心贡献，原因是语义风险太大：

```text
原图：完整 chart
refocus 图：局部放大或高亮后的 chart
整图复用：直接沿用旧 image embedding
风险：忽略 refocus 后新增或变清楚的局部证据
```

所以整图复用只能作为 baseline 或工程捷径，不能作为“局部 token 复用有效”的证据。

我们真正想证明的是：

```text
稳定区域可以复用；
变化区域必须重新计算。
```

## 7. 第二阶段：Merged Image Token 相似性

Window 失败后，我们回到最初的 token 级想法。

离线实验里，我们比较的是 **merged image token**：

```text
image -> 完整 ViT -> merger -> image tokens -> same-index cosine
```

这个层级就是 LLM 最终吃到的视觉 token。

### 7.1 Merged 层结果

我们实现了：

```text
granularity = "merged"
```

这条路径会：

1. 正常跑完整 target ViT。
2. 得到 target merged image tokens。
3. 和 donor merged image tokens 做 same-index cosine。
4. 对相似 token 做输出层替换。

阈值 sweep 结果：

| Threshold | Merged Token Reuse Ratio | Relaxed Accuracy |
|---:|---:|---:|
| 0.95 | 69.2% | 53.1% |
| 0.90 | 76.0% | 49.2% |
| 0.85 | 80.6% | 50.4% |
| 0.80 | 82.9% | 51.6% |
| 0.75 | 84.7% | 49.2% |

这证明了一件很重要的事：

```text
refocus 后，最终 merged image tokens 确实有大量局部冗余。
```

也就是说，最初的 token similarity 观察不是错的。

### 7.2 但是 Merged 层不能直接加速

问题是：

```text
merged image tokens 是完整 ViT + merger 之后才得到的。
```

所以在这个层级做复用，只能证明语义和冗余，不能节省 ViT 计算。

换句话说：

```text
高相似发生在太晚的层。
等我们知道它相似的时候，最贵的 ViT 已经算完了。
```

这就是后续工程困难的根源。

## 8. 第三阶段：真正少算的 Token Sparse 路线

为了真正 e2e 加速，我们实现了路线 B：

```text
granularity = "token_sparse"
```

它和之前的 probe 不一样。

| 路径 | 做法 | 是否真的少算 |
|---|---|---|
| `token` | 全部 token 都算完，再用 donor 覆盖 reused token | 否 |
| `merged` | 完整 ViT 跑完后，在输出层替换 merged token | 否 |
| `token_sparse` | reused token 不跑 early-layer FFN/MLP | 是，部分少算 |

### 8.1 token_sparse 的人话逻辑

`token_sparse` 做的是：
我们先比较新图和旧图里同位置的小图块。如果某些小图块看起来没变，就不重新计算它们，直接拿旧图里已经算好的结果；如果某些小图块变了，就只重新计算这些变了的小图块。为了不破坏语义，变了的小图块在计算时仍然可以参考那些没变的小图块的旧表示。
1. 在 ViT 早期，用 patch-hidden token 做 same-index cosine。
2. 相似度超过阈值的 token 标记为 reused token。
3. 在 prefull window 层：
   - changed token 重新计算；
   - reused token 使用 donor 的 hidden state；
   - changed token 做 attention 时仍然能看到完整 window；
   - reused token 跳过 FFN/MLP，直接拿 donor 的 layer output。

重点是：

```text
changed token 没有丢上下文。
它仍然能 attend 到 window 里所有 token。
只是 reused token 的 K/V 来源是 donor hidden state。
```

所以这不是简单删除 token，而是：

```text
保留上下文，跳过重复的 FFN 计算。
```

### 8.2 token_sparse 的结果

我们选择了较低阈值：

```text
TOKEN_REUSE_MODE=token_sparse
TOKEN_REUSE_THRESHOLD=0.80
```

测速结果：

| Run | Turn1 Rows | Mean Vision Time | Median Vision Time | Total Vision Time | Reused Tokens |
|---|---:|---:|---:|---:|---:|
| Baseline, cache off | 256 | 23.6 ms | 21.4 ms | 6.05 s | 0 |
| Token sparse, threshold 0.80 | 225 | 47.4 ms | 48.2 ms | 10.66 s | 32,089 / 916,236 |

复用率：

```text
32,089 / 916,236 = 3.50%
```

性能：

```text
23.6 ms -> 47.4 ms
```

结论：

```text
token_sparse 确实真的少算了 reused token 的 FFN，
但最终不但没加速，反而慢了大约一倍。
```

## 9. 为什么 token_sparse 会变慢

这不是简单 bug，而是结构性问题。

Qwen2.5-VL-7B-Instruct 的 ViT 配置：

```text
ViT depth = 32
fullatt_block_indexes = [7, 15, 23, 31]
hidden size = 1280
intermediate size = 3420
```

这意味着：

```text
只有 layer 0-6 是 prefull window 层。
token_sparse 只能在这前 7 层省一点计算。
后面的层仍然要正常跑。
```

图片
↓
1. patchify / patch_embed
   把图片切成小块，并变成最早期的 patch tokens
   **代码**: `Qwen2_5_VisionTransformer.forward` / `_prepare_window_ordered_hidden` → `self.patch_embed`
↓
2. ViT blocks 前几层
   对 patch tokens 做局部/window attention 和 MLP
   **代码**: `forward` 中 `layer_num not in fullatt_block_indexes`；partial reuse 在 `forward_with_partial_window_reuse` 的 `layer_num < first_fullatt` 分支
↓
3. full attention blocks
   让更远位置的 token 互相交流
   **代码**: `forward` 中 `layer_num in fullatt_block_indexes`（Qwen2.5-VL-7B: layers 7,15,23,31）
↓
4. merger
   把很多 patch-level tokens 合并成更少的 merged image tokens
↓
5. projector / 对齐到 LLM hidden size
   变成 LLM 能接收的视觉 embedding
   **代码**: `Qwen2_5_VisionPatchMerger.forward`（§4 与 §5 在同一模块内完成）
↓
6. LLM 位置编码 v vs t 是否复用率更高 更激进的复用策略

   把这些 image tokens 和文字 tokens 一起推理
   **代码**: `Qwen2_5_VLForCausalLM.forward` → `general_mm_embed_routine`；图像编码入口为 `get_image_feature`

1. 先把图切成很多小格子
2. 每个小格子提取颜色/边缘/纹理
3. 小格子之间互相交换信息
4. 把很多小格子合成更大的语义区域
5. 把这些视觉区域交给语言模型理解

### 9.1 理论天花板

粗略计算每层 per-token 计算量：

```text
MLP: gate_up + down
1280 x 6840 + 3420 x 1280 ≈ 13.1M MAC

attention projection: qkv + out
1280 x 3840 + 1280 x 1280 ≈ 6.5M MAC
```

MLP 大约占每层 per-token 计算的：

```text
约 67%
```

token_sparse 最多只能省：

```text
prefull 7 层里的 reused token 的 MLP
```

如果 100% token 都能复用，理论上限大约是：

```text
100% x 67% x 7/32 ≈ 14.6%
```

但实际复用率只有：

```text
3.5%
```

所以实际理论收益大约只有：

```text
3.5% x 67% x 7/32 ≈ 0.5%
```

也就是说：

```text
最多只省约 0.5% 的 ViT 计算。
```

但为了实现 sparse path，需要额外做：

```text
clone
mask
scatter / gather
搬 donor hidden states
每层进入 sparse 分支
```

这些开销远大于 0.5% 的理论省算，所以最终变慢是合理的。

### 9.2 即使继续做 attention sparse，也救不回来

如果第二阶段把 attention 也做稀疏，理论上可以从“只省 FFN”变成“省整个 prefull block”。

但复用率仍然只有 3.5%。

即使假设 attention 和 MLP 都能跳过，理论上也只是：

```text
3.5% x 7/32 ≈ 0.7%
```

这仍然太小，很容易被 sparse 调度开销吃掉。

所以当前瓶颈不是“attention 没省”，而是：

```text
能在早期层安全复用的 token 太少。
```

## 10. 最核心的矛盾

现在实验已经把问题讲清楚了：

| 层级 | 相似度 | 能否节省 ViT 计算 | 结果 |
|---|---:|---|---|
| merged image token | 高，0.90 阈值下约 76% 可复用 | 不能，ViT 已经算完 | 能证明冗余，但不能直接加速 |
| patch-hidden token | 低，0.80 阈值下只有 3.5% 可复用 | 能，发生在早期 | 复用太少，省不动 |
| window | 太粗，最高约 0.86 | 能 | 复用率太低 |

一句话：

> 相似度高的层，已经太晚，省不了计算；  
> 能省计算的早期层，相似度又不够高。

这就是为什么前面的 idea 做了很久都没能变成 e2e 加速。

不是因为“token similarity 是错的”，而是因为：

```text
token similarity 成立的位置，不是系统能直接省算的位置。
```

## 11. 当前结论

这轮实验支持以下结论：

1. 局部视觉冗余确实存在，尤其是在 merged image token 层。
2. window 级复用太粗，Chart refocus 下很难获得高复用率。
3. merged-token 复用率很高，但它发生在完整 ViT 之后，不能直接带来 e2e 加速。
4. `token_sparse` 是真正少算的实现，但早期 patch-hidden token 复用率只有 3.5%。
5. 在 Qwen2.5-VL 当前结构下，3.5% early-token 复用率对应的理论收益大约只有 0.5%。
6. sparse path 的工程开销大于这点收益，所以实际变慢。
7. 继续简单降阈值会增加语义风险，不一定能换来有效加速。
8. 继续做 attention sparse 在当前复用率下也大概率不值得。

总的结论：

```text
最终视觉 token 层存在明显局部冗余；
但当前在线复用方法还不能把这种冗余转化成有效 e2e 加速。
```

## 12. 对项目的意义

这个结果不是完全失败，而是一个有价值的负结果。

它说明：

```text
不能简单地从“merged image token 很相似”
推出“ViT 可以直接 token-level cache 加速”。
```

更准确的论文表述应该是：

```text
Refocus 后的最终视觉表示存在大量局部冗余。
但要把这种冗余用于系统加速，需要一个早期可用的 predictor，
或者需要模型/工具暴露出更明确的可复用区域。
```

不能声称：

```text
我们已经实现了有效的 token-level e2e 加速。
```

可以声称：

```text
1. merged image tokens 的局部冗余很明显；
2. window 级复用在 Chart refocus 上太粗；
3. 真正少算的 token_sparse 可以实现，但当前 early-token 相似度太低；
4. 未能加速的原因可以被量化解释，不是单纯日志或实现问题。
```

## 13. 后续建议

不建议继续盲目做：

```text
继续降低 token_sparse threshold
继续硬做 attention sparse
继续把整图复用包装成 partial token reuse
```

更合理的方向是：

1. 找 early predictor：在不完整跑 ViT 的情况下，预测哪些最终 merged tokens 会稳定。
2. 利用工具信息：refocus/crop/highlight 的几何区域本身可能告诉我们哪些区域没变。
3. 换更适合的 workload：如果图像变换保留大块不变区域，局部复用更可能有效。
4. 保留 token_sparse 作为诊断路径，不把它当最终加速方案。
5. 加 cost-aware gate：

```text
如果预测复用率太低，直接关闭 partial reuse，
避免为了很少的复用付出额外调度成本。
```

最终可以把问题拆成两层：

```text
局部视觉计算是否存在冗余？
当前系统能否低成本利用这种冗余？
```

这次实验的答案是：

```text
存在冗余，但当前方法还不能低成本利用它实现 e2e 加速。
```


算法逻辑：
┌─────────────────────────────────────────────────────────────┐
│  策略层：grpo_similarity_cache.py                            │
│  「该不该复用？找谁当 donor？整图 skip 还是走 partial？」      │
└──────────────────────────┬──────────────────────────────────┘
                           │ 调用
┌──────────────────────────▼──────────────────────────────────┐
│  执行层：qwen2_5_vl.py → forward_with_partial_window_reuse   │
│  「怎么少算 ViT？window/token/merged 各走哪条计算路径？」    │
└─────────────────────────────────────────────────────────────┘

vtool_agent_loop (branch 0..3, 同一 agent_uid)
    │
    ├─ generate(request_id=branchX, agent_uid=group_id, agent_turn=1)
    │
    ▼
async_sglang_server.generate()
    ├─ sglang_rid = f"{request_id}_t1"
    ├─ register_request_meta(sglang_rid, agent_uid, agent_turn=1)  → _REQUEST_META
    └─ SGLang forward → get_image_feature()
           │
           ├─ item.hash → rid → 查 _REQUEST_META 得 agent_uid
           │
           └─ encode_with_grpo_similarity_cache()
                  │
                  ├─ split_images_by_grid()  → slot0 原图 + slot1 refocus（分别处理）
                  │
                  └─ 每个 slot 走下面决策树 ↓

查 _GROUP_CACHE[key] → 有 donor 吗？
│
├─ 无 donor（本 group 第一个编码这张图的 branch）
│     └─ encode_partial(capture_cache=True)  或  encode_single()
│        → 写入 _GROUP_CACHE（pixel + embedding + partial_cache）
│
├─ 有 donor + exact（patch 逐字节相同率 ≥ 99.9%）
│     └─ skip_vit = True → 直接用 donor.embedding（整图复用）
│
├─ 有 donor + 不 exact + slot0
│     └─ 不复用（原图必须 exact）
│
├─ 有 donor + 不 exact + slot1 + whole_slot 模式
│     └─ raw patch cosine 过门槛 → skip_vit，整图用 donor.embedding
│
└─ 有 donor + 不 exact + slot1 + partial 模式
      └─ partial_attempted = True
         └─ encode_partial(donor 的 partial_cache)
            → 走 forward_with_partial_window_reuse（window/token/merged/...）
