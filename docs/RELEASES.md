# GraftRL 版本标签

**完整中文说明（做了什么、算法步骤、代码位置）请读：**

→ **[GraftRL_项目全历程.md](./GraftRL_项目全历程.md)**

---

## 标签列表

| Tag | 内容 |
|-----|------|
| `v0.0-baseline` | 上游 verl `802256a7` + sglang `0189f41`，无 GraftRL 代码 |
| `v0.1-profiling` | EPD 三路日志 + Geo3K profile |
| `v0.1.1-vision-cold-warm` | ViT 冷/热路径分析脚本 |
| `v0.2-motivation` | Refocus agent、相似度、text-only 对照 |
| `v0.2.1-exploratory-workloads` | Sokoban、DeepEyes、archive |
| `v0.3-vit-reuse` | ViT 组内缓存 + partial 实验（负结果） |
| `v0.4-graft-core` | `vlm_cacheblend` 核心 + donor 捕获 |
| `v0.5-graft-e2e` | attention 快路径 + AB 质量验证 |
| `v0.6-stress` | stress 数据集 + chunked prefill 修复 |

```bash
git checkout v0.4-graft-core   # 查看某一阶段代码
```
