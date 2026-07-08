# Geo3K Rollout 实验复现指南

给复现 **baseline vs rollout E+P 优化（kvdev + warm barrier）** 用的。  
代码仓库：[GraftRL](https://github.com/Rabenda/GraftRL)（monorepo：`verl/` + `sglang/`）。

**最小有效规模**：`TRAIN_BATCH_SIZE=64` × `ROLLOUT_N=4`（64×4），4 GPU，**不要缩小做 smoke**。

---

## 1. 硬件与软件

| 项 | 要求 |
|----|------|
| GPU | 4× 80GB（H100/A100 均可；作者实测 H100） |
| CUDA | 12.x（作者环境：12.8 + `torch 2.9.1+cu128`） |
| Python | **3.12** |
| 磁盘 | HuggingFace 模型缓存 + parquet 数据；Ray tmp 建议 `/dev/shm/xxx`（路径要短） |

---

## 2. 拉代码

```bash
git clone https://github.com/Rabenda/GraftRL.git
cd GraftRL
git log -1 --oneline   # 确认含 242b1d0（barrier）和 307076e（SGLang opts）
```

---

## 3. 建环境（参考 verl_vision）

### 3.1 Conda

```bash
conda create -n graftrl python=3.12 -y
conda activate graftrl
```

### 3.2 PyTorch（按你的 CUDA 版本，勿盲目抄 cu128）

作者环境：

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

其他 CUDA 版本见 [PyTorch 官网](https://pytorch.org/get-started/locally/)，**版本尽量接近 2.9.1**。

### 3.3 安装本仓库的 verl + sglang（必须 editable）

```bash
cd GraftRL
pip install -e sglang/python
pip install -e verl
```

⚠️ **不要** `pip install verl` / `pip install sglang` 从 PyPI 装——那是上游，没有 CacheBlend patch。

### 3.4 FlashAttention（常需单独装）

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

若编译失败，检查 CUDA 工具链、`nvcc` 是否在 PATH。

### 3.5 其余依赖（与 verl_vision 对齐）

```bash
pip install -r requirements-reproduce.txt
```

---

## 4. 模型与数据

```bash
export HF_HOME="${HF_HOME:-/data/huggingface_cache}"   # 改成你的缓存目录
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"
```

- **模型**：`Qwen/Qwen2.5-VL-7B-Instruct`（脚本首次运行会自动从 HF 拉）
- **数据**：Geo3K refocus exact — 跑 demo 时若缺 parquet 会自动执行 `prepare_refocus_data.sh`

---

## 5. 运行实验（两条命令）

```bash
cd GraftRL/verl

export CUDA_VISIBLE_DEVICES=0,1,2,3    # 改成你的 4 张卡
export RAY_TMPDIR=/dev/shm/graftrl     # 短路径，避免 Ray socket 报错
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
mkdir -p "${RAY_TMPDIR}"
```

**ARM A — baseline（CacheBlend 全关）**

```bash
bash examples/profile/workloads/geo3k/run_geo3k_rollout_demo.sh baseline
```

**ARM B — rollout E+P 优化（kvdev + warm barrier + bounded wait）**

```bash
bash examples/profile/workloads/geo3k/run_geo3k_rollout_demo.sh optimized
```

每臂跑 **2 个 training step**；**分析 step 2**（step 1 含冷启动）。

---

## 6. 看什么结果

### 6.1 整步 timing（console log）

```bash
grep 'step:2' profile_logs_geo3k_rollout_ab/geo3k_refocus_exact_demo_off_*.log | grep timing_s
grep 'step:2' profile_logs_geo3k_rollout_ab/geo3k_refocus_exact_demo_kvdev_*.log | grep timing_s
```

关注：

| 字段 | 含义 |
|------|------|
| `timing_s/gen` | rollout 墙钟（优化目标） |
| `timing_s/old_log_prob` | training 重算 logprob |
| `timing_s/update_actor` | actor 更新 |

两臂 training 配置相同（标准 PPO recompute），**只有 rollout 栈不同**。

### 6.2 单请求 turn1 E+P（CSV）

```bash
python3 examples/profile/shared/analysis/analyze_profiling_logs.py \
  --log-dir profile_logs_geo3k_refocus_exact_demo_kvdev_slotslast_fa0_cp0 \
  --suffix geo3k_refocus_exact_demo_kvdev_slotslast_fa0_cp0_bs64_n4 --report
```

### 6.3 Barrier 是否生效

```bash
# optimized 臂：donor_ready 应接近 100%（demo 默认 MAX_WAIT_S=10）
head -5 profile_logs_geo3k_refocus_exact_demo_kvdev_slotslast_fa0_cp0/cacheblend_barrier_log_*.csv
```

若 `donor_ready=0%` 占多数 → 提高等待：

```bash
SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_MAX_WAIT_S=10 \
  bash examples/profile/workloads/geo3k/run_geo3k_rollout_demo.sh optimized
```

### 6.4 Graft 覆盖率（不要数错）

- ❌ 不要只看 `model_forward_log` 里 `cacheblend_used=1` 的行数  
- ✅ 看 `cacheblend_fallback_reason` 里的 **`batch:recipient_kv_blended=192`**（64 组 × 3 recipient）

---

## 7. 作者环境参考数字（2367 四卡，step2）

| 臂 | turn1 e2e (mean) | donor_ready | 备注 |
|----|------------------|-------------|------|
| off | ~3383 ms | — | baseline |
| kvdev + wait10 | ~2899 ms (**−14%**) | **100%** | 推荐 optimized 配置 |
| kvdev + wait0 | ~6163 ms (+85%) | **0%** | barrier 失效，勿用 |

整步 wall-clock 受 cluster 负载/Ray/磁盘影响较大；**turn1 e2e + donor_ready%** 是更稳的 rollout 指标。

更完整归档：[`verl/docs/geo3k_profiling_archive_0702.md`](../verl/docs/geo3k_profiling_archive_0702.md)

---

## 8. 当前优化栈说明（optimized 臂）

| 层 | 配置 |
|----|------|
| KV graft | `CACHEBLEND_SELECTOR=kvdev` |
| Image slot | `CACHEBLEND_IMAGE_SLOTS=-1`（refocus 最后一张图） |
| Warm barrier | turn0 prefix + turn1 cacheblend |
| Bounded wait | `MAX_WAIT_S=10` |
| fast_apply / compact_prefill | **关**（fa0/cp0；Geo3K decode 短，cp1 曾证负） |
| Training | 标准 PPO recompute，**无 bypass** |

---

## 9. 常见问题

**Q: Ray 报 socket / path too long**  
A: `export RAY_TMPDIR=/dev/shm/短名字` 并 `mkdir -p`。

**Q: OOM**  
A: 试 `export GPU_MEMORY_UTILIZATION=0.25`；或 `ACTOR_MAX_TOKEN_LEN_PER_GPU=4096`（两臂要一致）。

**Q: 和作者数字差很多**  
A: 看是否同 GPU 型号、是否空卡、step 是否看 step2、optimized 是否 donor_ready≈100%。

**Q: 项目背景 / 算法**  
A: [`docs/GraftRL_项目全历程.md`](GraftRL_项目全历程.md)

---

*文档版本：2026-07-07，与 main 上 demo 脚本一致。*
