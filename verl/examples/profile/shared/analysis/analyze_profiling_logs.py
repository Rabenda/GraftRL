#!/usr/bin/env python3
"""
Analyze verl_vision + sglang_vision_profile CSV logs.

Features:
  - Request table: prompt, image_ratio, vision_ms, e2e, vision/e2e, cached, pid
  - EPD raw + adjusted (cache-miss, first-encode, multi-image / multi-req split)
  - Cold/warm by SGLang pid (first vision_encoder row per pid = cold ViT)
  - Cold/warm by pass_id (pass_id==1 single-request = structural cold)
  - Per global_step breakdown (multi-step repeat experiments)
  - Model forward EXTEND vs DECODE stats
  - Full pipeline module breakdown CSV (e2e vs E/P/D vs wall-clock GPU vs overhead)

EPD definitions (from existing CSV logs, no double-count):
  E = vision_encoder_time_ms          (vision_encoder_log)
  P = sum(EXTEND forward_time_ms) - E  (model_forward_log; EXTEND includes ViT+LLM prefill)
  D = sum(DECODE forward_time_ms)      (model_forward_log)

EPD adjusted (GRPO / prefix-mm-cache aware):
  - Drop vision rows with cached_image_features==1 (precomputed embedding, no ViT)
  - Per request_id: count only the first ViT encode row (chronological cache-miss)
  - Split one encode row across request_ids (batched) and image_count (multi-image)
  - Requests with no vision row -> E=0 (likely mm/prefix cache; listed in coverage stats)

Usage:
  # Auto paths from log dir + suffix
  python examples/profile/analyze_profiling_logs.py \\
    --log-dir profile_logs_coldwarm \\
    --suffix coldwarm_repeat_3step

  # Compare stress 1/2/4img in one shot
  python examples/profile/analyze_profiling_logs.py \\
    --log-dir profile_logs_stress \\
    --compare-suffixes stress_1img,stress_2img,stress_4img

  # Explicit paths
  python examples/profile/analyze_profiling_logs.py \\
    --generate-log profile_logs_geo3k/verl_sglang_generate_log_qwen25vl_geo3k_2gpu_ab_req8.csv \\
    --vision-log profile_logs_geo3k/vision_encoder_log_qwen25vl_geo3k_2gpu_ab_req8.csv \\
    --model-forward-log profile_logs_geo3k/model_forward_log_qwen25vl_geo3k_2gpu_ab_req8.csv

  # Export per-request module breakdown + summary CSV
  python examples/profile/analyze_profiling_logs.py \\
    --log-dir profile_logs_geo3k \\
    --suffix qwen25vl_geo3k_2gpu_ab_req8 \\
    --export-breakdown-csv profile_logs_geo3k/e2e_module_breakdown_qwen25vl_geo3k_2gpu_ab_req8.csv

  # One-shot: CSVs + Markdown report (recommended)
  python examples/profile/analyze_profiling_logs.py \\
    --log-dir profile_logs_geo3k_full \\
    --suffix geo3k_full_bs64_n4 \\
    --report
"""

from __future__ import annotations

from datetime import datetime, timezone

import argparse
import csv
import glob
import os
import statistics as st
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

# Markdown blocks for §5 in profiling_report_*.md (plain-language legend for A/B/C/D).
PAPER_ALIGNED_VIEW_INTRO: tuple[str, ...] = (
    "## 5. 双视角指标（对齐 EPD 论文的两种看法）",
    "",
    "同一次实验会同时看 **GPU 算力花在哪** 和 **用户墙钟卡在哪**；下面 A/B/C/D 是报告里的四块，"
    "**不是四组独立实验**。",
    "",
    "| 块 | 白话 | 分母 | 回答的问题 |",
    "| --- | --- | --- | --- |",
    "| **A — GPU 阶段算力** | 把整次 run 里所有请求的 ViT encode、LLM prefill、decode "
    "的 **GPU 前向时间加总**，看 E/P/D 各占多少 | 全部 **E+P+D** | "
    "「真正在 GPU 上算的时候，主要算编图+prefill 还是 decode？」"
    "（E+P 超过 50% 时称 **前端/ prefill 侧重**） |",
    "| **B — 单请求端到端墙钟** | 每个请求从进入到返回的 **总等待时间 e2e**，"
    "拆成排队、等同 batch 的 prefill、自己的 extend/decode 等 | 该请求的 **e2e** | "
    "「用户等那么久，有多少是在 **等别人/等同 replica**，有多少在自己算？」 |",
    "| **C — ViT 单次耗时** | 不计入整次加总占比，只看 **每一次** vision encoder forward 多快 | "
    "单次 pass | 「单张图 encode 本身贵不贵？」（通常远小于 B 里的 batch 等待） |",
    "| **D — Decode 是否被拖住** | 按 replica 看 decode 窗口里有多少时间在空等、"
    "prefill(EXTEND) 是否插进 decode | replica 窗口 | "
    "「decode 阶段有没有被前面的 prefill 打断或饿死？」 |",
    "",
    "**A 与 B 为何常「看起来矛盾」**：A 只统计 **GPU 在干活** 的时间；B 里 "
    "`prefill_blocking_wait` 是 **墙钟等待**（同 replica 上先完成的 EXTEND 占满时间，"
    "本请求在等），**不算进 E/P/D 的 forward 加总**。因此可能出现：A 显示算力主要在 prefill，"
    "B 显示 e2e 里一大块是 **等待** 而非 ViT 单次成本。",
    "",
    "**阶段缩写**：E = ViT/vision encode；P = LLM prefill（EXTEND 减去 E）；"
    "D = autoregressive decode。",
    "",
)

PAPER_ALIGNED_GROUP_TITLES: dict[str, str] = {
    "A_stage_execution": "A — GPU 阶段算力（分母 = 整次 E+P+D 之和）",
    "B_per_request_e2e": "B — 单请求墙钟（分母 = 该请求 e2e）",
    "C_encode_raw": "C — ViT 单次 forward（每次 pass，不参与 A 的百分比分母）",
    "D_decode_stall": "D — Decode 停顿与 prefill 插队（按 replica）",
}


def load_csv(path: str) -> list[dict]:
    """Load profiling CSV, skipping duplicate header rows from append-mode logs."""
    rows: list[dict] = []
    fieldnames: list[str] | None = None
    with open(path, newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if (
                line.startswith("timestamp,")
                or line.startswith("request_id,")
                or line.startswith("server_pid,")
            ):
                fieldnames = line.split(",")
                continue
            if fieldnames is None:
                continue
            parts = next(csv.reader([line]))
            if len(parts) != len(fieldnames):
                continue
            row = dict(zip(fieldnames, parts))
            first_val = next(iter(row.values()), "")
            if first_val in fieldnames:
                continue
            rows.append(row)
    return rows


def parse_request_ids(field: str | None) -> list[str]:
    if not field:
        return []
    return [x.strip() for x in str(field).split("|") if x.strip()]


def find_log(log_dir: str, prefix: str, suffix: str) -> Optional[str]:
    patterns = [
        os.path.join(log_dir, f"{prefix}_{suffix}.csv"),
        os.path.join(log_dir, f"{prefix}*{suffix}*.csv"),
    ]
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]
    return None


def stats(vals: list[float]) -> str:
    if not vals:
        return "n=0"
    return (
        f"n={len(vals):3d} mean_ms={st.mean(vals):7.2f} "
        f"min_ms={min(vals):7.2f} max_ms={max(vals):7.2f}"
    )


def print_generate_only_summary(label: str, gen: list[dict]) -> None:
    print(f"\n{'=' * 80}\n{label}\nGENERATE LOG SUMMARY")
    print(f"  requests: {len(gen)}")
    if not gen:
        return

    def int_field(name: str) -> list[int]:
        return [int(float(r.get(name) or 0)) for r in gen]

    image_counts = int_field("image_count")
    prompt_tokens = int_field("prompt_tokens")
    image_tokens = int_field("image_prompt_tokens")
    output_tokens = int_field("output_tokens")
    e2e_ms = [float(r.get("generate_e2e_ms") or 0) for r in gen]

    print(f"  image_count values: {sorted(set(image_counts))}")
    print(f"  prompt_tokens: min={min(prompt_tokens)} max={max(prompt_tokens)}")
    print(f"  image_prompt_tokens: min={min(image_tokens)} max={max(image_tokens)}")
    print(f"  output_tokens: min={min(output_tokens)} max={max(output_tokens)}")
    print(f"  generate_e2e_ms: {stats(e2e_ms)}")

    by_image_count: dict[int, list[dict]] = defaultdict(list)
    for r, c in zip(gen, image_counts):
        by_image_count[c].append(r)
    print("  by image_count:")
    for c in sorted(by_image_count):
        rows = by_image_count[c]
        row_e2e = [float(r.get("generate_e2e_ms") or 0) for r in rows]
        row_img = [int(float(r.get("image_prompt_tokens") or 0)) for r in rows]
        print(
            f"    image_count={c}: n={len(rows)} "
            f"image_tokens_minmax={min(row_img)}..{max(row_img)} "
            f"e2e={stats(row_e2e)}"
        )


@dataclass
class VisionRow:
    timestamp: float
    pid: str
    global_step: int
    pass_id: int
    mode: str
    request_ids: list[str]
    image_count: int
    image_tokens: int
    prefill_tokens: int
    image_token_ratio: float
    vision_ms: float
    cached: int
    structural_cold: bool  # pass_id==1 and single request


def parse_vision_rows(rows: list[dict]) -> list[VisionRow]:
    out: list[VisionRow] = []
    for r in rows:
        ids = parse_request_ids(r.get("request_ids"))
        out.append(
            VisionRow(
                timestamp=float(r.get("timestamp") or 0),
                pid=str(r.get("pid") or ""),
                global_step=int(float(r.get("global_step") or -1)),
                pass_id=int(float(r.get("pass_id") or -1)),
                mode=str(r.get("mode") or "EXTEND"),
                request_ids=ids,
                image_count=max(1, int(float(r.get("image_count") or 1))),
                image_tokens=int(float(r.get("image_tokens") or 0)),
                prefill_tokens=int(float(r.get("prefill_tokens") or 0)),
                image_token_ratio=float(r.get("image_token_ratio") or 0),
                vision_ms=float(r.get("vision_encoder_time_ms") or 0),
                cached=int(float(r.get("cached_image_features") or 0)),
                structural_cold=(r.get("pass_id") == "1" and len(ids) == 1),
            )
        )
    return out


def pid_cold_warm_labels(vision: list[VisionRow]) -> dict[tuple[str, str], str]:
    """First vision row per pid (by timestamp) -> cold; later -> warm."""
    by_pid: dict[str, list[VisionRow]] = defaultdict(list)
    for v in vision:
        by_pid[v.pid].append(v)
    labels: dict[tuple[str, str], str] = {}
    for pid, rows in by_pid.items():
        rows_sorted = sorted(rows, key=lambda x: x.timestamp)
        for i, row in enumerate(rows_sorted):
            for rid in row.request_ids:
                labels[(pid, rid)] = "cold" if i == 0 else "warm"
    return labels


def match_vision_to_request(
    request_id: str, vision: list[VisionRow], pid_labels: dict[tuple[str, str], str]
) -> Optional[VisionRow]:
    """Pick the vision row that best matches this request (single-id row preferred)."""
    candidates = [v for v in vision if request_id in v.request_ids]
    if not candidates:
        return None
    singles = [v for v in candidates if len(v.request_ids) == 1]
    pool = singles if singles else candidates
    return min(pool, key=lambda v: v.timestamp)


@dataclass
class RequestRecord:
    request_id: str
    global_step: int
    prompt: int
    image_prompt_tokens: int
    image_ratio: float
    vision_ms: float
    e2e_ms: float
    vision_over_e2e_pct: float
    cached: int
    pid: str
    pass_id: int
    pid_path: str  # cold/warm by first row per pid
    structural_path: str  # cold if pass_id==1 single else warm/batched


def build_request_records(gen: list[dict], vision: list[VisionRow]) -> list[RequestRecord]:
    vis_parsed = parse_vision_rows(vision)
    pid_labels = pid_cold_warm_labels(vis_parsed)
    records: list[RequestRecord] = []

    for g in gen:
        rid = g["request_id"]
        e2e = float(g.get("generate_e2e_ms") or 0)
        vrow = match_vision_to_request(rid, vis_parsed, pid_labels)
        if vrow is None:
            continue
        pid_path = pid_labels.get((vrow.pid, rid), "unknown")
        structural = "cold" if vrow.structural_cold else "warm"
        vision_ms = vrow.vision_ms
        pct = (vision_ms / e2e * 100) if e2e > 0 else 0.0
        records.append(
            RequestRecord(
                request_id=rid,
                global_step=int(float(g.get("global_step") or 0)),
                prompt=int(float(g.get("prompt_tokens") or 0)),
                image_prompt_tokens=int(float(g.get("image_prompt_tokens") or 0)),
                image_ratio=float(g.get("image_prompt_ratio") or 0),
                vision_ms=vision_ms,
                e2e_ms=e2e,
                vision_over_e2e_pct=pct,
                cached=vrow.cached,
                pid=vrow.pid,
                pass_id=vrow.pass_id,
                pid_path=pid_path,
                structural_path=structural,
            )
        )
    return records


def print_request_table(label: str, records: list[RequestRecord]) -> None:
    print(f"\n{'=' * 80}\n{label}\nREQUEST TABLE (join by request_id)")
    print(
        "request   step  prompt  img_ratio  img_tok  vision_ms  e2e_ms  "
        "vision/e2e  cached  pass  pid_path  pid"
    )
    for r in records:
        print(
            f"{r.request_id[:8]:8s}  {r.global_step:4d}  {r.prompt:6d}  "
            f"{r.image_ratio:8.2f}  {r.image_prompt_tokens:7d}  {r.vision_ms:8.1f}  "
            f"{r.e2e_ms:7.0f}  {r.vision_over_e2e_pct:9.1f}%  {r.cached:6d}  "
            f"{r.pass_id:4d}  {r.pid_path:8s}  {r.pid}"
        )


def print_summary(records: list[RequestRecord], key: str) -> None:
    print(f"\nSUMMARY by {key}")
    groups: dict[str, list[RequestRecord]] = defaultdict(list)
    for r in records:
        groups[getattr(r, key)].append(r)

    for name in sorted(groups.keys(), key=lambda x: str(x)):
        grp = groups[name]
        v = [r.vision_ms for r in grp]
        e = [r.e2e_ms for r in grp]
        p = [r.vision_over_e2e_pct for r in grp]
        print(
            f"  {name:12s}  n={len(grp):3d}  "
            f"vision_ms mean={st.mean(v):7.1f}  e2e_ms mean={st.mean(e):7.0f}  "
            f"vision/e2e mean={st.mean(p):6.2f}%"
        )

    all_v = [r.vision_ms for r in records]
    all_e = [r.e2e_ms for r in records]
    all_p = [r.vision_over_e2e_pct for r in records]
    print(
        f"  {'all':12s}  n={len(records):3d}  "
        f"vision_ms mean={st.mean(all_v):7.1f}  e2e_ms mean={st.mean(all_e):7.0f}  "
        f"vision/e2e mean={st.mean(all_p):6.2f}%"
    )


def print_per_pid_vision(vision: list[VisionRow]) -> None:
    print("\nPER-PID VISION ENCODER (sorted by time)")
    by_pid: dict[str, list[VisionRow]] = defaultdict(list)
    for v in vision:
        by_pid[v.pid].append(v)
    for pid in sorted(by_pid.keys()):
        rows = sorted(by_pid[pid], key=lambda x: x.timestamp)
        print(f"  pid={pid}")
        for i, v in enumerate(rows):
            tag = "COLD" if i == 0 else "warm"
            ids = "|".join(x[:8] for x in v.request_ids)
            print(
                f"    [{tag}] pass={v.pass_id} step={v.global_step} "
                f"vision={v.vision_ms:.1f}ms img_cnt={v.image_count} "
                f"img_tok={v.image_tokens} reqs={ids} cached={v.cached}"
            )


def print_per_step(records: list[RequestRecord]) -> None:
    steps = sorted({r.global_step for r in records})
    if len(steps) <= 1:
        return
    print("\nPER global_step (training step)")
    for step in steps:
        sub = [r for r in records if r.global_step == step]
        cold = [r for r in sub if r.pid_path == "cold"]
        warm = [r for r in sub if r.pid_path == "warm"]
        print(
            f"  step={step}: n={len(sub)}  cold={len(cold)} warm={len(warm)}  "
            f"vision_ms cold={st.mean([r.vision_ms for r in cold]):.0f} warm={st.mean([r.vision_ms for r in warm]):.0f}"
            if cold and warm
            else f"  step={step}: n={len(sub)}"
        )


@dataclass
class EpdBreakdown:
    """EPD-style stage times in milliseconds (GPU forward oriented)."""

    e_ms: float
    p_ms: float
    d_ms: float
    extend_ms: float
    decode_ms: float
    n_requests: int = 0
    n_vision_rows: int = 0
    n_extend_passes: int = 0
    n_decode_passes: int = 0
    mean_image_tokens: float = 0.0
    mean_vision_over_e2e_pct: float = 0.0
    mean_e2e_ms: float = 0.0
    # adjusted (optional; set by compute_epd_adjusted)
    e_ms_adj: float = 0.0
    p_ms_adj: float = 0.0
    d_ms_adj: float = 0.0
    n_vision_rows_cached_skip: int = 0
    n_vision_rows_miss_used: int = 0
    n_vision_rows_miss_total: int = 0
    n_requests_no_vision_row: int = 0
    n_requests_first_encode: int = 0
    n_requests_encode_skipped_dup: int = 0  # later encode rows for same request_id

    @property
    def total_ms(self) -> float:
        return self.e_ms + self.p_ms + self.d_ms

    @property
    def total_ms_adj(self) -> float:
        return self.e_ms_adj + self.p_ms_adj + self.d_ms_adj

    def pct(self, part_ms: float, total: Optional[float] = None) -> float:
        denom = total if total is not None else self.total_ms
        return (part_ms / denom * 100) if denom > 0 else 0.0

    @property
    def ep_pct(self) -> float:
        return self.pct(self.e_ms + self.p_ms)

    @property
    def ep_pct_adj(self) -> float:
        return self.pct(self.e_ms_adj + self.p_ms_adj, self.total_ms_adj)


def sum_model_forward_by_mode(mf_path: Optional[str]) -> tuple[float, float, int, int]:
    """Return (extend_ms, decode_ms, n_extend_passes, n_decode_passes)."""
    if not mf_path or not os.path.exists(mf_path):
        return 0.0, 0.0, 0, 0
    extend_ms = decode_ms = 0.0
    n_extend = n_decode = 0
    for r in load_csv(mf_path):
        mode = (r.get("mode") or "").upper()
        t = float(r.get("forward_time_ms") or 0)
        if mode == "EXTEND":
            extend_ms += t
            n_extend += 1
        elif mode == "DECODE":
            decode_ms += t
            n_decode += 1
    return extend_ms, decode_ms, n_extend, n_decode


def total_vision_encoder_ms(vision_rows: list[dict]) -> float:
    return sum(float(r.get("vision_encoder_time_ms") or 0) for r in vision_rows)


def _vision_rows_miss(vision: list[VisionRow]) -> list[VisionRow]:
    """Rows where ViT actually ran (not precomputed-embedding shortcut)."""
    return [v for v in vision if v.cached == 0 and v.vision_ms > 0]


def allocate_first_encode_e_per_request(
    vision: list[VisionRow],
) -> tuple[dict[str, float], dict[str, int]]:
    """
  Attribute encode time to request_id (first cache-miss encode only).

  One vision row with vision_ms=T, N=request_ids, M=image_count:
    each new request_id in the row receives T/N (M images encoded in one ViT call).
  """
    per_req: dict[str, float] = {}
    meta: dict[str, int] = {
        "rows_cached_skip": 0,
        "rows_miss": 0,
        "encode_skipped_dup": 0,
    }
    for v in vision:
        if v.cached != 0:
            meta["rows_cached_skip"] += 1
            continue
        if v.vision_ms <= 0:
            continue
        meta["rows_miss"] += 1
        ids = v.request_ids if v.request_ids else [f"__row_{v.timestamp:.3f}__"]
        share_per_req = v.vision_ms / len(ids)
        for rid in ids:
            if rid in per_req:
                meta["encode_skipped_dup"] += 1
                continue
            per_req[rid] = share_per_req
    return per_req, meta


def apply_epd_adjusted(
    epd: EpdBreakdown,
    gen: list[dict],
    vision_parsed: list[VisionRow],
) -> EpdBreakdown:
    per_req_e, meta = allocate_first_encode_e_per_request(vision_parsed)
    epd.e_ms_adj = sum(per_req_e.values())
    # When nothing was image-cached, every encode is real GPU compute and there is
    # nothing to "adjust down" — the dedup is only meaningful for cache-hit / repeat
    # scenarios. Fall back to raw E so a no-cache run doesn't get a bogusly small
    # E_adj (this also neutralizes legacy logs whose request_ids were batch-level).
    if meta["rows_cached_skip"] == 0:
        epd.e_ms_adj = epd.e_ms
    epd.p_ms_adj = max(0.0, epd.extend_ms - epd.e_ms_adj)
    epd.d_ms_adj = epd.decode_ms
    epd.n_vision_rows_cached_skip = meta["rows_cached_skip"]
    epd.n_vision_rows_miss_total = meta["rows_miss"]
    epd.n_vision_rows_miss_used = len(
        [v for v in _vision_rows_miss(vision_parsed) if v.request_ids]
    )
    epd.n_requests_first_encode = len(per_req_e)
    gen_ids = [g["request_id"] for g in gen]
    epd.n_requests_no_vision_row = sum(1 for rid in gen_ids if rid not in per_req_e)
    epd.n_requests_encode_skipped_dup = meta["encode_skipped_dup"]
    return epd


def sum_passes_by_mode(
    forward_passes: list["ForwardPass"],
) -> tuple[float, float, int, int]:
    extend_ms = sum(fp.forward_ms for fp in forward_passes if fp.mode == "EXTEND")
    decode_ms = sum(fp.forward_ms for fp in forward_passes if fp.mode == "DECODE")
    n_extend = sum(1 for fp in forward_passes if fp.mode == "EXTEND")
    n_decode = sum(1 for fp in forward_passes if fp.mode == "DECODE")
    return extend_ms, decode_ms, n_extend, n_decode


def compute_epd_breakdown(
    gen: list[dict],
    vision: list[dict],
    mf_path: Optional[str],
    records: Optional[list[RequestRecord]] = None,
    forward_passes: Optional[list["ForwardPass"]] = None,
) -> EpdBreakdown:
    # Prefer already-parsed (and possibly cold-start-filtered) forward passes;
    # otherwise re-read the raw model_forward_log.
    if forward_passes is not None:
        extend_ms, decode_ms, n_extend, n_decode = sum_passes_by_mode(forward_passes)
    else:
        extend_ms, decode_ms, n_extend, n_decode = sum_model_forward_by_mode(mf_path)
    e_ms = total_vision_encoder_ms(vision)
    # EXTEND forward includes ViT; subtract E to get LLM prefill (P).
    p_ms = max(0.0, extend_ms - e_ms)
    d_ms = decode_ms

    recs = records or build_request_records(gen, vision)
    img_toks = [float(g.get("image_prompt_tokens") or 0) for g in gen]
    mean_img = st.mean(img_toks) if img_toks else 0.0
    mean_e2e = st.mean([r.e2e_ms for r in recs]) if recs else 0.0
    mean_vis_e2e = st.mean([r.vision_over_e2e_pct for r in recs]) if recs else 0.0

    epd = EpdBreakdown(
        e_ms=e_ms,
        p_ms=p_ms,
        d_ms=d_ms,
        extend_ms=extend_ms,
        decode_ms=decode_ms,
        n_requests=len(gen),
        n_vision_rows=len(vision),
        n_extend_passes=n_extend,
        n_decode_passes=n_decode,
        mean_image_tokens=mean_img,
        mean_vision_over_e2e_pct=mean_vis_e2e,
        mean_e2e_ms=mean_e2e,
    )
    return apply_epd_adjusted(epd, gen, parse_vision_rows(vision))


def print_epd_table(label: str, epd: EpdBreakdown) -> None:
    print(f"\n{'=' * 80}\n{label}\nEPD STAGE BREAKDOWN (E / P / D)")
    print(
        "  E = vision_encoder_time_ms (vision_encoder_log)\n"
        "  P = sum(EXTEND forward_time_ms) - E  (model_forward_log)\n"
        "  D = sum(DECODE forward_time_ms)      (model_forward_log)"
    )
    print(
        f"\n  {'stage':6s}  {'ms':>10s}  {'% of E+P+D':>12s}  {'note':s}"
    )
    print(f"  {'E':6s}  {epd.e_ms:10.1f}  {epd.pct(epd.e_ms):11.1f}%  ViT / image encode")
    print(f"  {'P':6s}  {epd.p_ms:10.1f}  {epd.pct(epd.p_ms):11.1f}%  LLM prefill (EXTEND - E)")
    print(f"  {'D':6s}  {epd.d_ms:10.1f}  {epd.pct(epd.d_ms):11.1f}%  autoregressive decode")
    print(f"  {'E+P':6s}  {epd.e_ms + epd.p_ms:10.1f}  {epd.ep_pct:11.1f}%  front-end bound if >50%")
    print(f"  {'total':6s}  {epd.total_ms:10.1f}  {'100.0':>11s}%  E+P+D (GPU forward stages)")
    print(
        f"\n  extend_passes={epd.n_extend_passes}  decode_passes={epd.n_decode_passes}  "
        f"vision_rows={epd.n_vision_rows}  requests={epd.n_requests}"
    )
    print(
        f"  raw: EXTEND={epd.extend_ms:.1f}ms  DECODE={epd.decode_ms:.1f}ms  "
        f"vision_sum={epd.e_ms:.1f}ms"
    )
    print(
        f"  request-level (wall clock): mean e2e={epd.mean_e2e_ms:.0f}ms  "
        f"mean vision/e2e={epd.mean_vision_over_e2e_pct:.1f}%  "
        f"mean image_tokens={epd.mean_image_tokens:.0f}"
    )
    flag = "YES — E+P > 50% (EPD front-end regime)" if epd.ep_pct > 50 else "no — E+P <= 50%"
    print(f"  EPD threshold: {flag}")


def print_epd_adjusted_table(label: str, epd: EpdBreakdown) -> None:
    print(f"\n{'=' * 80}\n{label}\nEPD ADJUSTED (cache-miss / first-encode per request)")
    print(
        "  Rules:\n"
        "    - Skip vision rows with cached_image_features=1 (no ViT)\n"
        "    - Per request_id: only first encode row (later rows = repeat extend / warm)\n"
        "    - One row with N request_ids: split vision_ms evenly across N\n"
        "    - Generate requests with no vision row -> E=0 (mm/prefix cache, no log line)\n"
        "    - P_adj = sum(EXTEND) - E_adj ; D_adj = sum(DECODE)"
    )
    print(
        f"\n  {'stage':6s}  {'raw_ms':>10s}  {'adj_ms':>10s}  {'adj_%':>8s}"
    )
    print(
        f"  {'E':6s}  {epd.e_ms:10.1f}  {epd.e_ms_adj:10.1f}  "
        f"{epd.pct(epd.e_ms_adj, epd.total_ms_adj):7.1f}%"
    )
    print(
        f"  {'P':6s}  {epd.p_ms:10.1f}  {epd.p_ms_adj:10.1f}  "
        f"{epd.pct(epd.p_ms_adj, epd.total_ms_adj):7.1f}%"
    )
    print(
        f"  {'D':6s}  {epd.d_ms:10.1f}  {epd.d_ms_adj:10.1f}  "
        f"{epd.pct(epd.d_ms_adj, epd.total_ms_adj):7.1f}%"
    )
    print(
        f"  {'E+P':6s}  {epd.e_ms + epd.p_ms:10.1f}  "
        f"{epd.e_ms_adj + epd.p_ms_adj:10.1f}  {epd.ep_pct_adj:7.1f}%"
    )
    print(
        f"\n  coverage: gen_requests={epd.n_requests}  "
        f"first_encode={epd.n_requests_first_encode}  "
        f"no_vision_row={epd.n_requests_no_vision_row}  "
        f"dup_encode_skipped={epd.n_requests_encode_skipped_dup}"
    )
    print(
        f"  vision_rows: total={epd.n_vision_rows}  "
        f"miss_rows={epd.n_vision_rows_miss_total}  "
        f"cached_skip={epd.n_vision_rows_cached_skip}"
    )
    flag = (
        "YES — E+P(adj) > 50%"
        if epd.ep_pct_adj > 50
        else "no — E+P(adj) <= 50%"
    )
    print(f"  EPD threshold (adjusted): {flag}")


def print_vision_row_allocation(vision_parsed: list[VisionRow], gen: list[dict]) -> None:
    """Per vision row: how E is split (for debugging batch / multi-image)."""
    per_req, _ = allocate_first_encode_e_per_request(vision_parsed)
    gen_id_set = {g["request_id"] for g in gen}
    print("\nVISION ROW ALLOCATION (first-encode attribution)")
    print(
        "  pass  reqs(n)  img_cnt  vision_ms  cached  "
        "share/req  attributed_to"
    )
    seen_req: set[str] = set()
    for v in sorted(vision_parsed, key=lambda x: x.timestamp):
        ids = v.request_ids if v.request_ids else ["?"]
        n = len(ids)
        share = v.vision_ms / n if n else v.vision_ms
        tags = []
        for rid in ids:
            if v.cached:
                tags.append(f"{rid[:8]}:cached")
            elif rid in seen_req:
                tags.append(f"{rid[:8]}:dup_skip")
            elif rid in per_req:
                tags.append(f"{rid[:8]}:{per_req[rid]:.0f}ms")
                seen_req.add(rid)
            else:
                tags.append(f"{rid[:8]}:?")
        off_log = [rid[:8] for rid in ids if rid not in gen_id_set]
        extra = f"  (not in generate log: {','.join(off_log)})" if off_log else ""
        print(
            f"  {v.pass_id:4d}  {n:5d}  {v.image_count:7d}  {v.vision_ms:9.1f}  "
            f"{v.cached:6d}  {share:9.1f}  {','.join(tags)}{extra}"
        )


def print_epd_by_pid_path(records: list[RequestRecord], mf_path: Optional[str]) -> None:
    """Per cold/warm: E from matched requests; P/D allocated by request count."""
    extend_ms, decode_ms, _, _ = sum_model_forward_by_mode(mf_path)
    if not records:
        return
    n = len(records)
    decode_per = decode_ms / n
    groups: dict[str, list[RequestRecord]] = defaultdict(list)
    for r in records:
        groups[r.pid_path].append(r)
    extend_per = extend_ms / n

    print("\nEPD by pid_path (E exact; P/D allocated evenly across requests)")
    print(f"  {'path':8s}  {'n':>3s}  {'E%':>6s}  {'P%':>6s}  {'D%':>6s}  {'E+P%':>7s}  {'mean_E_ms':>10s}")
    for path in sorted(groups.keys()):
        grp = groups[path]
        e_sum = sum(r.vision_ms for r in grp)
        p_sum = sum(max(0.0, extend_per - r.vision_ms) for r in grp)
        d_sum = decode_per * len(grp)
        total = e_sum + p_sum + d_sum
        if total <= 0:
            continue
        print(
            f"  {path:8s}  {len(grp):3d}  "
            f"{e_sum / total * 100:5.1f}%  {p_sum / total * 100:5.1f}%  "
            f"{d_sum / total * 100:5.1f}%  {(e_sum + p_sum) / total * 100:6.1f}%  "
            f"{e_sum / len(grp):10.1f}"
        )


def print_epd_compare_table(rows: list[tuple[str, EpdBreakdown]]) -> None:
    print(f"\n{'=' * 80}\nEPD COMPARE TABLE (raw + adjusted)")
    print(
        f"  {'suffix':20s}  {'img_tok':>7s}  "
        f"{'E+P%':>7s}  {'E+P%adj':>8s}  "
        f"{'noVrow':>6s}  {'vis/e2e':>8s}  {'e2e':>7s}  "
        f"{'raw>50':>6s}  {'adj>50':>6s}"
    )
    for suffix, epd in rows:
        raw_flag = "yes" if epd.ep_pct > 50 else "no"
        adj_flag = "yes" if epd.ep_pct_adj > 50 else "no"
        print(
            f"  {suffix:20s}  {epd.mean_image_tokens:7.0f}  "
            f"{epd.ep_pct:6.1f}%  {epd.ep_pct_adj:7.1f}%  "
            f"{epd.n_requests_no_vision_row:6d}  "
            f"{epd.mean_vision_over_e2e_pct:7.1f}%  {epd.mean_e2e_ms:7.0f}  "
            f"{raw_flag:>6s}  {adj_flag:>6s}"
        )


def analyze_one(
    gen_path: str,
    vis_path: str,
    mf_path: Optional[str],
    label: str,
    *,
    verbose: bool = True,
) -> EpdBreakdown:
    gen = load_csv(gen_path)
    vis = load_csv(vis_path)
    vision_parsed = parse_vision_rows(vis)
    records = build_request_records(gen, vis)
    epd = compute_epd_breakdown(gen, vis, mf_path, records)

    if not verbose:
        return epd

    print(f"\nFiles:")
    print(f"  generate:      {gen_path}")
    print(f"  vision:        {vis_path}")
    if mf_path:
        print(f"  model_forward: {mf_path}")

    print_epd_table(label, epd)
    print_epd_adjusted_table(label, epd)
    print_vision_row_allocation(vision_parsed, gen)
    print_epd_by_pid_path(records, mf_path)
    print_request_table(label, records)
    print_summary(records, "pid_path")
    print_summary(records, "structural_path")
    print_per_pid_vision(vision_parsed)
    print_per_step(records)
    print_model_forward(mf_path)

    forward_passes = parse_forward_passes(mf_path)
    assign_pids_to_forward_passes(vision_parsed, forward_passes)
    module_rows = build_request_module_breakdown(gen, vision_parsed, forward_passes)
    print_module_breakdown_table(label, module_rows)

    cache_hits = sum(1 for v in vision_parsed if v.cached == 1)
    print(f"\nNotes:")
    print(f"  vision_rows={len(vision_parsed)}  matched_requests={len(records)}  cache_hit_rows={cache_hits}")
    print(
        "  pid_path: first vision_encoder row per SGLang pid = cold; later rows on same pid = warm"
    )
    print(
        "  structural_path: pass_id==1 and single request_id = cold (scheduling), else warm/batched"
    )
    print(
        "  EPD P uses aggregate (sum EXTEND - sum vision); per-request P is approximate in pid_path table"
    )
    return epd


@dataclass
class ForwardPass:
    row_index: int
    pass_id: int
    mode: str
    batch_size: int
    prefill_tokens: int
    forward_ms: float
    global_step: int
    timestamp: float = 0.0
    pid: str = ""
    request_ids: list[str] | None = None


def _gen_float(g: dict, key: str, default: float = 0.0) -> float:
    val = g.get(key)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _assigned_extends_by_pid(
    vision: list[VisionRow], forward_passes: list[ForwardPass]
) -> dict[str, list[tuple[float, list[str]]]]:
    """pid -> ordered list of (forward_ms, request_ids) for each EXTEND pass."""
    matched: set[int] = set()
    by_pid: dict[str, list[tuple[float, list[str]]]] = defaultdict(list)
    for sig in _build_extend_signatures(vision):
        pid = sig["pid"]
        pass_id = sig["pass_id"]
        fp_matches: list[ForwardPass] = []
        for fp in forward_passes:
            if fp.row_index in matched or fp.mode != "EXTEND" or fp.pid != pid:
                continue
            if (
                fp.pass_id == pass_id
                and fp.prefill_tokens == sig["prefill_tokens"]
                and fp.batch_size == sig["batch_size"]
            ):
                fp_matches.append(fp)
                matched.add(fp.row_index)
        if not fp_matches:
            continue
        rids = _request_ids_for_pass(vision, pid, pass_id)
        for fp in fp_matches:
            by_pid[pid].append((fp.forward_ms, rids))
    return by_pid


def compute_batch_sync_wait_by_request(
    vision: list[VisionRow], forward_passes: list[ForwardPass]
) -> dict[str, float]:
    """
    Time on the same replica spent on other requests' EXTEND passes before this
    request's first EXTEND (continuous-batching wait for batchmate prefill).
    """
    extends_by_pid = _assigned_extends_by_pid(vision, forward_passes)
    first_own: dict[str, int] = {}
    for pid, extends in extends_by_pid.items():
        for i, (_, rids) in enumerate(extends):
            for rid in rids:
                if rid not in first_own:
                    first_own[rid] = i
    out: dict[str, float] = {}
    for rid, idx in first_own.items():
        pid = request_pid_map(vision).get(rid, "")
        sync = 0.0
        for i, (forward_ms, _) in enumerate(extends_by_pid.get(pid, [])):
            if i < idx:
                sync += forward_ms
        out[rid] = sync
    return out


@dataclass
class RequestModuleBreakdown:
    """Per-request timing across the VERL → SGLang pipeline."""

    request_id: str
    global_step: int
    pid: str
    image_count: int
    output_tokens: int
    e2e_ms: float
    # VERL-side phases (generate log)
    verl_prepare_ms: float
    sglang_call_ms: float
    verl_post_ms: float
    # SGLang meta (generate log; needs enable_metrics)
    queue_ms: float
    prefill_launch_delay_ms: float
    prefill_launch_latency_ms: float
    # Replica timeline (model_forward + vision)
    batch_sync_wait_ms: float
    extend_wall_ms: float
    decode_wall_ms: float
    sglang_orchestration_ms: float
    decomp_residual_ms: float
    decomp_sum_ms: float
    decomp_sum_pct: float
    # EPD sum view (informational)
    e_ms: float
    p_ms: float
    d_sum_ms: float
    decode_batching_gap_ms: float
    epd_sum_ms: float
    epd_sum_pct: float
    gpu_wall_ms: float
    gpu_wall_pct: float
    overhead_wall_ms: float
    overhead_wall_pct: float
    overhead_sum_ms: float
    overhead_sum_pct: float


def parse_forward_passes(mf_path: Optional[str]) -> list[ForwardPass]:
    if not mf_path or not os.path.exists(mf_path):
        return []
    out: list[ForwardPass] = []
    for i, r in enumerate(load_csv(mf_path)):
        mode = (r.get("mode") or "").upper()
        out.append(
            ForwardPass(
                row_index=i,
                pass_id=int(float(r.get("pass_id") or 0)),
                mode=mode,
                batch_size=max(1, int(float(r.get("batch_size") or 1))),
                prefill_tokens=int(float(r.get("prefill_tokens") or 0)),
                forward_ms=float(r.get("forward_time_ms") or 0),
                global_step=int(float(r.get("global_step") or -1)),
                timestamp=float(r.get("timestamp") or 0),
                # pid is logged directly by patched SGLang (one row per replica
                # process). Older logs omit it -> reconstructed heuristically.
                pid=str(r.get("pid") or ""),
            )
        )
    return out


def forward_passes_have_pid(forward_passes: list[ForwardPass]) -> bool:
    """True when the model_forward_log carried a real pid column (new logs)."""
    return bool(forward_passes) and all(bool(fp.pid) for fp in forward_passes)


def _build_extend_signatures(vision: list[VisionRow]) -> list[dict]:
    """One EXTEND match signature per (pid, pass_id).

    Each vision row is a single encode (one request's image), so a batched EXTEND
    pass shows up as several rows that share (pid, pass_id). We aggregate the
    distinct request ids across all those rows to recover the real batch size,
    rather than reading one row's request_ids (which is now a single id).
    """
    earliest: dict[tuple[str, int], VisionRow] = {}
    rids_by_key: dict[tuple[str, int], list[str]] = defaultdict(list)
    seen_by_key: dict[tuple[str, int], set[str]] = defaultdict(set)
    for v in vision:
        key = (v.pid, v.pass_id)
        if key not in earliest or v.timestamp < earliest[key].timestamp:
            earliest[key] = v
        for rid in v.request_ids:
            if rid not in seen_by_key[key]:
                seen_by_key[key].add(rid)
                rids_by_key[key].append(rid)
    sigs: list[dict] = []
    for (pid, pass_id), v in earliest.items():
        if (v.mode or "EXTEND").upper() != "EXTEND":
            continue
        rids = rids_by_key[(pid, pass_id)]
        sigs.append(
            {
                "timestamp": v.timestamp,
                "pid": pid,
                "pass_id": pass_id,
                "prefill_tokens": v.prefill_tokens,
                "batch_size": max(1, len(rids)),
                "request_ids": list(rids),
            }
        )
    return sorted(sigs, key=lambda x: x["timestamp"])


def assign_pids_to_forward_passes(
    vision: list[VisionRow], forward_passes: list[ForwardPass]
) -> None:
    """Match EXTEND rows to SGLang pid via vision signatures; assign DECODE by replica timeline.

    If the model_forward_log already carries a real ``pid`` column (patched SGLang),
    trust it and only backfill EXTEND request_ids from vision signatures — no DECODE
    round-robin guessing.
    """
    if not forward_passes:
        return

    if forward_passes_have_pid(forward_passes):
        # Real per-replica pid present: only attach request_ids to EXTEND passes
        # (DECODE pid is already correct from the log).
        extend_sigs = _build_extend_signatures(vision)
        sig_by_key: dict[tuple[str, int, int, int], list[dict]] = defaultdict(list)
        for sig in extend_sigs:
            key = (sig["pid"], sig["pass_id"], sig["prefill_tokens"], sig["batch_size"])
            sig_by_key[key].append(sig)
        for fp in forward_passes:
            if fp.mode != "EXTEND":
                continue
            key = (fp.pid, fp.pass_id, fp.prefill_tokens, fp.batch_size)
            if sig_by_key.get(key):
                sig = sig_by_key[key].pop(0)
                fp.request_ids = sig["request_ids"]
        return

    extend_sigs = _build_extend_signatures(vision)
    sig_queues: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for sig in extend_sigs:
        key = (sig["pass_id"], sig["prefill_tokens"], sig["batch_size"])
        sig_queues[key].append(sig)

    # Chunked prefill may emit multiple EXTEND rows with the same signature on one pid.
    extend_keys = [
        (fp.pass_id, fp.prefill_tokens, fp.batch_size)
        for fp in forward_passes
        if fp.mode == "EXTEND"
    ]
    mf_key_counts: dict[tuple[int, int, int], int] = defaultdict(int)
    for key in extend_keys:
        mf_key_counts[key] += 1
    for key, sigs in list(sig_queues.items()):
        need = mf_key_counts.get(key, 0)
        while len(sig_queues[key]) < need and sigs:
            sig_queues[key].append(sigs[-1])

    for fp in forward_passes:
        if fp.mode != "EXTEND":
            continue
        key = (fp.pass_id, fp.prefill_tokens, fp.batch_size)
        if not sig_queues[key]:
            continue
        sig = sig_queues[key].pop(0)
        fp.pid = sig["pid"]
        fp.request_ids = sig["request_ids"]

    extends_per_pid: dict[str, int] = defaultdict(int)
    for sig in extend_sigs:
        extends_per_pid[sig["pid"]] += 1

    extends_done: dict[str, int] = defaultdict(int)
    decode_ready: list[str] = []
    decode_rr_idx = 0

    for fp in forward_passes:
        if fp.mode == "EXTEND" and fp.pid:
            extends_done[fp.pid] += 1
            if (
                extends_done[fp.pid] >= extends_per_pid.get(fp.pid, 0)
                and fp.pid not in decode_ready
            ):
                decode_ready.append(fp.pid)
        elif fp.mode == "DECODE":
            if not decode_ready:
                continue
            if len(decode_ready) == 1:
                fp.pid = decode_ready[0]
            else:
                fp.pid = decode_ready[decode_rr_idx % len(decode_ready)]
                decode_rr_idx += 1


def request_pid_map(vision: list[VisionRow]) -> dict[str, str]:
    """Map request_id -> SGLang pid using the first vision row that mentions it."""
    out: dict[str, str] = {}
    for v in sorted(vision, key=lambda x: x.timestamp):
        for rid in v.request_ids:
            out.setdefault(rid, v.pid)
    return out


def _vision_e_by_pass(vision: list[VisionRow]) -> dict[tuple[str, int], float]:
    total: dict[tuple[str, int], float] = defaultdict(float)
    for v in vision:
        if v.cached != 0:
            continue
        total[(v.pid, v.pass_id)] += v.vision_ms
    return total


def _request_ids_for_pass(vision: list[VisionRow], pid: str, pass_id: int) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for v in vision:
        if v.pid != pid or v.pass_id != pass_id:
            continue
        for rid in v.request_ids:
            if rid not in seen:
                seen.add(rid)
                ids.append(rid)
    return ids


def _build_breakdown_text_only(
    gen: list[dict], forward_passes: list[ForwardPass]
) -> list[RequestModuleBreakdown]:
    """No vision_encoder_log: evenly split aggregate EXTEND/DECODE across requests."""
    pid = "replica0"
    for fp in forward_passes:
        fp.pid = pid
    n = max(1, len(gen))
    extend_total = sum(fp.forward_ms for fp in forward_passes if fp.mode == "EXTEND")
    decode_total = sum(fp.forward_ms for fp in forward_passes if fp.mode == "DECODE")
    extend_share = extend_total / n
    decode_share = decode_total / n
    breakdowns: list[RequestModuleBreakdown] = []
    for g in gen:
        rid = g["request_id"]
        e2e = float(g.get("generate_e2e_ms") or 0)
        verl_prepare = _gen_float(g, "verl_prepare_ms")
        sglang_call = _gen_float(g, "sglang_call_ms")
        verl_post = _gen_float(g, "verl_post_ms")
        if sglang_call <= 0 and e2e > 0:
            sglang_call = max(0.0, e2e - verl_prepare - verl_post)
        queue_ms = _gen_float(g, "queue_time_ms")
        prefill_delay = _gen_float(g, "prefill_launch_delay_ms")
        prefill_latency = _gen_float(g, "prefill_launch_latency_ms")
        sync_wait = 0.0
        extend_wall = extend_share
        decode_wall = decode_share
        p_ms = extend_share
        d_sum = decode_share
        epd_sum = p_ms + d_sum
        sglang_orch = max(
            0.0, sglang_call - queue_ms - sync_wait - extend_wall - decode_wall
        )
        decomp_sum = (
            verl_prepare + queue_ms + sync_wait + extend_wall + decode_wall + sglang_orch + verl_post
        )
        decomp_residual = e2e - decomp_sum
        gpu_wall = extend_wall + decode_wall
        breakdowns.append(
            RequestModuleBreakdown(
                request_id=rid,
                global_step=int(float(g.get("global_step") or 0)),
                pid=pid,
                image_count=0,
                output_tokens=int(float(g.get("output_tokens") or 0)),
                e2e_ms=e2e,
                verl_prepare_ms=verl_prepare,
                sglang_call_ms=sglang_call,
                verl_post_ms=verl_post,
                queue_ms=queue_ms,
                prefill_launch_delay_ms=prefill_delay,
                prefill_launch_latency_ms=prefill_latency,
                batch_sync_wait_ms=sync_wait,
                extend_wall_ms=extend_wall,
                decode_wall_ms=decode_wall,
                sglang_orchestration_ms=sglang_orch,
                decomp_residual_ms=decomp_residual,
                decomp_sum_ms=decomp_sum,
                decomp_sum_pct=(decomp_sum / e2e * 100) if e2e > 0 else 0.0,
                e_ms=0.0,
                p_ms=p_ms,
                d_sum_ms=d_sum,
                decode_batching_gap_ms=0.0,
                epd_sum_ms=epd_sum,
                epd_sum_pct=(epd_sum / e2e * 100) if e2e > 0 else 0.0,
                gpu_wall_ms=gpu_wall,
                gpu_wall_pct=(gpu_wall / e2e * 100) if e2e > 0 else 0.0,
                overhead_wall_ms=max(0.0, e2e - gpu_wall),
                overhead_wall_pct=(max(0.0, e2e - gpu_wall) / e2e * 100) if e2e > 0 else 0.0,
                overhead_sum_ms=max(0.0, e2e - epd_sum),
                overhead_sum_pct=(max(0.0, e2e - epd_sum) / e2e * 100) if e2e > 0 else 0.0,
            )
        )
    return breakdowns


def build_request_module_breakdown(
    gen: list[dict],
    vision: list[VisionRow],
    forward_passes: list[ForwardPass],
) -> list[RequestModuleBreakdown]:
    if not vision:
        return _build_breakdown_text_only(gen, forward_passes)
    per_req_e, _ = allocate_first_encode_e_per_request(vision)
    pid_map = request_pid_map(vision)
    vision_e_pass = _vision_e_by_pass(vision)

    # Per-pid decode wall (sum of DECODE forward ms on that replica).
    decode_wall_by_pid: dict[str, float] = defaultdict(float)
    for fp in forward_passes:
        if fp.mode == "DECODE" and fp.pid:
            decode_wall_by_pid[fp.pid] += fp.forward_ms

    reqs_on_pid: dict[str, list[str]] = defaultdict(list)
    for rid, pid in pid_map.items():
        reqs_on_pid[pid].append(rid)

    extend_wall_by_req: dict[str, float] = defaultdict(float)
    p_sum_by_req: dict[str, float] = defaultdict(float)
    d_sum_by_req: dict[str, float] = defaultdict(float)

    matched_extends: set[int] = set()
    for sig in _build_extend_signatures(vision):
        pid = sig["pid"]
        pass_id = sig["pass_id"]
        rids = _request_ids_for_pass(vision, pid, pass_id)
        n_reqs = max(1, len(rids))
        fp_matches: list[ForwardPass] = []
        for fp in forward_passes:
            if fp.row_index in matched_extends or fp.mode != "EXTEND" or fp.pid != pid:
                continue
            if (
                fp.pass_id == pass_id
                and fp.prefill_tokens == sig["prefill_tokens"]
                and fp.batch_size == sig["batch_size"]
            ):
                fp_matches.append(fp)
                matched_extends.add(fp.row_index)
        if not fp_matches:
            continue
        vision_ms = vision_e_pass.get((pid, pass_id), 0.0)
        forward_total = sum(fp.forward_ms for fp in fp_matches)
        p_total = max(0.0, forward_total - vision_ms)
        p_share = p_total / n_reqs
        for fp in fp_matches:
            for rid in rids:
                extend_wall_by_req[rid] += fp.forward_ms
        for rid in rids:
            p_sum_by_req[rid] += p_share

    for fp in forward_passes:
        if fp.mode != "DECODE" or not fp.pid:
            continue
        n_reqs = max(1, len(reqs_on_pid.get(fp.pid, [])))
        share = fp.forward_ms / n_reqs
        for rid in reqs_on_pid.get(fp.pid, []):
            d_sum_by_req[rid] += share

    batch_sync = compute_batch_sync_wait_by_request(vision, forward_passes)

    breakdowns: list[RequestModuleBreakdown] = []
    for g in gen:
        rid = g["request_id"]
        e2e = float(g.get("generate_e2e_ms") or 0)
        pid = pid_map.get(rid, "")
        e_ms = per_req_e.get(rid, 0.0)
        p_ms = p_sum_by_req.get(rid, 0.0)
        d_sum = d_sum_by_req.get(rid, 0.0)
        epd_sum = e_ms + p_ms + d_sum
        decode_wall = decode_wall_by_pid.get(pid, 0.0)
        extend_wall = extend_wall_by_req.get(rid, 0.0)
        gpu_wall = extend_wall + decode_wall
        overhead_wall = max(0.0, e2e - gpu_wall)
        overhead_sum = max(0.0, e2e - epd_sum)
        decode_gap = max(0.0, decode_wall - d_sum)
        sync_wait = batch_sync.get(rid, 0.0)

        verl_prepare = _gen_float(g, "verl_prepare_ms")
        sglang_call = _gen_float(g, "sglang_call_ms")
        verl_post = _gen_float(g, "verl_post_ms")
        if sglang_call <= 0 and e2e > 0:
            sglang_call = max(0.0, e2e - verl_prepare - verl_post)

        queue_ms = _gen_float(g, "queue_time_ms")
        prefill_delay = _gen_float(g, "prefill_launch_delay_ms")
        prefill_latency = _gen_float(g, "prefill_launch_latency_ms")

        sglang_orch = max(
            0.0,
            sglang_call - queue_ms - sync_wait - extend_wall - decode_wall,
        )
        decomp_sum = (
            verl_prepare + queue_ms + sync_wait + extend_wall + decode_wall + sglang_orch + verl_post
        )
        decomp_residual = e2e - decomp_sum

        breakdowns.append(
            RequestModuleBreakdown(
                request_id=rid,
                global_step=int(float(g.get("global_step") or 0)),
                pid=pid,
                image_count=int(float(g.get("image_count") or 0)),
                output_tokens=int(float(g.get("output_tokens") or 0)),
                e2e_ms=e2e,
                verl_prepare_ms=verl_prepare,
                sglang_call_ms=sglang_call,
                verl_post_ms=verl_post,
                queue_ms=queue_ms,
                prefill_launch_delay_ms=prefill_delay,
                prefill_launch_latency_ms=prefill_latency,
                batch_sync_wait_ms=sync_wait,
                extend_wall_ms=extend_wall,
                decode_wall_ms=decode_wall,
                sglang_orchestration_ms=sglang_orch,
                decomp_residual_ms=decomp_residual,
                decomp_sum_ms=decomp_sum,
                decomp_sum_pct=(decomp_sum / e2e * 100) if e2e > 0 else 0.0,
                e_ms=e_ms,
                p_ms=p_ms,
                d_sum_ms=d_sum,
                decode_batching_gap_ms=decode_gap,
                epd_sum_ms=epd_sum,
                epd_sum_pct=(epd_sum / e2e * 100) if e2e > 0 else 0.0,
                gpu_wall_ms=gpu_wall,
                gpu_wall_pct=(gpu_wall / e2e * 100) if e2e > 0 else 0.0,
                overhead_wall_ms=overhead_wall,
                overhead_wall_pct=(overhead_wall / e2e * 100) if e2e > 0 else 0.0,
                overhead_sum_ms=overhead_sum,
                overhead_sum_pct=(overhead_sum / e2e * 100) if e2e > 0 else 0.0,
            )
        )
    return breakdowns


BREAKDOWN_CSV_FIELDS = [
    "request_id",
    "global_step",
    "pid",
    "image_count",
    "output_tokens",
    "e2e_ms",
    "verl_prepare_ms",
    "sglang_call_ms",
    "verl_post_ms",
    "queue_ms",
    "prefill_launch_delay_ms",
    "prefill_launch_latency_ms",
    "batch_sync_wait_ms",
    "extend_wall_ms",
    "decode_wall_ms",
    "sglang_orchestration_ms",
    "decomp_sum_ms",
    "decomp_sum_pct",
    "decomp_residual_ms",
    "e_ms",
    "p_ms",
    "d_sum_ms",
    "decode_batching_gap_ms",
    "epd_sum_ms",
    "epd_sum_pct",
    "gpu_wall_ms",
    "gpu_wall_pct",
    "overhead_wall_ms",
    "overhead_wall_pct",
]


def write_breakdown_csv(rows: list[RequestModuleBreakdown], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BREAKDOWN_CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "request_id": r.request_id,
                    "global_step": r.global_step,
                    "pid": r.pid,
                    "image_count": r.image_count,
                    "output_tokens": r.output_tokens,
                    "e2e_ms": f"{r.e2e_ms:.2f}",
                    "verl_prepare_ms": f"{r.verl_prepare_ms:.2f}",
                    "sglang_call_ms": f"{r.sglang_call_ms:.2f}",
                    "verl_post_ms": f"{r.verl_post_ms:.2f}",
                    "queue_ms": f"{r.queue_ms:.2f}",
                    "prefill_launch_delay_ms": f"{r.prefill_launch_delay_ms:.2f}",
                    "prefill_launch_latency_ms": f"{r.prefill_launch_latency_ms:.2f}",
                    "batch_sync_wait_ms": f"{r.batch_sync_wait_ms:.2f}",
                    "extend_wall_ms": f"{r.extend_wall_ms:.2f}",
                    "decode_wall_ms": f"{r.decode_wall_ms:.2f}",
                    "sglang_orchestration_ms": f"{r.sglang_orchestration_ms:.2f}",
                    "decomp_sum_ms": f"{r.decomp_sum_ms:.2f}",
                    "decomp_sum_pct": f"{r.decomp_sum_pct:.2f}",
                    "decomp_residual_ms": f"{r.decomp_residual_ms:.2f}",
                    "e_ms": f"{r.e_ms:.2f}",
                    "p_ms": f"{r.p_ms:.2f}",
                    "d_sum_ms": f"{r.d_sum_ms:.2f}",
                    "decode_batching_gap_ms": f"{r.decode_batching_gap_ms:.2f}",
                    "epd_sum_ms": f"{r.epd_sum_ms:.2f}",
                    "epd_sum_pct": f"{r.epd_sum_pct:.2f}",
                    "gpu_wall_ms": f"{r.gpu_wall_ms:.2f}",
                    "gpu_wall_pct": f"{r.gpu_wall_pct:.2f}",
                    "overhead_wall_ms": f"{r.overhead_wall_ms:.2f}",
                    "overhead_wall_pct": f"{r.overhead_wall_pct:.2f}",
                }
            )


SUMMARY_CSV_FIELDS = [
    "metric",
    "n",
    "mean_ms",
    "median_ms",
    "min_ms",
    "max_ms",
    "pct_of_e2e",
]


def write_breakdown_summary_csv(rows: list[RequestModuleBreakdown], path: str) -> None:
    if not rows:
        return
    mean_e2e = st.mean(r.e2e_ms for r in rows)

    def col(getter):
        vals = [getter(r) for r in rows]
        return vals, st.mean(vals), st.median(vals), min(vals), max(vals)

    metrics = [
        ("e2e", lambda r: r.e2e_ms),
        ("verl_prepare", lambda r: r.verl_prepare_ms),
        ("sglang_call", lambda r: r.sglang_call_ms),
        ("verl_post", lambda r: r.verl_post_ms),
        ("queue", lambda r: r.queue_ms),
        ("prefill_launch_delay", lambda r: r.prefill_launch_delay_ms),
        ("prefill_launch_latency", lambda r: r.prefill_launch_latency_ms),
        ("batch_sync_wait", lambda r: r.batch_sync_wait_ms),
        ("extend_wall", lambda r: r.extend_wall_ms),
        ("decode_wall", lambda r: r.decode_wall_ms),
        ("sglang_orchestration", lambda r: r.sglang_orchestration_ms),
        ("decomp_residual", lambda r: r.decomp_residual_ms),
        ("e_vision_encoder", lambda r: r.e_ms),
        ("decode_batching_gap", lambda r: r.decode_batching_gap_ms),
        ("epd_sum_allocated", lambda r: r.epd_sum_ms),
    ]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        for name, getter in metrics:
            vals, mean_v, med_v, min_v, max_v = col(getter)
            pct = (mean_v / mean_e2e * 100) if mean_e2e > 0 else 0.0
            writer.writerow(
                {
                    "metric": name,
                    "n": len(vals),
                    "mean_ms": f"{mean_v:.2f}",
                    "median_ms": f"{med_v:.2f}",
                    "min_ms": f"{min_v:.2f}",
                    "max_ms": f"{max_v:.2f}",
                    "pct_of_e2e": f"{pct:.2f}",
                }
            )


def print_module_breakdown_table(label: str, rows: list[RequestModuleBreakdown]) -> None:
    if not rows:
        return
    has_queue = any(r.queue_ms > 0 for r in rows)
    print(f"\n{'=' * 80}\n{label}\nE2E DECOMPOSITION (components sum to e2e)")
    print(
        "  e2e = verl_prepare + queue + batch_sync + extend_wall + decode_wall\n"
        "        + sglang_orchestration + verl_post  (+ decomp_residual ≈ 0)\n"
        "  sglang_orchestration = sglang_call - queue - batch_sync - extend_wall - decode_wall\n"
        "    (mm preprocess, inter-phase scheduling, detokenize, IPC inside SGLang)"
    )
    if not has_queue:
        print("  NOTE: queue_ms=0 — re-run with updated VERL + enable_metrics profiling build.")

    print(
        f"\n  {'request':8s}  {'e2e':>6s}  {'vPrep':>5s}  {'queue':>5s}  {'sync':>5s}  "
        f"{'extW':>5s}  {'decW':>5s}  {'sgOr':>5s}  {'vPost':>5s}  {'resid':>5s}  pid"
    )
    for r in rows:
        print(
            f"  {r.request_id[:8]:8s}  {r.e2e_ms:6.0f}  {r.verl_prepare_ms:5.0f}  "
            f"{r.queue_ms:5.0f}  {r.batch_sync_wait_ms:5.0f}  {r.extend_wall_ms:5.0f}  "
            f"{r.decode_wall_ms:5.0f}  {r.sglang_orchestration_ms:5.0f}  "
            f"{r.verl_post_ms:5.0f}  {r.decomp_residual_ms:5.0f}  {r.pid}"
        )

    mean_e2e = st.mean(r.e2e_ms for r in rows)
    part_rows = [
        ("verl_prepare", [r.verl_prepare_ms for r in rows]),
        ("queue", [r.queue_ms for r in rows]),
        ("batch_sync", [r.batch_sync_wait_ms for r in rows]),
        ("extend_wall", [r.extend_wall_ms for r in rows]),
        ("decode_wall", [r.decode_wall_ms for r in rows]),
        ("sglang_orch", [r.sglang_orchestration_ms for r in rows]),
        ("verl_post", [r.verl_post_ms for r in rows]),
    ]
    print(f"\n  MEAN pct of e2e (n={len(rows)}, e2e={mean_e2e:.0f}ms):")
    for name, vals in part_rows:
        m = st.mean(vals)
        print(f"    {name:14s}  {m:7.0f} ms  ({m / mean_e2e * 100:5.1f}%)")
    print(f"    {'decomp_residual':14s}  {st.mean(r.decomp_residual_ms for r in rows):7.0f} ms")


def compute_decode_stall_stats(forward_passes: list[ForwardPass]) -> dict:
    """Per replica: idle time inside the decode window = decode stalled by other work.

    Evidence for "E/P blocks D": once a replica starts decoding, an interleaved EXTEND
    (a new request's prefill) pauses the decode stream. We measure:
      - decode_span   = last_decode_ts - first_decode_ts on a pid
      - decode_active = sum of DECODE forward_ms on that pid
      - decode_idle   = max(0, decode_span - decode_active)  (time decode was NOT running)
      - extend_in_decode = EXTEND passes whose timestamp falls inside the decode window
    Requires timestamps in model_forward_log; returns zeros if missing.
    """
    has_ts = any(fp.timestamp > 0 for fp in forward_passes)
    by_pid_dec: dict[str, list[ForwardPass]] = defaultdict(list)
    by_pid_ext: dict[str, list[ForwardPass]] = defaultdict(list)
    for fp in forward_passes:
        if not fp.pid:
            continue
        if fp.mode == "DECODE":
            by_pid_dec[fp.pid].append(fp)
        elif fp.mode == "EXTEND":
            by_pid_ext[fp.pid].append(fp)

    per_pid: list[dict] = []
    tot_idle = tot_span = tot_active = 0.0
    tot_extend_in_decode = 0
    tot_extend_in_decode_ms = 0.0
    for pid, decs in by_pid_dec.items():
        if not decs:
            continue
        decs_sorted = sorted(decs, key=lambda x: x.timestamp)
        first_ts = decs_sorted[0].timestamp
        last_ts = decs_sorted[-1].timestamp
        span_ms = max(0.0, (last_ts - first_ts) * 1000.0) if has_ts else 0.0
        # active excludes the first pass (timestamp ~ end of pass); approx with sum of all.
        active_ms = sum(fp.forward_ms for fp in decs_sorted)
        idle_ms = max(0.0, span_ms - active_ms) if has_ts else 0.0
        ext_in = [
            fp for fp in by_pid_ext.get(pid, [])
            if has_ts and first_ts <= fp.timestamp <= last_ts
        ]
        ext_in_ms = sum(fp.forward_ms for fp in ext_in)
        per_pid.append(
            {
                "pid": pid,
                "n_decode": len(decs_sorted),
                "decode_span_ms": span_ms,
                "decode_active_ms": active_ms,
                "decode_idle_ms": idle_ms,
                "extend_in_decode": len(ext_in),
                "extend_in_decode_ms": ext_in_ms,
            }
        )
        tot_idle += idle_ms
        tot_span += span_ms
        tot_active += active_ms
        tot_extend_in_decode += len(ext_in)
        tot_extend_in_decode_ms += ext_in_ms

    return {
        "has_timestamp": has_ts,
        "per_pid": per_pid,
        "total_idle_ms": tot_idle,
        "total_span_ms": tot_span,
        "total_active_ms": tot_active,
        "total_extend_in_decode": tot_extend_in_decode,
        "total_extend_in_decode_ms": tot_extend_in_decode_ms,
        "n_pids": len(per_pid),
    }


PAPER_ALIGNED_FIELDS = ["group", "metric", "value_ms", "denominator", "pct", "note"]


def build_paper_aligned_rows(
    rows: list[RequestModuleBreakdown],
    epd: EpdBreakdown,
    vision: list[VisionRow],
    stall: dict,
) -> list[dict]:
    """Two-denominator view aligned with EPD-disaggregation papers.

    Group A (stage execution): denominator = E+P+D GPU-forward compute (NOT e2e).
    Group B (per-request wait/exec): denominator = e2e wall clock (mean across requests).
    Group C (encode execution raw): per-ViT-pass wall time, no denominator.
    Group D (decode stall): interference evidence.
    """
    out: list[dict] = []
    fwd_total = epd.total_ms or 1.0

    # Group A: stage execution, denom = E+P+D (GPU forward compute)
    out.append(
        {"group": "A_stage_execution", "metric": "encode_execution", "value_ms": epd.e_ms,
         "denominator": "E+P+D", "pct": epd.e_ms / fwd_total * 100,
         "note": "sum ViT forward (GPU-seconds, not wall clock)"}
    )
    out.append(
        {"group": "A_stage_execution", "metric": "prefill_execution", "value_ms": epd.p_ms,
         "denominator": "E+P+D", "pct": epd.p_ms / fwd_total * 100,
         "note": "sum EXTEND forward - encode (LLM prefill)"}
    )
    out.append(
        {"group": "A_stage_execution", "metric": "decode_execution", "value_ms": epd.d_ms,
         "denominator": "E+P+D", "pct": epd.d_ms / fwd_total * 100,
         "note": "sum DECODE forward"}
    )
    out.append(
        {"group": "A_stage_execution", "metric": "encode+prefill", "value_ms": epd.e_ms + epd.p_ms,
         "denominator": "E+P+D", "pct": epd.ep_pct,
         "note": "front-end-bound if >50% (paper EPD regime)"}
    )

    # Group B: per-request wall-clock attribution, denom = e2e (mean)
    n = len(rows) or 1
    mean_e2e = st.mean(r.e2e_ms for r in rows) if rows else 0.0
    e2e_denom = mean_e2e or 1.0

    def add_b(metric, getter, note):
        m = sum(getter(r) for r in rows) / n
        out.append(
            {"group": "B_per_request_e2e", "metric": metric, "value_ms": m,
             "denominator": "e2e", "pct": m / e2e_denom * 100, "note": note}
        )

    add_b("e2e", lambda r: r.e2e_ms, "per-request end-to-end wall clock")
    add_b("queue_wait", lambda r: r.queue_ms, "SGLang waiting-queue time")
    add_b("prefill_blocking_wait", lambda r: r.batch_sync_wait_ms,
          "WAIT: same-replica EXTEND before this req (stage interference, NOT a module)")
    add_b("prefill_execution_extend", lambda r: r.extend_wall_ms,
          "this request's EXTEND GPU wall (incl ViT)")
    add_b("decode_execution", lambda r: r.decode_wall_ms, "replica decode-stream wall")
    add_b("orchestration", lambda r: r.sglang_orchestration_ms,
          "host/scheduling/detokenize residual")

    # Group C: raw encode execution per ViT pass (no denominator)
    miss = [v for v in vision if v.cached == 0 and v.vision_ms > 0]
    if miss:
        vt = [v.vision_ms for v in miss]
        out.append(
            {"group": "C_encode_raw", "metric": "encode_execution_raw_mean",
             "value_ms": st.mean(vt), "denominator": "per_pass", "pct": 0.0,
             "note": f"raw ViT pass wall; n_passes={len(vt)} cached=0"}
        )
        out.append(
            {"group": "C_encode_raw", "metric": "encode_execution_raw_median",
             "value_ms": st.median(vt), "denominator": "per_pass", "pct": 0.0, "note": ""}
        )
        out.append(
            {"group": "C_encode_raw", "metric": "encode_execution_raw_max",
             "value_ms": max(vt), "denominator": "per_pass", "pct": 0.0,
             "note": "cold first-encode per replica"}
        )

    # Group D: decode stall (interference evidence)
    if stall.get("has_timestamp"):
        out.append(
            {"group": "D_decode_stall", "metric": "decode_idle_in_window",
             "value_ms": stall["total_idle_ms"], "denominator": "sum_over_pids", "pct": 0.0,
             "note": "decode_span - decode_active summed over replicas (decode paused)"}
        )
        out.append(
            {"group": "D_decode_stall", "metric": "extend_interleaved_in_decode",
             "value_ms": stall["total_extend_in_decode_ms"], "denominator": "sum_over_pids",
             "pct": 0.0,
             "note": f"count={stall['total_extend_in_decode']} EXTEND passes inside decode window"}
        )
    else:
        out.append(
            {"group": "D_decode_stall", "metric": "decode_idle_in_window", "value_ms": 0.0,
             "denominator": "sum_over_pids", "pct": 0.0,
             "note": "model_forward_log has no timestamp; re-run patched SGLang to enable"}
        )
    return out


def write_paper_aligned_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAPER_ALIGNED_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "group": r["group"],
                    "metric": r["metric"],
                    "value_ms": f"{r['value_ms']:.2f}",
                    "denominator": r["denominator"],
                    "pct": f"{r['pct']:.2f}",
                    "note": r["note"],
                }
            )


def print_paper_aligned_table(label: str, rows: list[dict], stall: dict) -> None:
    print(f"\n{'=' * 80}\n{label}\nPAPER-ALIGNED VIEW (EPD-disaggregation口径)")
    print(
        "  Two denominators on purpose:\n"
        "    A stage_execution  -> denom = E+P+D GPU-forward compute (paper 'front-end bound')\n"
        "    B per_request_e2e  -> denom = e2e wall clock (where a request's time goes)\n"
        "    C encode_raw       -> raw ViT pass wall (not divided)\n"
        "    D decode_stall     -> decode paused by interleaved prefill (interference)"
    )
    cur = None
    for r in rows:
        if r["group"] != cur:
            cur = r["group"]
            print(f"\n  [{cur}]")
        pct = f"{r['pct']:5.1f}%" if r["denominator"] in ("E+P+D", "e2e") else "   -  "
        print(f"    {r['metric']:26s} {r['value_ms']:9.1f} ms  {pct}  ({r['denominator']})")
        if r["note"]:
            print(f"        ^ {r['note']}")
    if stall.get("has_timestamp") and stall.get("per_pid"):
        print("\n  decode-stall per replica:")
        for p in stall["per_pid"]:
            print(
                f"    pid {p['pid']}: decode n={p['n_decode']} span={p['decode_span_ms']:.0f}ms "
                f"active={p['decode_active_ms']:.0f}ms idle={p['decode_idle_ms']:.0f}ms "
                f"extend_in_window={p['extend_in_decode']}"
            )


@dataclass
class ProfilingAnalysisBundle:
    """All artifacts from one profiling run (for CSV export + Markdown report)."""

    label: str
    gen_path: str
    vis_path: str
    mf_path: Optional[str]
    breakdown_csv: str
    summary_csv: str
    paper_csv: str
    rows: list[RequestModuleBreakdown]
    epd: EpdBreakdown
    vision: list[VisionRow]
    gen: list[dict]
    stall: dict
    paper_rows: list[dict]
    cold_info: Optional[dict] = None


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    lines = ["| " + " | ".join(headers) + " |", sep]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _fmt_ms(v: float, digits: int = 1) -> str:
    return f"{v:.{digits}f}"


def _fmt_pct(v: float, digits: int = 1) -> str:
    return f"{v:.{digits}f}%"


def apply_cold_start_filter(
    gen: list[dict],
    vis_raw: list[dict],
    vision: list[VisionRow],
    forward_passes: list[ForwardPass],
    drop_cold_step: bool,
) -> tuple[list[dict], list[dict], list[VisionRow], list[ForwardPass], dict]:
    """Remove cold-start work that otherwise dominates the EXTEND/prefill bucket.

    The first forward on each replica pays a one-time cost (FSDP param onload,
    CUDA allocator warmup, kernel autotune) that can be ~6-7s and is NOT steady
    state. Two cases:
      - Multiple training steps logged -> drop the lowest global_step entirely
        (recommended; the rest is warm).
      - Only one step (legacy single-step run) -> drop each replica's first EXTEND
        pass (and its vision rows) as the cold-compile pass.
    """
    info = {
        "enabled": drop_cold_step,
        "mode": "none",
        "dropped_step": None,
        "steps": [],
        "single_step": False,
        "n_dropped_extend": 0,
    }
    steps = sorted(
        {fp.global_step for fp in forward_passes if fp.global_step is not None and fp.global_step >= 0}
    )
    info["steps"] = steps
    info["single_step"] = len(steps) <= 1
    if not drop_cold_step:
        return gen, vis_raw, vision, forward_passes, info

    def _step(d: dict) -> int:
        return int(float(d.get("global_step") or -1))

    if len(steps) > 1:
        cold = steps[0]
        info["mode"] = "dropped_step"
        info["dropped_step"] = cold
        gen = [g for g in gen if _step(g) != cold]
        vis_raw = [r for r in vis_raw if _step(r) != cold]
        vision = [v for v in vision if v.global_step != cold]
        forward_passes = [fp for fp in forward_passes if fp.global_step != cold]
        return gen, vis_raw, vision, forward_passes, info

    # Single step: drop each replica's first (cold-compile) EXTEND pass.
    by_pid_extends: dict[str, list[ForwardPass]] = defaultdict(list)
    for fp in forward_passes:
        if fp.mode == "EXTEND" and fp.pid:
            by_pid_extends[fp.pid].append(fp)
    drop_rows: set[int] = set()
    cold_keys: set[tuple[str, int]] = set()
    for pid, fps in by_pid_extends.items():
        earliest = min(fps, key=lambda x: (x.timestamp, x.pass_id, x.row_index))
        drop_rows.add(earliest.row_index)
        cold_keys.add((earliest.pid, earliest.pass_id))
    if drop_rows:
        info["mode"] = "dropped_first_extend"
        info["n_dropped_extend"] = len(drop_rows)
        forward_passes = [fp for fp in forward_passes if fp.row_index not in drop_rows]
        vision = [v for v in vision if (v.pid, v.pass_id) not in cold_keys]
        keep_keys = cold_keys
        vis_raw = [
            r
            for r in vis_raw
            if (str(r.get("pid") or ""), int(float(r.get("pass_id") or -1))) not in keep_keys
        ]
    return gen, vis_raw, vision, forward_passes, info


def run_breakdown_analysis(
    gen_path: str,
    vis_path: Optional[str],
    mf_path: Optional[str],
    drop_cold_step: bool = True,
) -> tuple[
    list[RequestModuleBreakdown],
    EpdBreakdown,
    list[VisionRow],
    list[dict],
    dict,
    list[dict],
    dict,
]:
    gen = load_csv(gen_path)
    vis_raw = load_csv(vis_path) if vis_path and os.path.isfile(vis_path) else []
    vision = parse_vision_rows(vis_raw)
    forward_passes = parse_forward_passes(mf_path)
    assign_pids_to_forward_passes(vision, forward_passes)
    gen, vis_raw, vision, forward_passes, cold_info = apply_cold_start_filter(
        gen, vis_raw, vision, forward_passes, drop_cold_step
    )
    rows = build_request_module_breakdown(gen, vision, forward_passes)
    epd = compute_epd_breakdown(gen, vis_raw, mf_path, forward_passes=forward_passes)
    stall = compute_decode_stall_stats(forward_passes)
    paper_rows = build_paper_aligned_rows(rows, epd, vision, stall)
    return rows, epd, vision, gen, stall, paper_rows, cold_info


def write_profiling_report_md(bundle: ProfilingAnalysisBundle, report_md: str) -> None:
    """Write a self-contained Markdown report with all summary tables."""
    rows = bundle.rows
    epd = bundle.epd
    gen = bundle.gen
    stall = bundle.stall
    paper_rows = bundle.paper_rows
    n = len(rows) or 1
    mean_e2e = st.mean(r.e2e_ms for r in rows) if rows else 0.0

    # --- workload stats from generate log ---
    def int_field(name: str) -> list[int]:
        return [int(float(r.get(name) or 0)) for r in gen]

    image_counts = int_field("image_count")
    prompt_tokens = int_field("prompt_tokens")
    image_tokens = int_field("image_prompt_tokens")
    output_tokens = int_field("output_tokens")
    e2e_list = [float(r.get("generate_e2e_ms") or 0) for r in gen]
    cached_rows = sum(1 for v in bundle.vision if v.cached != 0)
    miss_rows = sum(1 for v in bundle.vision if v.cached == 0 and v.vision_ms > 0)

    # --- e2e decomposition rows ---
    e2e_metrics = [
        ("verl_prepare", lambda r: r.verl_prepare_ms),
        ("queue", lambda r: r.queue_ms),
        ("prefill_blocking_wait (batch_sync)", lambda r: r.batch_sync_wait_ms),
        ("prefill_execution (extend_wall)", lambda r: r.extend_wall_ms),
        ("decode_execution (decode_wall)", lambda r: r.decode_wall_ms),
        ("sglang_orchestration", lambda r: r.sglang_orchestration_ms),
        ("verl_post", lambda r: r.verl_post_ms),
        ("decomp_residual", lambda r: r.decomp_residual_ms),
    ]
    e2e_table_rows: list[list[str]] = []
    for name, getter in e2e_metrics:
        vals = [getter(r) for r in rows]
        m = st.mean(vals) if vals else 0.0
        pct = m / mean_e2e * 100 if mean_e2e > 0 else 0.0
        e2e_table_rows.append([name, str(n), _fmt_ms(m), _fmt_pct(pct)])

    # --- EPD raw ---
    epd_raw_rows = [
        ["E (encode)", _fmt_ms(epd.e_ms), _fmt_pct(epd.pct(epd.e_ms)), "ViT / vision_encoder_log sum"],
        ["P (prefill)", _fmt_ms(epd.p_ms), _fmt_pct(epd.pct(epd.p_ms)), "sum(EXTEND) - E"],
        ["D (decode)", _fmt_ms(epd.d_ms), _fmt_pct(epd.pct(epd.d_ms)), "sum(DECODE)"],
        ["E+P", _fmt_ms(epd.e_ms + epd.p_ms), _fmt_pct(epd.ep_pct), "front-end bound if >50%"],
        ["E+P+D total", _fmt_ms(epd.total_ms), "100.0%", "GPU forward stages"],
    ]

    # --- EPD adjusted ---
    epd_adj_rows = [
        ["E (adj)", _fmt_ms(epd.e_ms_adj), _fmt_pct(epd.pct(epd.e_ms_adj, epd.total_ms_adj))],
        ["P (adj)", _fmt_ms(epd.p_ms_adj), _fmt_pct(epd.pct(epd.p_ms_adj, epd.total_ms_adj))],
        ["D (adj)", _fmt_ms(epd.d_ms_adj), _fmt_pct(epd.pct(epd.d_ms_adj, epd.total_ms_adj))],
        ["E+P (adj)", _fmt_ms(epd.e_ms_adj + epd.p_ms_adj), _fmt_pct(epd.ep_pct_adj)],
    ]

    # --- paper-aligned by group ---
    paper_by_group: dict[str, list[list[str]]] = defaultdict(list)
    for r in paper_rows:
        pct_s = _fmt_pct(r["pct"]) if r["denominator"] in ("E+P+D", "e2e") else "-"
        paper_by_group[r["group"]].append(
            [r["metric"], _fmt_ms(r["value_ms"]), pct_s, r["denominator"], r["note"]]
        )

    # --- decode stall ---
    stall_rows: list[list[str]] = []
    if stall.get("has_timestamp"):
        for p in stall.get("per_pid", []):
            stall_rows.append(
                [
                    str(p["pid"]),
                    str(p["n_decode"]),
                    _fmt_ms(p["decode_span_ms"], 0),
                    _fmt_ms(p["decode_active_ms"], 0),
                    _fmt_ms(p["decode_idle_ms"], 0),
                    str(p["extend_in_decode"]),
                ]
            )
    else:
        stall_rows.append(["(no timestamp in model_forward_log)", "-", "-", "-", "-", "-"])

    # --- auto bullets ---
    blocking_pct = next(
        (r["pct"] for r in paper_rows if r["metric"] == "prefill_blocking_wait"), 0.0
    )
    ep_pct_a = next((r["pct"] for r in paper_rows if r["metric"] == "encode+prefill"), 0.0)
    bullets: list[str] = []
    if ep_pct_a > 50:
        bullets.append(
            f"- **GPU 算力偏 prefill 侧（见上文 §5 块 A）**：E+P 占全部 GPU 前向时间 **{_fmt_pct(ep_pct_a)}**（>50% 即 EPD 所称 front-end / prefill 侧重）。"
        )
    if blocking_pct > 40:
        bullets.append(
            f"- **墙钟主要在等同 batch（见上文 §5 块 B）**：`prefill_blocking_wait` 占 e2e **{_fmt_pct(blocking_pct)}**——同 replica 上排队等别人 EXTEND，是耦合架构的主要痛点。"
        )
    if miss_rows > 0 and cached_rows == 0:
        bullets.append(
            f"- **无 image cache**：vision 行 {miss_rows} 条 encode、0 条 cached；GRPO 同图可能重复 ViT。"
        )
    if stall.get("has_timestamp"):
        if stall.get("total_extend_in_decode", 0) == 0:
            bullets.append(
                "- **Decode 停顿形态**：decode 窗口内无 EXTEND 插队（burst prefill 先跑完再 decode）；"
                f"但 decode_idle 合计 **{_fmt_ms(stall['total_idle_ms'], 0)}ms**（等 batchmate 同步）。"
            )
        else:
            bullets.append(
                f"- **Decode 被 prefill 打断**：decode 窗口内 interleaved EXTEND = {stall['total_extend_in_decode']} 次。"
            )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cold = bundle.cold_info or {}
    cold_lines: list[str] = []
    if cold.get("mode") == "dropped_step":
        cold_lines = [
            f"> **冷启动已剔除**：检测到多步 {cold.get('steps')}，分析已丢弃最低 step "
            f"`{cold.get('dropped_step')}`（首次 forward 的一次性 FSDP onload/编译/autotune 开销），"
            "下列数字为稳态步。",
            "",
        ]
    elif cold.get("mode") == "dropped_first_extend":
        cold_lines = [
            f"> **冷启动已部分剔除（单步日志）**：已丢弃每个 replica 的首条 EXTEND pass "
            f"（共 {cold.get('n_dropped_extend')} 条，约 6–7s 的一次性编译/onload 开销）及其 vision 行。",
            ">",
            "> ⚠️ 单步下这只修正了 **§3 / §5-A 的 GPU 算力视角**；**§2 / §5-B 墙钟分解仍不可信**——"
            "因为各请求 `e2e` 里那段“等冷启动”的墙钟无法从 generate 日志里扣除，会被归到 `orchestration` 残差里。"
            "**请重跑 ≥2 步**（脚本已默认 `TOTAL_STEPS=2`），分析会自动丢弃 step 0、给出真正稳态的墙钟分解。",
            "",
        ]
    elif cold.get("enabled") and cold.get("single_step"):
        cold_lines = [
            "> **冷启动警告**：仅单步日志且未能定位 replica 首条 EXTEND；下列 EXTEND/prefill 可能被一次性"
            "编译/onload 开销主导。建议重跑 ≥2 步（脚本已默认 `TOTAL_STEPS=2`）。",
            "",
        ]
    elif not cold.get("enabled"):
        cold_lines = [
            "> 注：`--keep-cold-step` 已开启，未剔除冷启动；若为首步/单步数据，EXTEND 占比会偏高。",
            "",
        ]
    lines: list[str] = [
        f"# Profiling Report: {bundle.label}",
        "",
        f"Generated: {ts}",
        "",
        *cold_lines,
        "## 输入日志",
        "",
        "| 文件 | 路径 |",
        "| --- | --- |",
        f"| generate | `{bundle.gen_path}` |",
        f"| vision | `{bundle.vis_path}` |",
        f"| model_forward | `{bundle.mf_path or '(missing)'}` |",
        "",
        "## 导出 CSV",
        "",
        "| 文件 | 路径 |",
        "| --- | --- |",
        f"| per-request breakdown | `{bundle.breakdown_csv}` |",
        f"| e2e summary | `{bundle.summary_csv}` |",
        f"| paper-aligned | `{bundle.paper_csv}` |",
        "",
        "## 1. Workload 概览",
        "",
        _md_table(
            ["指标", "值"],
            [
                ["请求数", str(len(gen))],
                ["image_count 取值", ", ".join(str(x) for x in sorted(set(image_counts)))],
                ["prompt_tokens", f"{min(prompt_tokens)} .. {max(prompt_tokens)}"],
                ["image_prompt_tokens", f"{min(image_tokens)} .. {max(image_tokens)}"],
                ["output_tokens", f"{min(output_tokens)} .. {max(output_tokens)}"],
                ["e2e (ms)", stats(e2e_list)],
                ["vision rows (cached / miss)", f"{cached_rows} / {miss_rows}"],
                ["EXTEND passes / DECODE passes", f"{epd.n_extend_passes} / {epd.n_decode_passes}"],
            ],
        ),
        "",
        "## 2. E2E 分解（分母 = mean e2e）",
        "",
        "公式：`e2e ≈ verl_prepare + queue + prefill_blocking_wait + extend_wall + decode_wall + sglang_orch + verl_post`",
        "",
        _md_table(
            ["模块", "n", "mean_ms", "% of e2e"],
            e2e_table_rows,
        ),
        "",
        "## 3. EPD 阶段执行（分母 = E+P+D GPU compute）",
        "",
        _md_table(
            ["阶段", "ms", "% of E+P+D", "说明"],
            epd_raw_rows,
        ),
        "",
        f"EPD threshold (E+P > 50%): **{'YES' if epd.ep_pct > 50 else 'no'}**",
        "",
        "## 4. EPD Adjusted（首 encode / cache-miss）",
        "",
        _md_table(
            ["阶段", "adj_ms", "adj_%"],
            epd_adj_rows,
        ),
        "",
        "| 覆盖统计 | 值 |",
        "| --- | --- |",
        f"| gen_requests | {epd.n_requests} |",
        f"| first_encode | {epd.n_requests_first_encode} |",
        f"| no_vision_row | {epd.n_requests_no_vision_row} |",
        f"| dup_encode_skipped | {epd.n_requests_encode_skipped_dup} |",
        "",
    ]
    lines.extend(PAPER_ALIGNED_VIEW_INTRO)

    for gkey in ("A_stage_execution", "B_per_request_e2e", "C_encode_raw", "D_decode_stall"):
        if gkey not in paper_by_group:
            continue
        lines.append(f"### {PAPER_ALIGNED_GROUP_TITLES.get(gkey, gkey)}")
        lines.append("")
        lines.append(
            _md_table(
                ["metric", "value_ms", "pct", "denominator", "note"],
                paper_by_group[gkey],
            )
        )
        lines.append("")

    lines.extend(
        [
            "## 6. Decode Stall（按 replica）",
            "",
            _md_table(
                ["pid", "n_decode", "span_ms", "active_ms", "idle_ms", "extend_in_decode"],
                stall_rows,
            ),
            "",
            "## 7. 自动解读（供汇报参考）",
            "",
        ]
    )
    if bullets:
        lines.extend(bullets)
    else:
        lines.append("- （无自动规则命中；请结合上表自行解读。）")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "复现命令：`python examples/profile/analyze_profiling_logs.py "
        f"--log-dir {os.path.dirname(bundle.gen_path)} --suffix {bundle.label} --report`"
    )

    os.makedirs(os.path.dirname(os.path.abspath(report_md)), exist_ok=True)
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_module_breakdown(
    gen_path: str,
    vis_path: Optional[str],
    mf_path: Optional[str],
    breakdown_csv: str,
    summary_csv: Optional[str] = None,
    report_md: Optional[str] = None,
    label: Optional[str] = None,
    drop_cold_step: bool = True,
) -> ProfilingAnalysisBundle:
    rows, epd, vision, gen, stall, paper_rows, cold_info = run_breakdown_analysis(
        gen_path, vis_path, mf_path, drop_cold_step=drop_cold_step
    )
    write_breakdown_csv(rows, breakdown_csv)
    summary_path = summary_csv or (
        breakdown_csv.rsplit(".", 1)[0] + "_summary.csv"
        if "." in breakdown_csv
        else breakdown_csv + "_summary"
    )
    write_breakdown_summary_csv(rows, summary_path)
    paper_path = (
        breakdown_csv.rsplit(".", 1)[0] + "_paper_aligned.csv"
        if "." in breakdown_csv
        else breakdown_csv + "_paper_aligned"
    )
    write_paper_aligned_csv(paper_rows, paper_path)

    bundle_label = label or os.path.basename(breakdown_csv).replace(
        "e2e_module_breakdown_", ""
    ).replace(".csv", "")
    bundle = ProfilingAnalysisBundle(
        label=bundle_label,
        gen_path=gen_path,
        vis_path=vis_path or "(none — text-only / no vision_encoder_log)",
        mf_path=mf_path,
        breakdown_csv=breakdown_csv,
        summary_csv=summary_path,
        paper_csv=paper_path,
        rows=rows,
        epd=epd,
        vision=vision,
        gen=gen,
        stall=stall,
        paper_rows=paper_rows,
        cold_info=cold_info,
    )

    print(f"\nWrote per-request breakdown: {breakdown_csv}")
    print(f"Wrote summary:               {summary_path}")
    print(f"Wrote paper-aligned view:    {paper_path}")
    print_paper_aligned_table(os.path.basename(breakdown_csv), paper_rows, stall)

    if report_md:
        write_profiling_report_md(bundle, report_md)
        print(f"Wrote Markdown report:       {report_md}")

    return bundle


def default_report_paths(log_dir: str, suffix: str) -> tuple[str, str]:
    """Return (breakdown_csv, report_md) under log_dir for a suffix."""
    base = os.path.join(log_dir, f"e2e_module_breakdown_{suffix}")
    return f"{base}.csv", os.path.join(log_dir, f"profiling_report_{suffix}.md")


def print_model_forward(mf_path: Optional[str]) -> None:
    if not mf_path or not os.path.exists(mf_path):
        return
    rows = load_csv(mf_path)
    print(f"\nMODEL FORWARD  ({os.path.basename(mf_path)})")
    by_mode: dict[str, list[float]] = defaultdict(list)
    by_step_mode: dict[tuple[int, str], list[float]] = defaultdict(list)
    for r in rows:
        t = float(r.get("forward_time_ms") or 0)
        mode = r.get("mode") or "?"
        step = int(float(r.get("global_step") or 0))
        by_mode[mode].append(t)
        by_step_mode[(step, mode)].append(t)
    for mode in ("EXTEND", "DECODE"):
        if mode in by_mode:
            print(f"  {mode:8s}  {stats(by_mode[mode])}")
    other = [m for m in by_mode if m not in ("EXTEND", "DECODE")]
    for m in other:
        print(f"  {m:8s}  {stats(by_mode[m])}")
    if len({s for s, _ in by_step_mode}) > 1:
        print("  by step:")
        for (step, mode) in sorted(by_step_mode.keys()):
            print(f"    step={step} {mode:8s}  {stats(by_step_mode[(step, mode)])}")


def resolve_log_paths(
    log_dir: Optional[str],
    suffix: str,
    gen_path: Optional[str],
    vis_path: Optional[str],
    mf_path: Optional[str],
) -> tuple[str, Optional[str], Optional[str]]:
    g = gen_path
    v = vis_path
    m = mf_path
    if log_dir:
        g = g or find_log(log_dir, "verl_sglang_generate_log", suffix)
        v = v or find_log(log_dir, "vision_encoder_log", suffix)
        m = m or find_log(log_dir, "model_forward_log", suffix)
        # Newer local SGLang profiling patches may write model-forward rows
        # under inference_step_log_* and omit the separate vision_encoder_log.
        m = m or find_log(log_dir, "inference_step_log", suffix)
    if not g:
        raise SystemExit(f"Missing generate log for suffix={suffix!r} under log_dir={log_dir!r}")
    return g, v, m


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze VLM profiling CSV logs")
    parser.add_argument("--log-dir", default=None, help="Directory containing profiling CSVs")
    parser.add_argument("--suffix", default=None, help="SGLANG_INFERENCE_LOG_SUFFIX value")
    parser.add_argument(
        "--compare-suffixes",
        default=None,
        help="Comma-separated suffixes; print EPD compare table (e.g. stress_1img,stress_2img)",
    )
    parser.add_argument("--generate-log", default=None)
    parser.add_argument("--vision-log", default=None)
    parser.add_argument("--model-forward-log", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--epd-only",
        action="store_true",
        help="Print only EPD breakdown + compare table (skip request table)",
    )
    parser.add_argument(
        "--export-breakdown-csv",
        default=None,
        metavar="PATH",
        help="Write per-request pipeline module breakdown CSV (+ *_summary.csv)",
    )
    parser.add_argument(
        "--export-breakdown-summary",
        default=None,
        metavar="PATH",
        help="Summary CSV path (default: <breakdown-csv>_summary.csv)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="One-shot: export all CSVs + write Markdown report (needs --log-dir + --suffix)",
    )
    parser.add_argument(
        "--keep-cold-step",
        action="store_true",
        help="Do NOT drop cold-start (lowest global_step, or first EXTEND per replica "
        "for single-step logs). By default cold-start is removed.",
    )
    parser.add_argument(
        "--report-md",
        default=None,
        metavar="PATH",
        help="Markdown report path (with --report: default log_dir/profiling_report_{suffix}.md)",
    )
    args = parser.parse_args()

    if args.compare_suffixes:
        if not args.log_dir:
            raise SystemExit("--compare-suffixes requires --log-dir")
        compare_rows: list[tuple[str, EpdBreakdown]] = []
        for suf in [s.strip() for s in args.compare_suffixes.split(",") if s.strip()]:
            g, v, m = resolve_log_paths(args.log_dir, suf, None, None, None)
            gen = load_csv(g)
            vis = load_csv(v)
            epd = compute_epd_breakdown(gen, vis, m)
            compare_rows.append((suf, epd))
            if args.suffix == suf or (not args.suffix and len(compare_rows) == 1):
                label = args.label or suf
                if not args.epd_only:
                    analyze_one(g, v, m, label, verbose=True)
                else:
                    print_epd_table(label, epd)
                    print_epd_adjusted_table(label, epd)
        print_epd_compare_table(compare_rows)
        return

    gen_path = args.generate_log
    vis_path = args.vision_log
    mf_path = args.model_forward_log

    if args.log_dir and args.suffix:
        gen_path, vis_path, mf_path = resolve_log_paths(
            args.log_dir, args.suffix, gen_path, vis_path, mf_path
        )

    if not gen_path:
        raise SystemExit("Need --generate-log (or --log-dir + --suffix)")

    label = args.label or args.suffix or os.path.basename(gen_path)

    # One-shot report mode: export CSVs + profiling_report_{suffix}.md
    if args.report:
        if not vis_path:
            print("NOTE: vision_encoder_log missing — text-only / P+D only breakdown (E=0).")
        if args.log_dir and args.suffix:
            breakdown_csv, report_md = default_report_paths(args.log_dir, args.suffix)
        else:
            if not args.export_breakdown_csv:
                raise SystemExit(
                    "--report without --log-dir/--suffix needs --export-breakdown-csv"
                )
            breakdown_csv = args.export_breakdown_csv
            report_md = args.report_md or (
                breakdown_csv.rsplit(".", 1)[0] + "_report.md"
                if "." in breakdown_csv
                else breakdown_csv + "_report.md"
            )
        if args.export_breakdown_csv:
            breakdown_csv = args.export_breakdown_csv
        if args.report_md:
            report_md = args.report_md
        export_module_breakdown(
            gen_path,
            vis_path,
            mf_path,
            breakdown_csv,
            args.export_breakdown_summary,
            report_md=report_md,
            label=label,
            drop_cold_step=not args.keep_cold_step,
        )
        print(f"\nDone. Open report: {report_md}")
        return

    if not vis_path:
        gen = load_csv(gen_path)
        print_generate_only_summary(label, gen)
        print("\nVISION ENCODER LOG: missing; skipping E/P/D vision-cache breakdown.")
        if mf_path:
            print_model_forward(mf_path)
        else:
            print("MODEL FORWARD / INFERENCE STEP LOG: missing")
        return

    if args.export_breakdown_csv:
        bundle = export_module_breakdown(
            gen_path,
            vis_path,
            mf_path,
            args.export_breakdown_csv,
            args.export_breakdown_summary,
            report_md=args.report_md,
            label=label,
            drop_cold_step=not args.keep_cold_step,
        )
        print_module_breakdown_table(label, bundle.rows)
        if not args.epd_only:
            analyze_one(gen_path, vis_path, mf_path, label, verbose=True)
        else:
            print_epd_table(label, bundle.epd)
            print_epd_adjusted_table(label, bundle.epd)
        return

    if args.epd_only:
        gen = load_csv(gen_path)
        vis = load_csv(vis_path)
        epd = compute_epd_breakdown(gen, vis, mf_path)
        print_epd_table(label, epd)
        print_epd_adjusted_table(label, epd)
        return

    analyze_one(gen_path, vis_path, mf_path, label, verbose=True)


if __name__ == "__main__":
    main()
