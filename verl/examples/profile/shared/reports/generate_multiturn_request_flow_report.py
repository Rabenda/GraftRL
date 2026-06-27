#!/usr/bin/env python3
"""Walkthrough for one multiturn rollout (Refocus_Chart / dummy-crop): before/after images + timing.

Requires profiling run with PROFILE_IMAGE_DUMP_DIR set.

Usage:
  python3 examples/profile/generate_multiturn_request_flow_report.py \\
    --log-dir profile_logs_refocus_chart \\
    --suffix refocus_chart_multiturn_bs64_n4 \\
    --image-dump-dir profile_logs_refocus_chart/image_dump_refocus_chart_multiturn_bs64_n4 \\
    --request-id <agent request_id from manifest> \\
    --train-parquet /data/refocus_chart_multiturn/train.parquet

  python3 ... --list-requests
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from analyze_profiling_logs import (
    assign_pids_to_forward_passes,
    build_request_module_breakdown,
    load_csv,
    parse_forward_passes,
    parse_vision_rows,
    resolve_log_paths,
)
from generate_request_flow_report import (
    _grpo_siblings,
    _prompt_plain,
    _short,
    _vision_rows_for_request,
)


def _load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _manifest_for_request(manifest: list[dict], request_id: str) -> list[dict]:
    rid = request_id.strip()
    out = [r for r in manifest if r.get("request_id") == rid]
    if out:
        return sorted(out, key=lambda x: (int(x.get("turn", 0)), int(x.get("image_idx", 0))))
    prefix = rid[:8] if len(rid) >= 8 else rid
    out = [
        r
        for r in manifest
        if prefix in r.get("path", "") or str(r.get("request_id", "")).startswith(prefix)
    ]
    return sorted(out, key=lambda x: (int(x.get("turn", 0)), int(x.get("image_idx", 0))))


def _list_request_candidates(manifest: list[dict]) -> list[dict]:
    by_rid: dict[str, dict] = {}
    for r in manifest:
        rid = str(r.get("request_id", ""))
        if not rid:
            continue
        entry = by_rid.setdefault(
            rid,
            {"request_id": rid, "uid": r.get("uid"), "roles": set(), "turns": set(), "rollout_idx": r.get("rollout_idx")},
        )
        entry["roles"].add(r.get("role", ""))
        entry["turns"].add(int(r.get("turn", 0)))
    complete = []
    for rid, meta in by_rid.items():
        has_refocus = "refocus_output" in meta["roles"] or "crop" in meta["roles"]
        has_input = "input" in meta["roles"] or "chart_input" in meta["roles"]
        if has_input and has_refocus:
            complete.append(
                {
                    "request_id": rid,
                    "uid": meta["uid"],
                    "turns": sorted(meta["turns"]),
                    "rollout_idx": meta.get("rollout_idx"),
                }
            )
    return sorted(complete, key=lambda x: x["request_id"])


def _read_agent_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, list) and data:
        return data[0] if isinstance(data[0], dict) else {}
    return data if isinstance(data, dict) else {}


def _detect_mode(manifest_entries: list[dict], agent_cfg: dict) -> str:
    roles = {e.get("role") for e in manifest_entries}
    if "refocus_output" in roles:
        return "refocus"
    if "crop" in roles:
        return "dummy_crop"
    target = str(agent_cfg.get("_target_", ""))
    if "vtool" in target.lower():
        return "refocus"
    if "dummy_crop" in target.lower():
        return "dummy_crop"
    return "refocus"


def _user_question_from_prompt(prompt_plain: str) -> str:
    m = re.search(r"USER REQUEST:\s*(.+?)(?:\n|$)", prompt_plain, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"<image>\s*(.+)", prompt_plain)
    if m:
        return m.group(1).strip()[:300]
    return prompt_plain.replace("<image>", "").strip()[:300]


def _copy_role_images(
    *,
    manifest_entries: list[dict],
    dump_dir: Path,
    assets_dir: Path,
    request_id: str,
) -> dict[str, str]:
    """role -> markdown-relative image path."""
    assets_prefix = assets_dir.name
    out: dict[str, str] = {}
    for e in manifest_entries:
        role = str(e.get("role", ""))
        src = dump_dir / e["path"]
        if not src.is_file():
            continue
        turn = e.get("turn", 0)
        dst_name = f"{_short(request_id)}_t{turn}_{role}.png"
        shutil.copy2(src, assets_dir / dst_name)
        out[role] = f"{assets_prefix}/{dst_name}"
    return out


def _grpo_agent_rollouts(manifest: list[dict], uid: str) -> list[dict]:
    by_rid: dict[str, set[str]] = {}
    rollout_idx = None
    for r in manifest:
        if r.get("uid") != uid:
            continue
        rid = str(r.get("request_id", ""))
        if not rid:
            continue
        rollout_idx = rollout_idx or r.get("rollout_idx")
        by_rid.setdefault(rid, set()).add(str(r.get("role", "")))
    rows = []
    for rid, roles in sorted(by_rid.items()):
        rows.append(
            {
                "request_id": rid,
                "refocus_ok": "refocus_output" in roles or "crop" in roles,
            }
        )
    return rows, rollout_idx


def _closest_gen(gen_rows: list[dict], *, image_count: int, target_prompt: int) -> dict | None:
    pool = [g for g in gen_rows if int(float(g.get("image_count") or 0)) == image_count]
    if not pool:
        return None
    return min(pool, key=lambda g: abs(int(float(g.get("prompt_tokens") or 0)) - target_prompt))


def _gen_by_agent_turn(gen_rows: list[dict], agent_request_id: str, agent_turn: int) -> dict | None:
    """Exact match when profiling logs agent_request_id + agent_turn (or {rid}_t{n} request_id)."""
    turn_s = str(agent_turn)
    expected_rid = f"{agent_request_id}_t{agent_turn}"
    for g in gen_rows:
        if str(g.get("agent_request_id") or "") == agent_request_id and str(g.get("agent_turn")) == turn_s:
            return g
        if str(g.get("request_id") or "") == expected_rid:
            return g
    return None


def _measure_prompt_tokens(
    *,
    train_parquet: str,
    dataset_row: int,
    dump_dir: Path,
    manifest_entries: list[dict],
    num_images: int,
) -> int | None:
    """Processor token count for this parquet row (+ optional refocus PNG from dump)."""
    try:
        from PIL import Image
        from transformers import AutoProcessor

        from verl.utils.tokenizer import build_multimodal_processor_inputs
    except ImportError:
        return None

    if not os.path.isfile(train_parquet):
        return None
    table = pq.read_table(train_parquet)
    if dataset_row < 0 or dataset_row >= table.num_rows:
        return None
    row = {n: table.column(n)[dataset_row].as_py() for n in table.column_names}
    messages = row.get("prompt")
    if not messages:
        return None

    pil_images: list[Image.Image] = []
    chart_entry = next((e for e in manifest_entries if e.get("role") in ("chart_input", "input")), None)
    if chart_entry and (dump_dir / chart_entry["path"]).is_file():
        pil_images.append(Image.open(dump_dir / chart_entry["path"]).convert("RGB"))
    elif row.get("images"):
        img0 = row["images"][0]
        if isinstance(img0, dict) and img0.get("bytes"):
            pil_images.append(Image.open(io.BytesIO(img0["bytes"])).convert("RGB"))

    if num_images >= 2:
        ref_entry = next((e for e in manifest_entries if e.get("role") in ("refocus_output", "crop")), None)
        if ref_entry and (dump_dir / ref_entry["path"]).is_file():
            pil_images.append(Image.open(dump_dir / ref_entry["path"]).convert("RGB"))

    if not pil_images:
        return None

    proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
    raw = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = build_multimodal_processor_inputs(
        proc,
        text=[raw],
        images=pil_images if num_images >= 2 else pil_images[:1],
    )
    return len(inputs["input_ids"][0])


def _estimate_turn2_prompt_tokens(t1_target: int, gen_rows: list[dict]) -> int:
    """Turn-2 prompt = turn-1 context + observation image; estimate via run-level Δ median."""
    deltas: list[int] = []
    for g2 in gen_rows:
        if int(float(g2.get("image_count") or 0)) != 2:
            continue
        pt2 = int(float(g2["prompt_tokens"]))
        g1 = _closest_gen(gen_rows, image_count=1, target_prompt=pt2 - 700)
        if g1:
            deltas.append(pt2 - int(float(g1["prompt_tokens"])))
    if deltas:
        return t1_target + int(sorted(deltas)[len(deltas) // 2])
    return t1_target + 700


def _match_sglang_turns(
    gen_rows: list[dict],
    *,
    agent_request_id: str | None,
    t1_target: int | None,
    t2_target: int | None,
) -> tuple[dict | None, dict | None, dict[str, int]]:
    """Prefer agent_request_id+turn; fall back to prompt_tokens nearest-neighbor."""
    meta: dict[str, int] = {}
    gen_turn1 = gen_turn2 = None
    match_mode = "token_nearest"
    if agent_request_id:
        gen_turn1 = _gen_by_agent_turn(gen_rows, agent_request_id, 0)
        gen_turn2 = _gen_by_agent_turn(gen_rows, agent_request_id, 1)
        if gen_turn1 or gen_turn2:
            match_mode = "agent_id"
    if gen_turn1 is None and t1_target:
        gen_turn1 = _closest_gen(gen_rows, image_count=1, target_prompt=t1_target)
    if gen_turn2 is None and t2_target:
        gen_turn2 = _closest_gen(gen_rows, image_count=2, target_prompt=t2_target)
    meta["match_mode"] = match_mode
    if gen_turn1 and t1_target is not None:
        meta["t1_target"] = t1_target
        meta["t1_matched"] = int(float(gen_turn1["prompt_tokens"]))
        meta["t1_delta"] = meta["t1_matched"] - t1_target
    if gen_turn2 and t2_target is not None:
        meta["t2_target"] = t2_target
        meta["t2_matched"] = int(float(gen_turn2["prompt_tokens"]))
        meta["t2_delta"] = meta["t2_matched"] - t2_target
    return gen_turn1, gen_turn2, meta


def _breakdown_for(gen_rows: list[dict], vis_path: str | None, mf_path: str | None, gen_row: dict | None) -> dict | None:
    if not gen_row or not vis_path or not mf_path:
        return None
    vis_raw = load_csv(vis_path)
    vision = parse_vision_rows(vis_raw)
    forward_passes = parse_forward_passes(mf_path)
    assign_pids_to_forward_passes(vision, forward_passes)
    breakdowns = build_request_module_breakdown(gen_rows, vision, forward_passes)
    rid = gen_row["request_id"]
    b = next((x for x in breakdowns if x.request_id == rid), None)
    return b.__dict__ if b else None


def _format_gen_table(gen_row: dict | None, label: str) -> list[str]:
    if not gen_row:
        return [f"> 未在 SGLang generate log 中匹配到 **{label}**（见下文说明）。", ""]
    e2e = float(gen_row.get("generate_e2e_ms") or 0)
    return [
        f"### {label}",
        "",
        f"SGLang `request_id`：`{gen_row.get('request_id')}`",
        "",
        "| 字段 | 值 |",
        "| --- | --- |",
        f"| generate_e2e_ms | {e2e:.1f} |",
        f"| prompt_tokens | {gen_row.get('prompt_tokens')} |",
        f"| image_prompt_tokens | {gen_row.get('image_prompt_tokens')} |",
        f"| text_prompt_tokens | {gen_row.get('text_prompt_tokens')} |",
        f"| image_count | {gen_row.get('image_count')} |",
        f"| output_tokens | {gen_row.get('output_tokens')} |",
        f"| finish_reason | {gen_row.get('finish_reason')} |",
        "",
    ]


def _pct_of_e2e(ms: float, e2e: float) -> str:
    """Only show % when the metric is a sub-interval of this request's e2e wall clock."""
    if e2e <= 0:
        return "—"
    if ms > e2e * 1.02:
        return "—（与 e2e 重叠计量）"
    return f"{ms / e2e * 100:.1f}%"


def _format_breakdown_table(bd: dict | None, gen_row: dict | None, e2e: float) -> list[str]:
    """Per-request e2e split: queue + prefill_launch_latency + decode residual (no double-count)."""
    if not bd:
        return []
    queue_ms = float((gen_row or {}).get("queue_time_ms") or bd.get("queue_ms") or 0)
    prefill_lat = float((gen_row or {}).get("prefill_launch_latency_ms") or 0)
    extend_wall = float(bd.get("extend_wall_ms", 0))
    batch_sync = float(bd.get("batch_sync_wait_ms", 0))
    decode_wall = float(bd.get("decode_wall_ms", 0))
    d_sum = float(bd.get("d_sum_ms", 0))
    e_ms = float(bd.get("e_ms", 0))
    p_ms = float(bd.get("p_ms", 0))

    remainder = max(0.0, e2e - queue_ms - prefill_lat)

    lines = [
        "| 阶段（**本请求**墙钟，互斥分项） | ms | 占 e2e | 说明 |",
        "| --- | ---: | ---: | --- |",
        f"| queue_wait | {queue_ms:.1f} | {_pct_of_e2e(queue_ms, e2e)} | `queue_time_ms` |",
        (
            f"| prefill（EXTEND 墙钟） | {prefill_lat:.1f} | {_pct_of_e2e(prefill_lat, e2e)} | "
            f"`prefill_launch_latency_ms`（含 ViT+LLM prefill） |"
        ),
        (
            f"| post-prefill 残差 | {remainder:.1f} | {_pct_of_e2e(remainder, e2e)} | "
            f"e2e − queue − prefill；含 decode 等待/执行、scheduler、detokenize、HTTP/异步编排；"
            f"**勿**再叠加 `extend_wall` |"
        ),
        f"| **e2e 合计** | **{e2e:.1f}** | **100%** | `generate_e2e_ms` |",
        "",
        f"> 参考：`extend_wall`={extend_wall:.1f} ms（GPU EXTEND 账本，与上表 prefill 行口径不同）。",
        "",
        "**Replica 干扰（coupled batch；勿去除以 e2e）**：",
        "",
        "| 指标 | ms | 说明 |",
        "| --- | ---: | --- |",
        f"| prefill_blocking_wait | {batch_sync:.1f} | 等同 replica 上 **等别人 EXTEND**；可 > 本请求 e2e |",
        f"| decode_wall（replica 流） | {decode_wall:.1f} | 同 pid 并发 decode 流墙钟；可 ≫ 本请求 e2e |",
        "",
        "**EPD 分摊（GPU forward 账本，分母≠ e2e）**：",
        "",
        f"- Encode (E): {e_ms:.1f} ms · Prefill (P): {p_ms:.1f} ms · Decode (D): {d_sum:.1f} ms",
        "",
    ]
    return lines


def write_multiturn_report(
    *,
    out_md: Path,
    request_id: str,
    gen_rows: list[dict],
    manifest_entries: list[dict],
    full_manifest: list[dict],
    dump_dir: Path,
    assets_dir: Path,
    dataset_row: int | None,
    sample: dict | None,
    agent_cfg: dict,
    gen_turn1: dict | None,
    gen_turn2: dict | None,
    bd_turn1: dict | None,
    bd_turn2: dict | None,
    vision_turn2: list[dict],
    sglang_match: dict[str, int | str],
) -> None:
    uid = manifest_entries[0].get("uid", "?") if manifest_entries else "?"
    mode = _detect_mode(manifest_entries, agent_cfg)
    images = _copy_role_images(
        manifest_entries=manifest_entries,
        dump_dir=dump_dir,
        assets_dir=assets_dir,
        request_id=request_id,
    )
    chart_rel = images.get("chart_input") or images.get("input")
    refocus_rel = images.get("refocus_output") or images.get("crop")
    grpo_rollouts, rollout_idx = _grpo_agent_rollouts(full_manifest, uid)

    lines = [
        f"# Multiturn 单请求：原图 vs Refocus 新图（{_short(request_id)}）",
        "",
        f"生成时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## 0. 这条轨迹在 batch 里是什么？",
        "",
        "| 概念 | 说明 |",
        "| --- | --- |",
        "| 训练样本 | batch=64 中的一条 Refocus_Chart（1 张 chart + 长 tool prompt） |",
        "| Agent `uid` | 同一张图 + 同题的 **GRPO 分组**（4 路采样） |",
        f"| 本 GRPO 组 `uid` | `{uid}` |",
        f"| 本报告 agent `request_id` | `{request_id}` |",
        "| SGLang `request_id` | 新 run：`{agent_request_id}_t{turn}`；旧 run 为随机 uuid（见 §4） |",
        "",
        "## 0.1 证据强度（读报告前先看）",
        "",
        "| 结论类型 | 可信度 | 依据 |",
        "| --- | --- | --- |",
        "| refocus 流程发生、第 2 枪双图 | **高** | manifest 有 `refocus_output`；generate `image_count=2` |",
        "| refocus 新图重新 ViT、`cached=0` | **高** | 第 2 枪 vision log 可按 SGLang `request_id` 精确对齐 |",
        "| 第 1 枪 e2e / queue / prefill 分解 | **中–低**（旧 log） | 无 `agent_request_id` 时需 token 最近邻，Δ 大则为候选 |",
        "| 第 1 枪耗时（新 log） | **高** | generate log 含 `agent_request_id` + `agent_turn` 精确匹配 |",
        "| `batch_sync` / `decode_wall` 因果 | **低（单条）** | replica 级并发指标，可远大于本请求 e2e |",
        "",
        f"同 `uid` 下 **4 路 GRPO** refocus 是否成功：",
        "",
        "| agent request_id | refocus 出图 |",
        "| --- | --- |",
    ]
    for g in grpo_rollouts:
        mark = "是" if g["refocus_ok"] else "否"
        rid = g["request_id"]
        star = " **← 本报告**" if rid == request_id else ""
        lines.append(f"| `{_short(rid)}…` | {mark}{star} |")
    lines.append("")

    lines.extend(["## 1. 原图 vs Refocus 新图（先看这个）", ""])
    if chart_rel and refocus_rel:
        lines.extend(
            [
                "| **原图（输入 chart）** | **Refocus 后（tool 改图）** |",
                "| :---: | :---: |",
                f"| ![原图]({chart_rel}) | ![Refocus 新图]({refocus_rel}) |",
                "",
            ]
        )
        if mode == "refocus":
            lines.extend(
                [
                    "**发生了什么**：第 1 轮 assistant 触发 refocus Python（本 profiling 默认用数据集 **oracle** 代码），",
                    "对图表上指定类别/区域做 **mask / draw / highlight**，`display()` 得到第二张图；",
                    "第 2 轮把 **原图 + 新图** 一起送进 VLM 再生成答案。",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "**发生了什么**：`dummy_crop_agent` 对原图做 **中心裁剪**，得到第二张图后再生成。",
                    "",
                ]
            )
    else:
        lines.append("> 缺少 chart_input 或 refocus_output 的 PNG，请确认 refocus 是否执行成功。\n")

    # --- §2 Question + oracle ---
    lines.extend(["## 2. 题目与 refocus 动作", ""])
    if sample:
        lines.extend(
            [
                f"- **数据集**：`train.parquet` row **{dataset_row}**（rollout_idx=`{rollout_idx}`）",
                f"- **题目**：{sample.get('question', '')}",
                f"- **标准答案**：`{sample.get('answer', '')}`",
                "",
            ]
        )
        oracle = sample.get("oracle_refocus_code", "").strip()
        if oracle:
            lines.extend(
                [
                    "**本样本 oracle refocus 代码**（profiling 时 `VTOOL_ORACLE_REFOCUS=1` 执行）：",
                    "",
                    "```python",
                    oracle,
                    "```",
                    "",
                ]
            )
    else:
        lines.append("> 未加载 parquet 样本元数据；仅展示 image dump。\n")

    # --- §3 flow ---
    lines.extend(
        [
            "## 3. Multiturn 流程（agent 视角）",
            "",
            "```mermaid",
            "flowchart LR",
            "  A[原图 chart_input] --> B[第1轮 generate]",
            "  B --> C[执行 refocus Python]",
            "  C --> D[refocus_output 新图]",
            "  D --> E[第2轮 generate 双图上下文]",
            "  E --> F[最终答案]",
            "```",
            "",
        ]
    )

    # --- §4 SGLang timing ---
    match_mode = str(sglang_match.get("match_mode", "token_nearest"))
    match_lines = [
        "## 4. SGLang 两枪耗时",
        "",
    ]
    if match_mode == "agent_id":
        match_lines.extend(
            [
                f"> 匹配方式：**精确** — generate log 中 `agent_request_id={request_id}` + `agent_turn`",
                f">（SGLang `request_id` = `{{agent_request_id}}_t{{turn}}`）。",
                "",
            ]
        )
    else:
        match_lines.extend(
            [
                "> 匹配方式：**token 最近邻（启发式）** — 旧 run 无 `agent_request_id` 列。",
                "> 用本 parquet 行 + image dump 估计 `prompt_tokens`，在 generate log 里找最近邻；",
                "> **第 1 枪 Δ 大时，下表只能作候选，不能当严格单轨迹证据。**",
                "> 重跑 profiling（新 verl）后可用精确匹配。",
                "",
            ]
        )
    if sglang_match.get("t1_target"):
        warn1 = ""
        if match_mode != "agent_id" and abs(int(sglang_match.get("t1_delta", 0))) > 50:
            warn1 = " ⚠️ **偏差大，可能不是本条 agent 轨迹**"
        match_lines.append(
            f"- 第 1 枪："
            + (
                f"`{gen_turn1.get('request_id')}`（agent_turn=0）"
                if gen_turn1 and match_mode == "agent_id"
                else (
                    f"目标 **{sglang_match['t1_target']}** tokens → 匹配 **{sglang_match.get('t1_matched', '?')}** "
                    f"(Δ={sglang_match.get('t1_delta', '?')}){warn1}"
                )
            )
        )
    if sglang_match.get("t2_target") or gen_turn2:
        warn2 = ""
        if match_mode != "agent_id" and abs(int(sglang_match.get("t2_delta", 0))) > 80:
            warn2 = " ⚠️ **偏差大**"
        match_lines.append(
            f"- 第 2 枪："
            + (
                f"`{gen_turn2.get('request_id')}`（agent_turn=1，image_count=2）"
                if gen_turn2 and match_mode == "agent_id"
                else (
                    f"目标 **{sglang_match.get('t2_target', '?')}** tokens → 匹配 **{sglang_match.get('t2_matched', '?')}** "
                    f"(Δ={sglang_match.get('t2_delta', '?')}){warn2}"
                )
            )
        )
    match_lines.append("")
    lines.extend(match_lines)
    t1_title = "第 1 枪 · 1 张图（原 chart + tool 说明）"
    if match_mode != "agent_id" and gen_turn1 and abs(int(sglang_match.get("t1_delta", 0))) > 50:
        t1_title += "（**候选匹配，证据偏弱**）"
    lines.extend(_format_gen_table(gen_turn1, t1_title))
    if bd_turn1 and gen_turn1:
        e2e1 = float(gen_turn1.get("generate_e2e_ms") or 0)
        lines.append("**时间分解（第 1 枪）**")
        lines.append("")
        lines.extend(_format_breakdown_table(bd_turn1, gen_turn1, e2e1))

    lines.extend(_format_gen_table(gen_turn2, "第 2 枪 · 2 张图（原图 + refocus 后）"))
    if bd_turn2 and gen_turn2:
        e2e2 = float(gen_turn2.get("generate_e2e_ms") or 0)
        lines.append("**时间分解（第 2 枪）**")
        lines.append("")
        lines.extend(_format_breakdown_table(bd_turn2, gen_turn2, e2e2))

    if vision_turn2:
        lines.extend(["**第 2 枪 ViT（按 SGLang `request_id` 精确对齐）**：", ""])
        lines.append("| pass | vision_ms | log.image_count | image_grid_thw | image_tokens | cached |")
        lines.append("| ---: | ---: | ---: | --- | ---: | --- |")
        for v in vision_turn2[:8]:
            grid = v.get("image_grid_thw") or "—"
            lines.append(
                f"| {v.get('pass_id')} | {v.get('vision_encoder_time_ms')} | {v.get('image_count')} | "
                f"`{grid}` | {v.get('image_tokens')} | {v.get('cached_image_features')} |"
            )
        lines.append("")
        gen_ic = gen_turn2.get("image_count") if gen_turn2 else "?"
        lines.append(
            f"> **以 `generate_log.image_count={gen_ic}` 与 `image_grid_thw` 为准**（本样本 grid 中 `A/B` 表示两张图）。"
        )
        lines.append(
            "> `vision_encoder_log.image_count` 是 **MultimodalDataItem 条数**（多图常合并为 1 条 forward），"
            "**不能**解释成「本次 ViT 只编码 1 张图」。"
        )
        lines.append(
            "> `cached=0` + 非空 `image_grid_thw` 可证明 refocus 新图走了新 ViT encode。"
        )
        lines.append("")

    # --- §5 prompt (collapsed) ---
    if sample and sample.get("prompt_plain"):
        user_q = _user_question_from_prompt(sample["prompt_plain"])
        lines.extend(
            [
                "## 5. Prompt（仅摘要）",
                "",
                "完整 prompt 含大量 tool API 说明（数千 token），此处只保留 **USER REQUEST**：",
                "",
                "```",
                user_q,
                "```",
                "",
                f"<details><summary>展开完整 prompt（约 {len(sample['prompt_plain'])} 字符，截断展示）</summary>",
                "",
                sample["prompt_plain"][:4000].replace("```", "'''")
                + ("\n\n... (truncated)" if len(sample["prompt_plain"]) > 4000 else ""),
                "",
                "</details>",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "**换一条 request**：",
            "",
            "```bash",
            f"python3 examples/profile/generate_multiturn_request_flow_report.py \\\\",
            f"  --log-dir {out_md.parent} --suffix <suffix> \\\\",
            f"  --image-dump-dir {dump_dir} --list-requests",
            "```",
            "",
        ]
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiturn request walkthrough (refocus / dummy-crop)")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--image-dump-dir", required=True)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--list-requests", action="store_true")
    parser.add_argument(
        "--train-parquet",
        default="/data/refocus_chart_multiturn/train.parquet",
    )
    parser.add_argument(
        "--agent-config",
        default="/workspace/repo/verl_vision/examples/profile/vtool_agent_loop.yaml",
    )
    parser.add_argument("--dataset-row", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    dump_dir = Path(args.image_dump_dir)
    manifest_path = dump_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing {manifest_path}")

    manifest = _load_manifest(manifest_path)
    if args.list_requests:
        candidates = _list_request_candidates(manifest)
        print(f"Complete rollouts (input + refocus/crop): {len(candidates)}")
        for c in candidates[:50]:
            print(
                f"  {c['request_id']}  uid={str(c['uid'])[:8]}..  "
                f"row={c.get('rollout_idx')}  turns={c['turns']}"
            )
        if len(candidates) > 50:
            print(f"  ... and {len(candidates) - 50} more")
        return

    request_id = args.request_id
    if not request_id:
        candidates = _list_request_candidates(manifest)
        if not candidates:
            raise SystemExit("No complete rollouts in manifest.")
        request_id = candidates[0]["request_id"]
        print(f"No --request-id; using: {request_id}")

    manifest_entries = _manifest_for_request(manifest, request_id)
    if not manifest_entries:
        raise SystemExit(f"No manifest rows for request_id={request_id}")

    rollout_idx = manifest_entries[0].get("rollout_idx")
    dataset_row = args.dataset_row
    if dataset_row is None and rollout_idx is not None:
        try:
            dataset_row = int(rollout_idx)
        except (TypeError, ValueError):
            pass

    gen_path, vis_path, mf_path = resolve_log_paths(args.log_dir, args.suffix, None, None, None)
    gen_rows = load_csv(gen_path) if os.path.isfile(gen_path) else []

    sample = None
    if dataset_row is not None and os.path.isfile(args.train_parquet):
        table = pq.read_table(args.train_parquet)
        if 0 <= dataset_row < table.num_rows:
            row = {n: table.column(n)[dataset_row].as_py() for n in table.column_names}
            extra = row.get("extra_info") or {}
            plain = _prompt_plain(row.get("prompt"))
            sample = {
                "question": extra.get("question") or _user_question_from_prompt(plain),
                "answer": (row.get("reward_model") or {}).get("ground_truth", ""),
                "prompt_plain": plain,
                "oracle_refocus_code": extra.get("oracle_refocus_code") or "",
            }

    t1_target = t2_target = None
    if dataset_row is not None:
        t1_target = _measure_prompt_tokens(
            train_parquet=args.train_parquet,
            dataset_row=dataset_row,
            dump_dir=dump_dir,
            manifest_entries=manifest_entries,
            num_images=1,
        )
        if t1_target is not None:
            t2_target = _estimate_turn2_prompt_tokens(t1_target, gen_rows)
    gen_turn1, gen_turn2, sglang_match = _match_sglang_turns(
        gen_rows,
        agent_request_id=request_id,
        t1_target=t1_target,
        t2_target=t2_target,
    )
    if t1_target:
        print(f"Measured turn1 prompt_tokens={t1_target}, turn2={t2_target}")
    print(f"SGLang match mode: {sglang_match.get('match_mode')}")
    if gen_turn1:
        extra = "" if sglang_match.get("match_mode") == "agent_id" else f" (Δ={sglang_match.get('t1_delta')})"
        print(f"Matched SGLang turn1: {gen_turn1['request_id']}{extra}")
    if gen_turn2:
        extra = "" if sglang_match.get("match_mode") == "agent_id" else f" (Δ={sglang_match.get('t2_delta')})"
        print(f"Matched SGLang turn2: {gen_turn2['request_id']}{extra}")
    bd_turn1 = _breakdown_for(gen_rows, vis_path, mf_path, gen_turn1)
    bd_turn2 = _breakdown_for(gen_rows, vis_path, mf_path, gen_turn2)
    vision_turn2 = _vision_rows_for_request(load_csv(vis_path) if vis_path else [], gen_turn2["request_id"]) if gen_turn2 else []

    assets_dir = Path(args.log_dir) / f"request_flow_assets_{args.suffix}"
    assets_dir.mkdir(parents=True, exist_ok=True)
    agent_cfg = _read_agent_yaml(Path(args.agent_config))

    out_md = Path(
        args.out or os.path.join(args.log_dir, f"request_flow_{args.suffix}_{_short(request_id)}.md")
    )

    write_multiturn_report(
        out_md=out_md,
        request_id=request_id,
        gen_rows=gen_rows,
        manifest_entries=manifest_entries,
        full_manifest=manifest,
        dump_dir=dump_dir,
        assets_dir=assets_dir,
        dataset_row=dataset_row,
        sample=sample,
        agent_cfg=agent_cfg,
        gen_turn1=gen_turn1,
        gen_turn2=gen_turn2,
        bd_turn1=bd_turn1,
        bd_turn2=bd_turn2,
        vision_turn2=vision_turn2,
        sglang_match=sglang_match,
    )
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
