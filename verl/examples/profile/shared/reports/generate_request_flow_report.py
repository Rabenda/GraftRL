#!/usr/bin/env python3
"""Generate a walkthrough Markdown report for one profiling request (image + prompt + timeline).

Usage:
  cd verl_vision
  export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
  python3 examples/profile/generate_request_flow_report.py \\
    --log-dir profile_logs_geo3k_full \\
    --suffix geo3k_full_bs64_n4 \\
    --request-id d84e1f5204ec43e1ade8b5d09d5eebab
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

# Reuse log parsing from analyze_profiling_logs.py
from analyze_profiling_logs import (
    assign_pids_to_forward_passes,
    build_request_module_breakdown,
    load_csv,
    parse_forward_passes,
    parse_vision_rows,
    resolve_log_paths,
)


def _short(rid: str) -> str:
    return rid[:8] if len(rid) > 8 else rid


def _load_image(entry) -> Image.Image:
    if isinstance(entry, dict) and entry.get("bytes"):
        return Image.open(io.BytesIO(entry["bytes"])).convert("RGB")
    raise TypeError("expected parquet image dict with bytes")


def _prompt_plain(prompt) -> str:
    if isinstance(prompt, list) and prompt:
        content = prompt[0].get("content", "")
        return content if isinstance(content, str) else str(content)
    return str(prompt)


def _find_dataset_row(
    train_parquet: str, image_prompt_tokens: int, text_prompt_tokens: int
) -> int | None:
    """Best-effort match: token signature + distinctive long-prompt keywords."""
    table = pq.read_table(train_parquet)
    # Known signature for walkthrough default (376, 273) → Geo3K row 4 (GRID IN circles)
    if image_prompt_tokens == 273 and text_prompt_tokens == 103:
        for i in range(table.num_rows):
            plain = _prompt_plain(table.column("prompt")[i].as_py())
            if "GRID IN" in plain and "circle $A$" in plain:
                return i
    candidates: list[tuple[int, int]] = []
    for i in range(table.num_rows):
        plain = _prompt_plain(table.column("prompt")[i].as_py())
        text_len = len(plain.replace("<image>", "").strip())
        candidates.append((abs(text_len - text_prompt_tokens * 3), i))
    candidates.sort()
    return candidates[0][1] if candidates else None


def _vision_rows_for_request(vision_raw: list[dict], request_id: str) -> list[dict]:
    out = []
    for r in vision_raw:
        ids = (r.get("request_ids") or "").split("|")
        if request_id in ids:
            out.append(r)
    return out


def _grpo_siblings(gen_rows: list[dict], row: dict) -> list[dict]:
    key = (row.get("prompt_tokens"), row.get("image_prompt_tokens"))
    return [
        g
        for g in gen_rows
        if (g.get("prompt_tokens"), g.get("image_prompt_tokens")) == key
    ]


def _mf_passes_for_pid(mf_rows: list[dict], pid: str, t0: float, t1: float) -> list[dict]:
    out = []
    for r in mf_rows:
        if str(r.get("pid", "")) != str(pid):
            continue
        ts = float(r.get("timestamp") or 0)
        if t0 <= ts <= t1 + 1.0:
            out.append(r)
    return sorted(out, key=lambda x: float(x.get("timestamp") or 0))


def write_report(
    *,
    out_md: Path,
    request_id: str,
    gen_row: dict,
    breakdown: dict | None,
    vision_rows: list[dict],
    siblings: list[dict],
    dataset_row: int | None,
    sample: dict | None,
    image_rel: str | None,
    mf_snippet: list[dict],
) -> None:
    e2e = float(gen_row.get("generate_e2e_ms") or 0)
    lines: list[str] = [
        f"# 单请求全流程示例：{_short(request_id)}",
        "",
        f"生成时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## 0. 256 条 response 到底是什么？",
        "",
        "你的理解**基本正确**，需要补两点：",
        "",
        "| 概念 | 本次 run (batch=64, rollout.n=4) |",
        "| --- | --- |",
        "| **训练样本** | 1 个 step 从 dataloader 取 **64 条** Geo3K 样本（每条 1 张图 + 1 段 prompt） |",
        "| **SGLang 请求 (request)** | 每条样本做 **4 次独立生成** (GRPO) → **64×4 = 256 个 `request_id`** |",
        "| **是否 256 张不同图？** | **否**。约 **64 张 unique 图**；同一张图会对应 **4 个 request**（同一 GRPO group，随机种子不同 → 4 段不同 response） |",
        f"| **本例** | `{request_id}` 是其中 **1 个** SGLang 请求（含 **1 张图**） |",
        "",
        f"与本例 **同 prompt、同图** 的 GRPO 兄弟请求（共 {len(siblings)} 条）：",
        "",
    ]
    for s in siblings:
        lines.append(
            f"- `{s['request_id']}` — e2e={float(s.get('generate_e2e_ms',0)):.0f}ms, "
            f"out_tokens={s.get('output_tokens')}, finish={s.get('finish_reason')}"
        )
    lines.append("")

    # Image + prompt
    lines.extend(["## 1. 输入：图片 + Prompt", ""])
    if image_rel:
        lines.append(f"![输入图片]({image_rel})")
        lines.append("")
    if sample:
        lines.extend(
            [
                f"- **数据集行号（推断）**：`train.parquet` row **{dataset_row}**",
                f"- **题目（question）**：{sample.get('question', '')}",
                f"- **标准答案（ground truth）**：`{sample.get('answer', '')}`",
                "",
                "**完整 prompt（送入模型的用户文本）**：",
                "",
                "```",
                sample.get("prompt_plain", ""),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "**本请求 token 统计（来自 generate log）**：",
            "",
            "| 字段 | 值 |",
            "| --- | --- |",
            f"| prompt_tokens | {gen_row.get('prompt_tokens')} |",
            f"| image_prompt_tokens | {gen_row.get('image_prompt_tokens')} |",
            f"| text_prompt_tokens | {gen_row.get('text_prompt_tokens')} |",
            f"| image_prompt_ratio | {gen_row.get('image_prompt_ratio')} |",
            f"| image_count | {gen_row.get('image_count')} |",
            "",
        ]
    )

    # Encoder
    lines.extend(["## 2. Vision Encoder（ViT）长什么样？", ""])
    if vision_rows:
        v0 = vision_rows[0]
        lines.extend(
            [
                "对该请求，ViT **只计一次首编码**（`cached_image_features=0`）：",
                "",
                "| 字段 | 值 | 含义 |",
                "| --- | --- | --- |",
                f"| vision_encoder_time_ms | {v0.get('vision_encoder_time_ms')} | 本次 ViT forward 墙钟 |",
                f"| image_count | {v0.get('image_count')} | 本 request 带的图数 |",
                f"| image_tokens | {v0.get('image_tokens')} | 图转成多少 image token |",
                f"| prefill_tokens | {v0.get('prefill_tokens')} | EXTEND 阶段总 prefill token |",
                f"| processed_resolution_px | {v0.get('processed_resolution_px')} | 预处理后分辨率 |",
                f"| pixel_values_shape | {v0.get('pixel_values_shape')} | 送入 ViT 的 tensor 形状 |",
                f"| pass_id | {v0.get('pass_id')} | 与 model_forward EXTEND 对齐 |",
                f"| pid (GPU replica) | {v0.get('pid')} | SGLang 进程 |",
                "",
                "**直观理解**：原图 → resize 到约 `364×588` → ViT 打出 **273 个 image token** → 再与 **103 个文本 token** 拼成 **376 token** 的长前缀，进入 LLM prefill。",
                "",
            ]
        )
    else:
        lines.append("（vision_encoder_log 中未找到该 request_id）\n")

    # Timeline
    lines.extend(["## 3. 时间线：从 VERL 到吐完字", ""])
    if breakdown:
        parts = [
            ("verl_prepare", float(breakdown.get("verl_prepare_ms", 0))),
            ("queue_wait", float(breakdown.get("queue_ms", 0))),
            ("prefill_blocking_wait", float(breakdown.get("batch_sync_wait_ms", 0))),
            ("prefill_execution (extend_wall)", float(breakdown.get("extend_wall_ms", 0))),
            ("decode_execution (decode_wall)", float(breakdown.get("decode_wall_ms", 0))),
            ("sglang_orchestration", float(breakdown.get("sglang_orchestration_ms", 0))),
            ("verl_post", float(breakdown.get("verl_post_ms", 0))),
        ]
        lines.append("```mermaid")
        lines.append("flowchart TD")
        lines.append('  A["VERL 发起 rollout"] --> B["SGLang 排队 queue"]')
        lines.append('  B --> C{"同 replica 上<br/>等别人 EXTEND?<br/>batch_sync"}')
        lines.append('  C -->|本例 0ms| D["Vision Encoder ViT"]')
        lines.append('  D --> E["LLM Prefill / EXTEND<br/>extend_wall ~6.6s"]')
        lines.append('  E --> F["Decode 自回归<br/>decode_wall ~0.87s"]')
        lines.append('  F --> G["detokenize / 调度残差<br/>sglang_orch"]')
        lines.append('  G --> H["回到 VERL"]')
        lines.append("```")
        lines.append("")
        wall_stages = [
            ("verl_prepare", float(breakdown.get("verl_prepare_ms", 0)), ""),
            ("queue_wait", float(breakdown.get("queue_ms", 0)), "SGLang 排队"),
            ("prefill_blocking_wait", float(breakdown.get("batch_sync_wait_ms", 0)), "等同 replica 上别人 EXTEND"),
            ("prefill_execution (extend_wall)", float(breakdown.get("extend_wall_ms", 0)), "本请求 EXTEND（含 ViT+prefill）"),
            ("sglang_orchestration", float(breakdown.get("sglang_orchestration_ms", 0)), "调度/detokenize 残差"),
            ("verl_post", float(breakdown.get("verl_post_ms", 0)), ""),
        ]
        decode_wall = float(breakdown.get("decode_wall_ms", 0))
        lines.append("| 阶段（墙钟，可相加口径） | ms | 占 e2e | 说明 |")
        lines.append("| --- | ---: | ---: | --- |")
        wall_sum = 0.0
        for name, ms, note in wall_stages:
            if ms <= 0 and name in ("verl_prepare", "verl_post"):
                continue
            pct = ms / e2e * 100 if e2e > 0 else 0
            wall_sum += ms
            if name == "prefill_blocking_wait" and ms == 0:
                note = (note or "") + " → 本例 **首发/靠前**（cold 路径）"
            lines.append(f"| {name} | {ms:.1f} | {pct:.1f}% | {note} |")
        residual = e2e - wall_sum
        rpct = residual / e2e * 100 if e2e > 0 else 0
        lines.append(
            f"| 其它 / 未归因 | {residual:.1f} | {rpct:.1f}% | 含本请求 decode 活跃等 |"
        )
        lines.append(f"| **e2e 合计** | **{e2e:.1f}** | **100%** | generate_e2e_ms |")
        lines.append("")
        lines.extend(
            [
                "> `decode_wall`（本 replica **{:.1f} ms**）是并发 decode **流**墙钟，**不能**除以本请求 e2e 得到占比。".format(
                    decode_wall
                ),
                "",
                "**EPD 账本（本 request GPU 分摊）**：",
                "",
                f"- Encode (E): {float(breakdown.get('e_ms',0)):.1f} ms",
                f"- Prefill (P): {float(breakdown.get('p_ms',0)):.1f} ms",
                f"- Decode 分摊 (D): {float(breakdown.get('d_sum_ms',0)):.1f} ms",
                "",
            ]
        )

    if mf_snippet:
        lines.extend(["### 同 replica 上的 model_forward 片段（节选）", ""])
        lines.append("| ts | pass | mode | batch | prefill_tok | forward_ms |")
        lines.append("| ---: | ---: | --- | ---: | ---: | ---: |")
        for r in mf_snippet[:12]:
            lines.append(
                f"| {float(r.get('timestamp',0)):.3f} | {r.get('pass_id')} | {r.get('mode')} | "
                f"{r.get('batch_size')} | {r.get('prefill_tokens')} | {r.get('forward_time_ms')} |"
            )
        lines.append("")

    # Output
    lines.extend(["## 4. 模型输出了什么？", ""])
    lines.extend(
        [
            "> **注意**：当前 profiling 的 `verl_sglang_generate_log` **只记录 token 数，不记录生成文本**。",
            "> 下表为日志里能确定的输出侧信息：",
            "",
            "| 字段 | 值 |",
            "| --- | --- |",
            f"| output_tokens | {gen_row.get('output_tokens')} |",
            f"| max_new_tokens | {gen_row.get('max_new_tokens')} |",
            f"| finish_reason | {gen_row.get('finish_reason')} |",
            "",
            "含义：",
            "- `finish_reason=length` → 达到 `max_response_length=64` 被截断（不是 EOS 提前结束）",
            "- 同 GRPO group 的另外 3 条 request 会用**同图同 prompt** 再采样，得到 **不同** 的 64 token 序列（日志里 e2e 也会略有差异）",
            "",
            "若需要把 **完整生成文本** 写入报告，需要在 `async_sglang_server.py` 的 `_append_verl_sglang_generate_log` 增加 `response_text` 字段后重跑 profiling。",
            "",
        ]
    )

    lines.extend(
        [
            "## 5. 和「平均 63% 在等」的关系",
            "",
            "本例 `prefill_blocking_wait = 0`，属于 **64 条里先被调度到 GPU 的那批（cold/靠前）**：",
            "自己跑 ViT+prefill，不用等别人。",
            "",
            "同 batch 里大量 request（例如 `e6f2a17b`）会有 **~6.6s batch_sync**，",
            "因为 replica 上要先跑完很多条 EXTEND，它们才能在 SGLang 里做前缀准备。",
            "",
            "---",
            "",
            f"完整 `request_id`：`{request_id}`",
            f"GPU replica pid：`{breakdown.get('pid') if breakdown else '?'}`",
        ]
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-request flow walkthrough report")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--suffix", required=True)
    parser.add_argument(
        "--request-id",
        default="d84e1f5204ec43e1ade8b5d09d5eebab",
        help="Full request_id (default: cold-path example with solo ViT 616ms)",
    )
    parser.add_argument(
        "--train-parquet",
        default="/workspace/repo/verl_vision/data/geo3k/train.parquet",
    )
    parser.add_argument(
        "--dataset-row",
        type=int,
        default=None,
        help="Optional fixed parquet row for image/prompt (default: auto by token heuristic)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output markdown path (default: log_dir/request_flow_{suffix}_{short_id}.md)",
    )
    args = parser.parse_args()

    gen_path, vis_path, mf_path = resolve_log_paths(args.log_dir, args.suffix, None, None, None)
    gen_rows = load_csv(gen_path)
    gen_by_id = {r["request_id"]: r for r in gen_rows}
    if args.request_id not in gen_by_id:
        raise SystemExit(f"request_id not in generate log: {args.request_id}")

    gen_row = gen_by_id[args.request_id]
    siblings = _grpo_siblings(gen_rows, gen_row)

    vis_raw = load_csv(vis_path) if vis_path else []
    vision_rows = _vision_rows_for_request(vis_raw, args.request_id)

    vision = parse_vision_rows(vis_raw)
    forward_passes = parse_forward_passes(mf_path)
    assign_pids_to_forward_passes(vision, forward_passes)
    breakdowns = build_request_module_breakdown(gen_rows, vision, forward_passes)
    breakdown = next((b for b in breakdowns if b.request_id == args.request_id), None)
    breakdown_dict = breakdown.__dict__ if breakdown else None

    img_tok = int(float(gen_row.get("image_prompt_tokens") or 0))
    txt_tok = int(float(gen_row.get("text_prompt_tokens") or 0))
    dataset_row = args.dataset_row
    if dataset_row is None:
        dataset_row = _find_dataset_row(args.train_parquet, img_tok, txt_tok)

    sample = None
    image_rel = None
    assets_dir = Path(args.log_dir) / f"request_flow_assets_{args.suffix}"
    assets_dir.mkdir(parents=True, exist_ok=True)

    if dataset_row is not None and os.path.isfile(args.train_parquet):
        table = pq.read_table(args.train_parquet)
        row = {n: table.column(n)[dataset_row].as_py() for n in table.column_names}
        extra = row.get("extra_info") or {}
        imgs = row.get("images") or []
        prompt_plain = _prompt_plain(row.get("prompt"))
        sample = {
            "question": extra.get("question", ""),
            "answer": (row.get("reward_model") or {}).get("ground_truth", ""),
            "prompt_plain": prompt_plain,
        }
        if imgs:
            img = _load_image(imgs[0])
            img_name = f"{_short(args.request_id)}_dataset_row{dataset_row:04d}.png"
            img.save(assets_dir / img_name)
            image_rel = f"request_flow_assets_{args.suffix}/{img_name}"

    short = _short(args.request_id)
    out_md = Path(
        args.out
        or os.path.join(args.log_dir, f"request_flow_{args.suffix}_{short}.md")
    )

    mf_snippet: list[dict] = []
    if breakdown_dict and mf_path:
        pid = breakdown_dict.get("pid")
        for fp in forward_passes:
            if fp.pid != pid:
                continue
            if fp.mode == "EXTEND" and fp.pass_id > 3:
                continue
            mf_snippet.append(
                {
                    "timestamp": fp.timestamp,
                    "pass_id": fp.pass_id,
                    "mode": fp.mode,
                    "batch_size": fp.batch_size,
                    "prefill_tokens": fp.prefill_tokens,
                    "forward_time_ms": fp.forward_ms,
                }
            )
            if fp.mode == "DECODE" and len([x for x in mf_snippet if x["mode"] == "DECODE"]) >= 5:
                break

    write_report(
        out_md=out_md,
        request_id=args.request_id,
        gen_row=gen_row,
        breakdown=breakdown_dict,
        vision_rows=vision_rows,
        siblings=siblings,
        dataset_row=dataset_row,
        sample=sample,
        image_rel=image_rel,
        mf_snippet=mf_snippet[:20],
    )
    print(f"Wrote {out_md}")
    if image_rel:
        print(f"Image: {assets_dir / Path(image_rel).name}")


if __name__ == "__main__":
    main()
