#!/usr/bin/env python3
"""Write side-by-side Markdown comparing image vs padded text-only profiling reports."""

from __future__ import annotations

import argparse
import csv
import os
import statistics as st
from datetime import datetime, timezone


def _load_generate(log_path: str) -> list[dict]:
    rows = list(csv.DictReader(open(log_path)))
    if any(r.get("has_image") == "0" for r in rows) and any(
        r.get("has_image") == "1" for r in rows
    ):
        pass
    return rows


def _stats(rows: list[dict], image_only: bool | None) -> dict:
    if image_only is True:
        rows = [r for r in rows if r.get("has_image") == "1"]
    elif image_only is False:
        rows = [r for r in rows if r.get("has_image") == "0"]
    e2e = [float(r["generate_e2e_ms"]) for r in rows]
    pt = [int(r["prompt_tokens"]) for r in rows]
    it = [int(r["image_prompt_tokens"]) for r in rows]
    tt = [int(r["text_prompt_tokens"]) for r in rows]
    out = [int(r["output_tokens"]) for r in rows]
    return {
        "n": len(rows),
        "e2e_mean": st.mean(e2e) if e2e else 0,
        "e2e_med": st.median(e2e) if e2e else 0,
        "prompt_mean": st.mean(pt) if pt else 0,
        "prompt_med": st.median(pt) if pt else 0,
        "image_tok_mean": st.mean(it) if it else 0,
        "text_tok_mean": st.mean(tt) if tt else 0,
        "out_mean": st.mean(out) if out else 0,
        "out_med": st.median(out) if out else 0,
    }


def _read_summary(path: str) -> dict[str, float]:
    if not os.path.isfile(path):
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        out[r["metric"]] = float(r["mean_ms"])
        if r.get("pct_of_e2e"):
            out[f"{r['metric']}_pct"] = float(r["pct_of_e2e"])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image-log-dir", default="profile_logs_geo3k_full")
    p.add_argument("--image-suffix", default="geo3k_full_bs64_n4")
    p.add_argument("--text-log-dir", default="profile_logs_geo3k_text_only")
    p.add_argument("--text-suffix", default="geo3k_text_only_padded_bs64_n4")
    p.add_argument("--out", default="profile_logs_geo3k_text_only/compare_image_vs_text_padded.md")
    args = p.parse_args()

    img_gen = os.path.join(
        args.image_log_dir, f"verl_sglang_generate_log_{args.image_suffix}.csv"
    )
    txt_gen = os.path.join(
        args.text_log_dir, f"verl_sglang_generate_log_{args.text_suffix}.csv"
    )
    img_sum = os.path.join(
        args.image_log_dir, f"e2e_module_breakdown_{args.image_suffix}_summary.csv"
    )
    txt_sum = os.path.join(
        args.text_log_dir, f"e2e_module_breakdown_{args.text_suffix}_summary.csv"
    )

    si = _stats(_load_generate(img_gen), image_only=True)
    st_ = _stats(_load_generate(txt_gen), image_only=False)
    bi = _read_summary(img_sum)
    bt = _read_summary(txt_sum)

    lines = [
        "# 带图 vs 垫长 text-only 对照",
        "",
        f"生成: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "**text-only 已按带图 run 的 `text_prompt_tokens → prompt_tokens` 中位数垫长**（见 `data/geo3k_text_only_padded/padding_meta.json`）。",
        "",
        "## Generate log 对比",
        "",
        "| 指标 | 带图 | text-only 垫长 |",
        "| --- | ---: | ---: |",
        f"| n | {si['n']} | {st_['n']} |",
        f"| mean e2e (ms) | {si['e2e_mean']:.0f} | {st_['e2e_mean']:.0f} |",
        f"| mean prompt_tokens | {si['prompt_mean']:.0f} | {st_['prompt_mean']:.0f} |",
        f"| median prompt_tokens | {si['prompt_med']:.0f} | {st_['prompt_med']:.0f} |",
        f"| mean image_tokens | {si['image_tok_mean']:.0f} | 0 |",
        f"| mean text_tokens | {si['text_tok_mean']:.0f} | {st_['text_tok_mean']:.0f} |",
        f"| median output_tokens | {si['out_med']:.0f} | {st_['out_med']:.0f} |",
        "",
        f"e2e 倍数 (image/text): **{si['e2e_mean'] / st_['e2e_mean']:.2f}×**"
        if st_["e2e_mean"] > 0
        else "",
        "",
        "## E2E 分解 (summary mean_ms)",
        "",
        "| 模块 | 带图 ms | 带图 % | text-only ms | text-only % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in [
        ("batch_sync_wait", "prefill_blocking_wait"),
        ("extend_wall", "prefill_execution"),
        ("decode_wall", "decode_execution"),
        ("queue", "queue"),
        ("sglang_orchestration", "orchestration"),
    ]:
        lines.append(
            f"| {label} | {bi.get(key, 0):.0f} | {bi.get(key + '_pct', 0):.1f} | "
            f"{bt.get(key, 0):.0f} | {bt.get(key + '_pct', 0):.1f} |"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
