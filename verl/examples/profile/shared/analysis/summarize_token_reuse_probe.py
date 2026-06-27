#!/usr/bin/env python3
"""Summarize token-level partial ViT reuse from vision_encoder_log CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_mixed_header_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for parts in reader:
            if not parts:
                continue
            if parts[0] == "timestamp":
                header = parts
                continue
            if header is None:
                continue
            row = {key: parts[idx] if idx < len(parts) else "" for idx, key in enumerate(header)}
            rows.append(row)
    return rows


def _num(row: dict[str, str], key: str) -> float:
    value = (row.get(key) or "").strip()
    return float(value) if value else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-log", required=True, type=Path)
    parser.add_argument("--turn", default="_t1")
    parser.add_argument("--assert-used", action="store_true")
    args = parser.parse_args()

    rows = _read_mixed_header_csv(args.vision_log)

    rows = [row for row in rows if args.turn in (row.get("request_ids") or "")]
    total_tokens = int(sum(_num(row, "total_tokens") for row in rows))
    reused_tokens = int(sum(_num(row, "reused_tokens") for row in rows))
    computed_tokens = int(sum(_num(row, "computed_tokens") for row in rows))
    used_rows = sum(1 for row in rows if _num(row, "partial_vit_used") > 0)
    ratios = [_num(row, "token_reuse_ratio") for row in rows if (row.get("token_reuse_ratio") or "").strip()]
    token_cos_min = [_num(row, "partial_vit_token_cosine_min") for row in rows if row.get("partial_vit_token_cosine_min")]
    token_cos_mean = [_num(row, "partial_vit_token_cosine_mean") for row in rows if row.get("partial_vit_token_cosine_mean")]
    token_cos_max = [_num(row, "partial_vit_token_cosine_max") for row in rows if row.get("partial_vit_token_cosine_max")]
    reused_layer_tokens = int(sum(_num(row, "partial_vit_reused_token_layer_tokens") for row in rows))
    computed_layer_tokens = int(sum(_num(row, "partial_vit_computed_token_layer_tokens") for row in rows))
    merged_total_tokens = int(sum(_num(row, "merged_total_tokens") for row in rows))
    merged_reused_tokens = int(sum(_num(row, "merged_reused_tokens") for row in rows))
    merged_ratios = [
        _num(row, "merged_token_reuse_ratio") for row in rows if (row.get("merged_token_reuse_ratio") or "").strip()
    ]
    merged_cos_min = [
        _num(row, "partial_vit_merged_token_cosine_min")
        for row in rows
        if row.get("partial_vit_merged_token_cosine_min")
    ]
    merged_cos_mean = [
        _num(row, "partial_vit_merged_token_cosine_mean")
        for row in rows
        if row.get("partial_vit_merged_token_cosine_mean")
    ]
    merged_cos_max = [
        _num(row, "partial_vit_merged_token_cosine_max")
        for row in rows
        if row.get("partial_vit_merged_token_cosine_max")
    ]

    print(f"rows={len(rows)}")
    print(f"partial_vit_used_rows={used_rows}")
    print(f"reused_tokens={reused_tokens}")
    print(f"total_tokens={total_tokens}")
    print(f"computed_tokens={computed_tokens}")
    print(f"token_reuse_ratio={reused_tokens / total_tokens if total_tokens else 0.0:.6f}")
    print(f"token_reuse_ratio_max={max(ratios) if ratios else 0.0:.6f}")
    print(f"partial_vit_reused_token_layer_tokens={reused_layer_tokens}")
    print(f"partial_vit_computed_token_layer_tokens={computed_layer_tokens}")
    if token_cos_min:
        print(f"token_cosine_min={min(token_cos_min):.6f}")
    if token_cos_mean:
        print(f"token_cosine_mean_avg={sum(token_cos_mean) / len(token_cos_mean):.6f}")
    if token_cos_max:
        print(f"token_cosine_max={max(token_cos_max):.6f}")
    print(f"merged_reused_tokens={merged_reused_tokens}")
    print(f"merged_total_tokens={merged_total_tokens}")
    print(f"merged_token_reuse_ratio={merged_reused_tokens / merged_total_tokens if merged_total_tokens else 0.0:.6f}")
    print(f"merged_token_reuse_ratio_max={max(merged_ratios) if merged_ratios else 0.0:.6f}")
    if merged_cos_min:
        print(f"merged_token_cosine_min={min(merged_cos_min):.6f}")
    if merged_cos_mean:
        print(f"merged_token_cosine_mean_avg={sum(merged_cos_mean) / len(merged_cos_mean):.6f}")
    if merged_cos_max:
        print(f"merged_token_cosine_max={max(merged_cos_max):.6f}")

    if args.assert_used:
        assert used_rows > 0, "token partial path did not record any used row"
        assert reused_tokens > 0 or merged_reused_tokens > 0, "no token was reused"
        if reused_tokens > 0:
            assert reused_layer_tokens > 0, "no token-layer reuse was recorded"
        if merged_reused_tokens > 0:
            assert merged_total_tokens > 0, "no merged token total was recorded"
        print("TOKEN_REUSE_PROVEN")


if __name__ == "__main__":
    main()
