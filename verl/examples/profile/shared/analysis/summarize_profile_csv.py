#!/usr/bin/env python3
"""Summarize vision_encoder / verl_sglang_generate / model_forward CSV logs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean_col(rows: list[dict], col: str) -> float | None:
    vals = []
    for r in rows:
        if col not in r or r[col] in ("", None):
            continue
        try:
            vals.append(float(r[col]))
        except ValueError:
            pass
    return statistics.mean(vals) if vals else None


def summarize(name: str, path: Path):
    if not path.exists():
        print(f"[missing] {name}: {path}")
        return
    rows = load_csv(path)
    print(f"\n=== {name} ({path.name}) rows={len(rows)} ===")
    cols = {
        "verl_sglang": ["image_prompt_tokens", "image_prompt_ratio", "prefill_tokens", "generate_e2e_ms"],
        "vision_encoder": ["image_tokens", "image_token_ratio", "vision_encoder_time_ms", "prefill_tokens"],
        "model_forward": ["prefill_tokens", "decode_tokens"],
    }
    if "image_prompt_ratio" in (rows[0] if rows else {}):
        key = "verl_sglang"
    elif "vision_encoder_time_ms" in (rows[0] if rows else {}):
        key = "vision_encoder"
    else:
        key = "model_forward"
    for col in cols[key]:
        m = mean_col(rows, col)
        if m is not None:
            print(f"  {col}_mean: {m:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="/workspace/repo/verl_vision/profile_logs")
    parser.add_argument("--suffix", required=True, help="SGLANG_INFERENCE_LOG_SUFFIX value")
    args = parser.parse_args()
    d = Path(args.log_dir)
    suf = args.suffix
    summarize("generate", d / f"verl_sglang_generate_log_{suf}.csv")
    summarize("vision_encoder", d / f"vision_encoder_log_{suf}.csv")
    summarize("model_forward", d / f"model_forward_log_{suf}.csv")


if __name__ == "__main__":
    main()
