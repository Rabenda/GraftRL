#!/usr/bin/env python3
"""Compare turn1 ViT timing for the token-sparse reuse path vs a baseline.

Reads two vision_encoder_log CSVs (baseline = cache off, reuse = token_sparse)
and reports turn1 vision_encoder_time_ms plus the realized token reuse, so you
can see whether skipping reused tokens' FFN actually saves wall time.

Usage:
  python3 compare_token_sparse_timing.py \
    --baseline profile_logs_vtool_chart_token_reuse/vision_encoder_log_vtool_chart_baseline_nocache.csv \
    --reuse    profile_logs_vtool_chart_token_reuse/vision_encoder_log_vtool_chart_token_sparse_t084.csv

Both runs must use the SAME data/batch and have merged-token-sim logging OFF
(SGLANG_GRPO_LOG_MERGED_TOKEN_SIM=0), otherwise the reuse run runs a second
full ViT forward and the timing is meaningless.
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path


def _read_mixed_header_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV that may switch header rows mid-file (header line starts with 'timestamp')."""
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    with path.open(newline="") as f:
        for parts in csv.reader(f):
            if not parts:
                continue
            if parts[0] == "timestamp":
                header = parts
                continue
            if header is None:
                continue
            rows.append({k: parts[i] if i < len(parts) else "" for i, k in enumerate(header)})
    return rows


def _num(row: dict[str, str], key: str) -> float:
    v = (row.get(key) or "").strip()
    return float(v) if v else 0.0


def _turn1(rows: list[dict[str, str]], turn: str) -> list[dict[str, str]]:
    return [r for r in rows if turn in (r.get("request_ids") or "")]


def _summary(label: str, rows: list[dict[str, str]]) -> dict:
    times = [_num(r, "vision_encoder_time_ms") for r in rows if (r.get("vision_encoder_time_ms") or "").strip()]
    reused = int(sum(_num(r, "reused_tokens") for r in rows))
    total = int(sum(_num(r, "total_tokens") for r in rows))
    used_rows = sum(1 for r in rows if _num(r, "partial_vit_used") > 0)
    out = {
        "label": label,
        "n": len(rows),
        "time_sum_s": sum(times) / 1000.0 if times else 0.0,
        "time_mean_ms": st.mean(times) if times else 0.0,
        "time_median_ms": st.median(times) if times else 0.0,
        "reused_tokens": reused,
        "total_tokens": total,
        "token_reuse_ratio": reused / total if total else 0.0,
        "partial_vit_used_rows": used_rows,
    }
    return out


def _print(s: dict) -> None:
    print(f"\n[{s['label']}]")
    print(f"  turn1 rows           = {s['n']}")
    print(f"  vision_time mean     = {s['time_mean_ms']:.1f} ms")
    print(f"  vision_time median   = {s['time_median_ms']:.1f} ms")
    print(f"  vision_time sum      = {s['time_sum_s']:.2f} s")
    print(f"  partial_vit_used_rows= {s['partial_vit_used_rows']}")
    print(f"  reused/total tokens  = {s['reused_tokens']}/{s['total_tokens']} "
          f"({s['token_reuse_ratio'] * 100:.2f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--reuse", required=True, type=Path)
    ap.add_argument("--turn", default="_t1")
    args = ap.parse_args()

    base = _summary("baseline (cache off)", _turn1(_read_mixed_header_csv(args.baseline), args.turn))
    reuse = _summary("token_sparse reuse", _turn1(_read_mixed_header_csv(args.reuse), args.turn))

    _print(base)
    _print(reuse)

    print("\n[speedup: token_sparse vs baseline, turn1]")
    if base["time_mean_ms"] > 0:
        d_mean = base["time_mean_ms"] - reuse["time_mean_ms"]
        print(f"  mean   {base['time_mean_ms']:.1f} -> {reuse['time_mean_ms']:.1f} ms "
              f"(delta {d_mean:+.1f} ms, {d_mean / base['time_mean_ms'] * 100:+.1f}%)")
    if base["time_sum_s"] > 0:
        d_sum = base["time_sum_s"] - reuse["time_sum_s"]
        print(f"  sum    {base['time_sum_s']:.2f} -> {reuse['time_sum_s']:.2f} s "
              f"(delta {d_sum:+.2f} s, {d_sum / base['time_sum_s'] * 100:+.1f}%)")

    if reuse["reused_tokens"] == 0:
        print("\n  WARNING: 0 reused tokens -- nothing was skipped; lower the threshold "
              "(TOKEN_REUSE_THRESHOLD) or check the donor gate.")
    elif reuse["time_mean_ms"] >= base["time_mean_ms"]:
        print("\n  NOTE: reuse path is not faster despite reused tokens. Prefull-only FFN "
              "savings are likely smaller than gather/scatter + full-attention overhead.")


if __name__ == "__main__":
    main()
