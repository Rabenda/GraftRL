#!/usr/bin/env python3
"""Summarize decode phase batching and optional GRPO decode-quorum logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key)
        return default if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return default


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * pct))))
    return values[idx]


def _model_forward_path(log_dir: Path, suffix: str) -> Path:
    name = "model_forward_log"
    if suffix:
        name = f"{name}_{suffix}"
    return log_dir / f"{name}.csv"


def _decode_quorum_path(log_dir: Path, suffix: str) -> Path:
    name = "decode_quorum_log"
    if suffix:
        name = f"{name}_{suffix}"
    return log_dir / f"{name}.csv"


def summarize_model_forward(rows: list[dict[str, str]]) -> dict[str, object]:
    decode_rows = [r for r in rows if r.get("mode") == "DECODE"]
    extend_rows = [r for r in rows if r.get("mode") == "EXTEND"]
    decode_bs = [_float(r, "batch_size") for r in decode_rows]
    decode_ms = [_float(r, "forward_time_ms") for r in decode_rows]

    by_pid: dict[str, dict[str, float]] = {}
    for row in decode_rows:
        pid = row.get("pid", "")
        end_s = _float(row, "timestamp")
        dur_s = _float(row, "forward_time_ms") / 1000.0
        start_s = end_s - dur_s
        slot = by_pid.setdefault(
            pid,
            {
                "first_decode_start_s": start_s,
                "last_decode_end_s": end_s,
                "decode_active_ms": 0.0,
                "extend_interleaved_ms": 0.0,
                "extend_interleaved_count": 0.0,
            },
        )
        slot["first_decode_start_s"] = min(slot["first_decode_start_s"], start_s)
        slot["last_decode_end_s"] = max(slot["last_decode_end_s"], end_s)
        slot["decode_active_ms"] += dur_s * 1000.0

    for row in extend_rows:
        pid = row.get("pid", "")
        if pid not in by_pid:
            continue
        end_s = _float(row, "timestamp")
        dur_s = _float(row, "forward_time_ms") / 1000.0
        start_s = end_s - dur_s
        slot = by_pid[pid]
        if end_s >= slot["first_decode_start_s"] and start_s <= slot["last_decode_end_s"]:
            slot["extend_interleaved_ms"] += dur_s * 1000.0
            slot["extend_interleaved_count"] += 1

    decode_span_ms = 0.0
    decode_idle_ms = 0.0
    for slot in by_pid.values():
        span_ms = (slot["last_decode_end_s"] - slot["first_decode_start_s"]) * 1000.0
        decode_span_ms += span_ms
        decode_idle_ms += max(0.0, span_ms - slot["decode_active_ms"])

    return {
        "decode_passes": len(decode_rows),
        "extend_passes": len(extend_rows),
        "decode_batch_size": {
            "mean": mean(decode_bs) if decode_bs else 0.0,
            "median": median(decode_bs) if decode_bs else 0.0,
            "p10": _percentile(decode_bs, 0.10),
            "p90": _percentile(decode_bs, 0.90),
            "max": max(decode_bs) if decode_bs else 0.0,
        },
        "decode_forward_ms": {
            "mean": mean(decode_ms) if decode_ms else 0.0,
            "median": median(decode_ms) if decode_ms else 0.0,
        },
        "decode_window": {
            "span_ms_sum_over_pids": decode_span_ms,
            "active_ms_sum_over_pids": sum(v["decode_active_ms"] for v in by_pid.values()),
            "idle_ms_sum_over_pids": decode_idle_ms,
            "extend_interleaved_ms": sum(v["extend_interleaved_ms"] for v in by_pid.values()),
            "extend_interleaved_count": int(
                sum(v["extend_interleaved_count"] for v in by_pid.values())
            ),
        },
    }


def summarize_quorum(rows: list[dict[str, str]]) -> dict[str, object]:
    release_rows = [r for r in rows if r.get("event") == "release"]
    hold_rows = [r for r in rows if r.get("event") == "hold"]
    wait_ms = [_float(r, "wait_ms") for r in release_rows]
    group_sizes = [_float(r, "group_size") for r in release_rows]
    reason_counts: dict[str, int] = {}
    spread_ms: list[float] = []
    for row in release_rows:
        reason = row.get("reason", "")
        base = reason.split(";", 1)[0] if reason else ""
        reason_counts[base] = reason_counts.get(base, 0) + 1
        for part in reason.split(";"):
            if part.startswith("spread_ms="):
                try:
                    spread_ms.append(float(part.split("=", 1)[1]))
                except ValueError:
                    pass

    return {
        "hold_events": len(hold_rows),
        "release_events": len(release_rows),
        "release_reasons": reason_counts,
        "release_group_size": {
            "mean": mean(group_sizes) if group_sizes else 0.0,
            "median": median(group_sizes) if group_sizes else 0.0,
            "p10": _percentile(group_sizes, 0.10),
        },
        "release_wait_ms": {
            "mean": mean(wait_ms) if wait_ms else 0.0,
            "median": median(wait_ms) if wait_ms else 0.0,
            "p90": _percentile(wait_ms, 0.90),
        },
        "decode_ready_spread_ms": {
            "mean": mean(spread_ms) if spread_ms else 0.0,
            "median": median(spread_ms) if spread_ms else 0.0,
            "p90": _percentile(spread_ms, 0.90),
        },
    }


def _drop_min_step(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    def parse_step(row: dict[str, str]) -> int | None:
        try:
            return int(float(row.get("global_step", "")))
        except (TypeError, ValueError):
            return None

    steps = []
    for row in rows:
        step = parse_step(row)
        if step is None:
            continue
        if step >= 0:
            steps.append(step)
    if not steps:
        return rows
    min_step = min(steps)
    kept = []
    for row in rows:
        step = parse_step(row)
        if step is None or step != min_step:
            kept.append(row)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--drop-min-step",
        action="store_true",
        help="Drop the lowest non-negative global_step from model_forward rows.",
    )
    args = parser.parse_args()

    model_rows = _read_csv(_model_forward_path(args.log_dir, args.suffix))
    if args.drop_min_step:
        model_rows = _drop_min_step(model_rows)
    quorum_rows = _read_csv(_decode_quorum_path(args.log_dir, args.suffix))
    summary = {
        "model_forward_csv": str(_model_forward_path(args.log_dir, args.suffix)),
        "decode_quorum_csv": str(_decode_quorum_path(args.log_dir, args.suffix)),
        "model_forward": summarize_model_forward(model_rows),
        "decode_quorum": summarize_quorum(quorum_rows),
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()
