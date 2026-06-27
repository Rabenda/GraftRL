#!/usr/bin/env python3
"""Summarize merged image-token reuse threshold sweep."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


_ANSWER_PATTERNS = [
    re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL),
    re.compile(r"\\boxed\{([^{}]*)\}", re.IGNORECASE | re.DOTALL),
    re.compile(r"FINAL ANSWER:\s*(.+?)(?:\n\s*\n|\.?\s*TERMINATE|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"ANSWER:\s*(.+?)(?:\n\s*\n|FINAL ANSWER:|\.?\s*TERMINATE|$)", re.IGNORECASE | re.DOTALL),
]


def _tag(threshold: str) -> str:
    return "t" + threshold.replace(".", "")


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
            rows.append({key: parts[idx] if idx < len(parts) else "" for idx, key in enumerate(header)})
    return rows


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_answer(text: Any) -> str:
    text = str(text or "").strip().lower()
    text = text.replace(",", "")
    text = text.strip("`'\" ")
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_answer(text: Any) -> str:
    text = str(text or "")
    for pattern in _ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return str(matches[-1]).strip()
    return text.strip()


def _answer_acc(prediction: Any, ground_truth: Any) -> float:
    pred = _normalize_answer(_extract_answer(prediction))
    gt = _normalize_answer(ground_truth)
    if not pred or not gt:
        return 0.0
    if pred == gt:
        return 1.0
    try:
        pred_num = float(pred.rstrip("%"))
        gt_num = float(gt.rstrip("%"))
        if abs(pred_num - gt_num) < 1e-3:
            return 1.0
        if gt_num != 0 and abs(pred_num / gt_num - 1.0) < 0.01:
            return 1.0
    except ValueError:
        pass
    return 0.0


def _vision_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing_vision": True}
    rows = [row for row in _read_mixed_header_csv(path) if "_t1" in (row.get("request_ids") or "")]
    used = [row for row in rows if _f(row.get("partial_vit_used")) > 0]
    vision_ms = [_f(row.get("vision_encoder_time_ms")) for row in rows if row.get("vision_encoder_time_ms")]
    merged_total = int(sum(_f(row.get("merged_total_tokens")) for row in rows))
    merged_reused = int(sum(_f(row.get("merged_reused_tokens")) for row in rows))
    merged_ratios = [_f(row.get("merged_token_reuse_ratio")) for row in rows if row.get("merged_token_reuse_ratio")]
    merged_cos_mean = [
        _f(row.get("partial_vit_merged_token_cosine_mean"))
        for row in rows
        if row.get("partial_vit_merged_token_cosine_mean")
    ]
    merged_cos_max = [
        _f(row.get("partial_vit_merged_token_cosine_max"))
        for row in rows
        if row.get("partial_vit_merged_token_cosine_max")
    ]
    return {
        "missing_vision": False,
        "vision_rows_t1": len(rows),
        "partial_used_rows": len(used),
        "vision_ms_mean": mean(vision_ms) if vision_ms else 0.0,
        "reused_windows": int(sum(_f(row.get("reused_windows")) for row in rows)),
        "total_windows": int(sum(_f(row.get("total_windows")) for row in rows)),
        "window_reuse_ratio": (
            sum(_f(row.get("reused_windows")) for row in rows)
            / sum(_f(row.get("total_windows")) for row in rows)
            if sum(_f(row.get("total_windows")) for row in rows)
            else 0.0
        ),
        "merged_reused_tokens": merged_reused,
        "merged_total_tokens": merged_total,
        "merged_token_reuse_ratio": merged_reused / merged_total if merged_total else 0.0,
        "merged_token_reuse_ratio_max": max(merged_ratios) if merged_ratios else 0.0,
        "merged_cosine_mean_avg": mean(merged_cos_mean) if merged_cos_mean else 0.0,
        "merged_cosine_max": max(merged_cos_max) if merged_cos_max else 0.0,
    }


def _rollout_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.jsonl"))
    else:
        return []
    rows: list[dict[str, Any]] = []
    for file in files:
        with file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _rollout_summary(path: Path) -> dict[str, Any]:
    rows = _rollout_rows(path)
    if not rows:
        return {
            "rollout_rows": 0,
            "score_mean": 0.0,
            "positive_rate": 0.0,
            "acc_mean": 0.0,
            "relaxed_acc_mean": 0.0,
            "relaxed_positive_rate": 0.0,
        }
    scores = [_f(row.get("compute_score", row.get("reward_score", row.get("score")))) for row in rows]
    accs = [_f(row.get("acc")) for row in rows if row.get("acc") is not None]
    relaxed_accs = [
        _answer_acc(
            row.get("final_response_text", row.get("response_text", row.get("output", ""))),
            row.get("gts", row.get("ground_truth", "")),
        )
        for row in rows
    ]
    return {
        "rollout_rows": len(rows),
        "score_mean": mean(scores) if scores else 0.0,
        "positive_rate": sum(1 for score in scores if score > 0.0) / len(scores) if scores else 0.0,
        "acc_mean": mean(accs) if accs else 0.0,
        "relaxed_acc_mean": mean(relaxed_accs) if relaxed_accs else 0.0,
        "relaxed_positive_rate": sum(1 for acc in relaxed_accs if acc > 0.0) / len(relaxed_accs) if relaxed_accs else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--thresholds", nargs="+", default=["0.95", "0.90", "0.85", "0.80", "0.75"])
    parser.add_argument("--suffix-prefix", default="vtool_chart_merged_reuse_probe")
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    for threshold in args.thresholds:
        suffix = f"{args.suffix_prefix}_{_tag(threshold)}"
        vision_path = args.log_root / f"vision_encoder_log_{suffix}.csv"
        rollout_path = args.log_root / f"rollout_data_{suffix}"
        row = {
            "threshold": threshold,
            "suffix": suffix,
            **_vision_summary(vision_path),
            **_rollout_summary(rollout_path),
        }
        results.append(row)

    fields = [
        "threshold",
        "vision_rows_t1",
        "partial_used_rows",
        "merged_token_reuse_ratio",
        "merged_token_reuse_ratio_max",
        "merged_cosine_mean_avg",
        "merged_cosine_max",
        "window_reuse_ratio",
        "reused_windows",
        "total_windows",
        "vision_ms_mean",
        "rollout_rows",
        "score_mean",
        "positive_rate",
        "acc_mean",
        "relaxed_acc_mean",
        "relaxed_positive_rate",
        "suffix",
    ]
    print("| " + " | ".join(fields) + " |")
    print("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in results:
        vals = []
        for field in fields:
            value = row.get(field, "")
            vals.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        print("| " + " | ".join(vals) + " |")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in results:
                writer.writerow({field: row.get(field, "") for field in fields})


if __name__ == "__main__":
    main()
