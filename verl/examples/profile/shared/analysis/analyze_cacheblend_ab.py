#!/usr/bin/env python3
"""Summarize VLM-CacheBlend on/off rollout logs.

This is a lightweight correctness and timing gate for the FA3 CacheBlend path.
It compares:

* model_forward_log EXTEND rows: donor/recipient counts, attention-skip tokens,
  forward_time_ms, and ms/token.
* rollout JSONL: response changes, acc changes, correct_to_wrong, and refocus/tool
  subset parity by rollout_idx.
* verl_sglang_generate_log: coarse per-turn sglang_call_ms and prefill launch
  latency. These include scheduling/HTTP effects and should not be treated as
  kernel-only timing.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return float(st.mean(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(st.median(values)) if values else 0.0


def _sum(values: list[float]) -> float:
    return float(sum(values)) if values else 0.0


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    paths = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    rows: list[dict[str, Any]] = []
    for p in paths:
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row["_file"] = str(p)
                    rows.append(row)
    return rows


def _first_text(row: dict[str, Any]) -> str:
    return str(
        row.get("final_response_text")
        or row.get("response_text")
        or row.get("vtool_final_response_text")
        or row.get("output")
        or ""
    ).strip()


_ANSWER_PATTERNS = [
    re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL),
    re.compile(r"\\boxed\{([^{}]*)\}", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"FINAL ANSWER:\s*(.+?)(?:\n\s*\n|\.?\s*TERMINATE|$)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"ANSWER:\s*(.+?)(?:\n\s*\n|FINAL ANSWER:|\.?\s*TERMINATE|$)",
        re.IGNORECASE | re.DOTALL,
    ),
]


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
            return str(matches[-1]).strip().rstrip(".")
    return text.strip()


def _relaxed_acc(row: dict[str, Any]) -> float:
    pred = _normalize_answer(_extract_answer(_first_text(row)))
    gt = _normalize_answer(row.get("gts"))
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


def _is_refocus_or_tool(row: dict[str, Any]) -> bool:
    return bool(row.get("vtool_tool_attempted")) or str(
        row.get("vtool_refocus_source") or "none"
    ) != "none"


def _summarize_forward_group(
    rows: list[dict[str, str]],
    pred: Callable[[dict[str, str]], bool],
) -> dict[str, Any]:
    xs = [r for r in rows if pred(r)]
    fts = [_as_float(r.get("forward_time_ms")) for r in xs]
    toks = [_as_float(r.get("prefill_tokens")) for r in xs]
    per_tok = [ft / max(tok, 1.0) for ft, tok in zip(fts, toks)]
    skipped = [_as_int(r.get("cacheblend_attention_skipped_tokens")) for r in xs]
    reused = [_as_int(r.get("cacheblend_reused_tokens")) for r in xs]
    recomputed = [_as_int(r.get("cacheblend_recomputed_tokens")) for r in xs]
    return {
        "n": len(xs),
        "tokens_sum": int(_sum(toks)),
        "forward_time_ms_sum": round(_sum(fts), 3),
        "forward_time_ms_mean": round(_mean(fts), 3),
        "forward_time_ms_median": round(_median(fts), 3),
        "ms_per_token_mean": round(_mean(per_tok), 6),
        "attention_skipped_tokens_sum": int(_sum(skipped)),
        "reused_tokens_sum": int(_sum(reused)),
        "recomputed_tokens_sum": int(_sum(recomputed)),
    }


def _summarize_forward(log_dir: Path, suffix: str) -> dict[str, Any]:
    rows = [
        r
        for r in _csv_rows(log_dir / f"model_forward_log_{suffix}.csv")
        if r.get("mode") == "EXTEND"
    ]
    role_counts = Counter(r.get("cacheblend_role") or "<NA>" for r in rows)
    fallback_counts = Counter(r.get("cacheblend_fallback_reason") or "<NA>" for r in rows)
    skipped_counts = Counter(
        r.get("cacheblend_attention_skipped_tokens") or "<NA>" for r in rows
    )
    hook_exceptions = [
        r
        for r in rows
        if "hook_exception" in str(r.get("cacheblend_fallback_reason") or "")
    ]
    recipient_rows = [r for r in rows if r.get("cacheblend_role") == "recipient"]
    return {
        "path": str(log_dir / f"model_forward_log_{suffix}.csv"),
        "extend_rows": len(rows),
        "role_counts": dict(role_counts),
        "fallback_counts": dict(fallback_counts),
        "attention_skipped_token_counts": dict(skipped_counts),
        "hook_exceptions": len(hook_exceptions),
        "all": _summarize_forward_group(rows, lambda _: True),
        "turn0_none": _summarize_forward_group(
            rows,
            lambda r: str(r.get("cacheblend_fallback_reason") or "").startswith(
                "agent_turn_mismatch:0"
            ),
        ),
        "donor": _summarize_forward_group(
            rows, lambda r: r.get("cacheblend_role") == "donor"
        ),
        "recipient": _summarize_forward_group(
            rows, lambda r: r.get("cacheblend_role") == "recipient"
        ),
        "recipient_rows": [
            {
                key: r.get(key)
                for key in (
                    "pass_id",
                    "batch_size",
                    "prefill_tokens",
                    "forward_time_ms",
                    "cacheblend_reused_tokens",
                    "cacheblend_recomputed_tokens",
                    "cacheblend_attention_skipped_tokens",
                    "cacheblend_attention_active_ranges",
                )
            }
            for r in recipient_rows
        ],
    }


def _summarize_generate(log_dir: Path, suffix: str) -> dict[str, Any]:
    rows = _csv_rows(log_dir / f"verl_sglang_generate_log_{suffix}.csv")
    by_turn: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_turn.setdefault(str(row.get("agent_turn") or "<NA>"), []).append(row)

    out: dict[str, Any] = {
        "path": str(log_dir / f"verl_sglang_generate_log_{suffix}.csv"),
        "rows": len(rows),
        "by_turn": {},
    }
    for turn, xs in sorted(by_turn.items()):
        calls = [_as_float(r.get("sglang_call_ms")) for r in xs]
        prefill = [_as_float(r.get("prefill_launch_latency_ms")) for r in xs]
        out["by_turn"][turn] = {
            "n": len(xs),
            "sglang_call_ms_mean": round(_mean(calls), 3),
            "sglang_call_ms_median": round(_median(calls), 3),
            "prefill_launch_latency_ms_mean": round(_mean(prefill), 3),
            "prefill_launch_latency_ms_median": round(_median(prefill), 3),
        }
    return out


def _summarize_rollout(
    off_log_dir: Path,
    off_suffix: str,
    on_log_dir: Path,
    on_suffix: str,
) -> dict[str, Any]:
    off_rows = _jsonl_rows(off_log_dir / f"rollout_data_{off_suffix}")
    on_rows = _jsonl_rows(on_log_dir / f"rollout_data_{on_suffix}")

    def keyed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        seen: dict[str, int] = defaultdict(int)
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            rollout_idx = str(row.get("rollout_idx"))
            occurrence = seen[rollout_idx]
            seen[rollout_idx] += 1
            out[f"{rollout_idx}#{occurrence}"] = row
        return out

    off = keyed(off_rows)
    on = keyed(on_rows)
    def key_sort(value: str) -> tuple[int, int]:
        rollout_idx, _, occurrence = value.partition("#")
        return (_as_int(rollout_idx, 10**9), _as_int(occurrence, 10**9))

    common = sorted(set(off) & set(on), key=key_sort)

    changed: list[str] = []
    acc_changed: list[str] = []
    correct_to_wrong: list[str] = []
    wrong_to_correct: list[str] = []
    relaxed_acc_changed: list[str] = []
    relaxed_correct_to_wrong: list[str] = []
    relaxed_wrong_to_correct: list[str] = []
    refocus_or_tool: list[str] = []
    refocus_or_tool_changed: list[str] = []
    num_turns_changed = 0

    for key in common:
        a, b = off[key], on[key]
        same_text = _first_text(a) == _first_text(b)
        if not same_text:
            changed.append(key)
        aa = _as_float(a.get("acc") or a.get("score"))
        bb = _as_float(b.get("acc") or b.get("score"))
        if aa != bb:
            acc_changed.append(key)
        if aa > 0 and bb <= 0:
            correct_to_wrong.append(key)
        if aa <= 0 and bb > 0:
            wrong_to_correct.append(key)
        aaa = _relaxed_acc(a)
        bbb = _relaxed_acc(b)
        if aaa != bbb:
            relaxed_acc_changed.append(key)
        if aaa > 0 and bbb <= 0:
            relaxed_correct_to_wrong.append(key)
        if aaa <= 0 and bbb > 0:
            relaxed_wrong_to_correct.append(key)
        if _as_int(a.get("num_turns") or a.get("__num_turns__")) != _as_int(
            b.get("num_turns") or b.get("__num_turns__")
        ):
            num_turns_changed += 1
        if _is_refocus_or_tool(a) or _is_refocus_or_tool(b):
            refocus_or_tool.append(key)
            if not same_text:
                refocus_or_tool_changed.append(key)

    examples = []
    for key in changed[:5]:
        rollout_idx = key.split("#", 1)[0]
        examples.append(
            {
                "key": key,
                "rollout_idx": rollout_idx,
                "off_acc": _as_float(off[key].get("acc") or off[key].get("score")),
                "on_acc": _as_float(on[key].get("acc") or on[key].get("score")),
                "off_relaxed_acc": _relaxed_acc(off[key]),
                "on_relaxed_acc": _relaxed_acc(on[key]),
                "off_refocus_source": off[key].get("vtool_refocus_source"),
                "on_refocus_source": on[key].get("vtool_refocus_source"),
                "off_text_tail": _first_text(off[key])[-500:],
                "on_text_tail": _first_text(on[key])[-500:],
            }
        )

    return {
        "off_rows": len(off_rows),
        "on_rows": len(on_rows),
        "common_by_rollout_idx_occurrence": len(common),
        "response_changed": len(changed),
        "acc_changed": len(acc_changed),
        "correct_to_wrong": len(correct_to_wrong),
        "wrong_to_correct": len(wrong_to_correct),
        "relaxed_acc_changed": len(relaxed_acc_changed),
        "relaxed_correct_to_wrong": len(relaxed_correct_to_wrong),
        "relaxed_wrong_to_correct": len(relaxed_wrong_to_correct),
        "num_turns_changed": num_turns_changed,
        "off_acc_sum_common": _sum(
            [_as_float(off[k].get("acc") or off[k].get("score")) for k in common]
        ),
        "on_acc_sum_common": _sum(
            [_as_float(on[k].get("acc") or on[k].get("score")) for k in common]
        ),
        "off_relaxed_acc_sum_common": _sum([_relaxed_acc(off[k]) for k in common]),
        "on_relaxed_acc_sum_common": _sum([_relaxed_acc(on[k]) for k in common]),
        "refocus_or_tool_common": len(refocus_or_tool),
        "refocus_or_tool_response_changed": len(refocus_or_tool_changed),
        "changed_examples": examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--off-log-dir", required=True, type=Path)
    ap.add_argument("--off-suffix", required=True)
    ap.add_argument("--on-log-dir", required=True, type=Path)
    ap.add_argument("--on-suffix", required=True)
    args = ap.parse_args()

    result = {
        "off": {
            "forward": _summarize_forward(args.off_log_dir, args.off_suffix),
            "generate": _summarize_generate(args.off_log_dir, args.off_suffix),
        },
        "on": {
            "forward": _summarize_forward(args.on_log_dir, args.on_suffix),
            "generate": _summarize_generate(args.on_log_dir, args.on_suffix),
        },
        "rollout_compare": _summarize_rollout(
            args.off_log_dir,
            args.off_suffix,
            args.on_log_dir,
            args.on_suffix,
        ),
        "notes": [
            "model_forward recipient rows are the strongest kernel-side timing signal.",
            "generate log timings include scheduling/HTTP/queue effects and are coarse.",
            "off/on EXTEND rows may not be directly comparable if batching differs.",
        ],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
