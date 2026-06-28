#!/usr/bin/env python3
"""Semantic gate for VLM-CacheBlend A/B and selector comparisons.

The gate compares one baseline run, normally CacheBlend off, against one or more
candidate runs, e.g. kvdev and cosine selection. It is intentionally log-only: it
reads rollout JSONL plus optional model/generate CSVs and does not require GPUs.

Example:
  python3 examples/profile/shared/analysis/semantic_cacheblend_gate.py \
    --baseline-log-dir profile_logs_geo3k_refocus_exact_semantic_off_2g \
    --baseline-suffix geo3k_refocus_exact_semantic_off_2g \
    --candidate kvdev:profile_logs_geo3k_refocus_exact_semantic_kvdev_2g:geo3k_refocus_exact_semantic_kvdev_2g \
    --candidate cos:profile_logs_geo3k_refocus_exact_semantic_cos_2g:geo3k_refocus_exact_semantic_cos_2g \
    --fail-on-threshold
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


_ANSWER_PATTERNS = [
    re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL),
    re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.IGNORECASE | re.DOTALL),
    re.compile(r"FINAL ANSWER:\s*(.+?)(?:\n\s*\n|\.?\s*TERMINATE|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"ANSWER:\s*(.+?)(?:\n\s*\n|FINAL ANSWER:|\.?\s*TERMINATE|$)", re.IGNORECASE | re.DOTALL),
]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return float(st.mean(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(st.median(values)) if values else 0.0


def _rate(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    paths = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    rows: list[dict[str, Any]] = []
    for p in paths:
        if not p.is_file():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
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


def _extract_answer(text: Any) -> str:
    raw = str(text or "")
    for pattern in _ANSWER_PATTERNS:
        matches = pattern.findall(raw)
        if matches:
            return str(matches[-1]).strip().rstrip(".")
    return ""


def _normalize_answer(text: Any) -> str:
    text = str(text or "").strip().lower()
    text = text.replace(",", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "")
    text = re.sub(r"\\sqrt\s*\{\s*([^{}]+?)\s*\}", r"sqrt(\1)", text)
    text = re.sub(r"\\sqrt\s+([a-z0-9.]+)", r"sqrt(\1)", text)
    text = re.sub(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}", r"(\1)/(\2)", text)
    text = text.strip("`'\" .")
    text = re.sub(r"\s+", "", text)
    text = text.replace("{", "").replace("}", "")
    return text


def _answer(row: dict[str, Any]) -> str:
    return _normalize_answer(_extract_answer(_first_text(row)))


def _gt(row: dict[str, Any]) -> str:
    return _normalize_answer(row.get("gts"))


def _score(row: dict[str, Any]) -> float:
    return _as_float(row.get("acc", row.get("score", row.get("reward_score", 0.0))))


def _answer_acc(row: dict[str, Any]) -> float:
    pred, gt = _answer(row), _gt(row)
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


def _prompt_key(row: dict[str, Any]) -> str:
    material = json.dumps(
        {
            "input": row.get("input", ""),
            "gts": row.get("gts", ""),
            "step": row.get("step", ""),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]


def _keyed_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    seen: dict[str, int] = defaultdict(int)
    keyed: dict[str, dict[str, Any]] = {}
    for row in rows:
        base = _prompt_key(row)
        occ = seen[base]
        seen[base] += 1
        keyed[f"{base}#{occ}"] = row
    return keyed


def _summarize_rollout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [_score(r) for r in rows]
    answer_scores = [_answer_acc(r) for r in rows]
    return {
        "rows": len(rows),
        "score_sum": round(sum(scores), 6),
        "score_mean": round(_mean(scores), 6),
        "answer_acc_sum": round(sum(answer_scores), 6),
        "answer_acc_mean": round(_mean(answer_scores), 6),
    }


def _summarize_forward(log_dir: Path, suffix: str) -> dict[str, Any]:
    rows = [
        r
        for r in _csv_rows(log_dir / f"model_forward_log_{suffix}.csv")
        if r.get("mode") == "EXTEND"
    ]
    recipients = [r for r in rows if r.get("cacheblend_role") == "recipient"]
    used = [r for r in recipients if str(r.get("cacheblend_used") or "0") == "1"]
    forward_ms = [_as_float(r.get("forward_time_ms")) for r in recipients]
    reused = [_as_int(r.get("cacheblend_reused_tokens")) for r in recipients]
    recomputed = [_as_int(r.get("cacheblend_recomputed_tokens")) for r in recipients]
    skipped = [_as_int(r.get("cacheblend_attention_skipped_tokens")) for r in recipients]
    select_modes = Counter(r.get("cacheblend_select_mode") or "" for r in recipients)
    return {
        "extend_rows": len(rows),
        "recipient_rows": len(recipients),
        "recipient_used_rows": len(used),
        "recipient_forward_ms_median": round(_median(forward_ms), 3),
        "recipient_forward_ms_mean": round(_mean(forward_ms), 3),
        "reused_tokens_sum": int(sum(reused)),
        "recomputed_tokens_sum": int(sum(recomputed)),
        "attention_skipped_tokens_sum": int(sum(skipped)),
        "select_modes": dict(select_modes),
    }


def _summarize_generate(log_dir: Path, suffix: str) -> dict[str, Any]:
    rows = _csv_rows(log_dir / f"verl_sglang_generate_log_{suffix}.csv")
    by_turn: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_turn[str(row.get("agent_turn") or "")].append(row)
    out: dict[str, Any] = {"rows": len(rows)}
    for turn, xs in sorted(by_turn.items()):
        calls = [_as_float(r.get("sglang_call_ms")) for r in xs]
        prefill = [_as_float(r.get("prefill_launch_latency_ms")) for r in xs]
        out[f"turn{turn}_sglang_call_ms_median"] = round(_median(calls), 3)
        out[f"turn{turn}_sglang_call_ms_mean"] = round(_mean(calls), 3)
        out[f"turn{turn}_prefill_launch_latency_ms_median"] = round(_median(prefill), 3)
    return out


def _speedup(base: float, cand: float) -> float:
    if base <= 0 or cand <= 0:
        return 0.0
    return round((base - cand) / base * 100.0, 3)


def _compare_candidate(
    *,
    label: str,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    baseline_forward: dict[str, Any],
    candidate_forward: dict[str, Any],
    baseline_generate: dict[str, Any],
    candidate_generate: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    base = _keyed_rows(baseline_rows)
    cand = _keyed_rows(candidate_rows)
    common = sorted(set(base) & set(cand))

    response_changed = 0
    answer_changed = 0
    score_changed = 0
    correct_to_wrong = 0
    wrong_to_correct = 0
    answer_correct_to_wrong = 0
    answer_wrong_to_correct = 0
    score_drop = 0.0
    examples: list[dict[str, Any]] = []

    for key in common:
        a, b = base[key], cand[key]
        if _first_text(a) != _first_text(b):
            response_changed += 1
        ans_a, ans_b = _answer(a), _answer(b)
        if (ans_a or ans_b) and ans_a != ans_b:
            answer_changed += 1
        sa, sb = _score(a), _score(b)
        if sa != sb:
            score_changed += 1
        if sa > 0 and sb <= 0:
            correct_to_wrong += 1
        if sa <= 0 and sb > 0:
            wrong_to_correct += 1
        aa, ab = _answer_acc(a), _answer_acc(b)
        if aa > 0 and ab <= 0:
            answer_correct_to_wrong += 1
        if aa <= 0 and ab > 0:
            answer_wrong_to_correct += 1
        if sb < sa:
            score_drop += sa - sb
        if len(examples) < args.max_examples and (
            ((ans_a or ans_b) and ans_a != ans_b) or (sa > 0 and sb <= 0)
        ):
            examples.append(
                {
                    "key": key,
                    "baseline_score": sa,
                    "candidate_score": sb,
                    "baseline_answer": ans_a,
                    "candidate_answer": ans_b,
                    "ground_truth": _gt(a),
                    "baseline_tail": _first_text(a)[-300:],
                    "candidate_tail": _first_text(b)[-300:],
                }
            )

    common_n = len(common)
    answer_changed_rate = _rate(answer_changed, common_n)
    response_changed_rate = _rate(response_changed, common_n)
    semantic_pass = (
        common_n >= args.min_common
        and correct_to_wrong <= args.max_correct_to_wrong
        and answer_correct_to_wrong <= args.max_answer_correct_to_wrong
        and score_drop <= args.max_score_drop
        and answer_changed_rate <= args.max_answer_changed_rate
    )

    base_turn1 = _as_float(baseline_generate.get("turn1_sglang_call_ms_median"))
    cand_turn1 = _as_float(candidate_generate.get("turn1_sglang_call_ms_median"))
    base_rec = _as_float(baseline_forward.get("recipient_forward_ms_median"))
    cand_rec = _as_float(candidate_forward.get("recipient_forward_ms_median"))
    cacheblend_used = candidate_forward.get("recipient_used_rows", 0) > 0

    return {
        "label": label,
        "pass": semantic_pass,
        "common": common_n,
        "baseline_rows": len(baseline_rows),
        "candidate_rows": len(candidate_rows),
        "response_changed": response_changed,
        "response_changed_rate": round(response_changed_rate, 6),
        "answer_changed": answer_changed,
        "answer_changed_rate": round(answer_changed_rate, 6),
        "score_changed": score_changed,
        "correct_to_wrong": correct_to_wrong,
        "wrong_to_correct": wrong_to_correct,
        "answer_correct_to_wrong": answer_correct_to_wrong,
        "answer_wrong_to_correct": answer_wrong_to_correct,
        "score_drop": round(score_drop, 6),
        "score_gain": round(
            sum(max(_score(cand[k]) - _score(base[k]), 0.0) for k in common), 6
        ),
        "cacheblend_used": cacheblend_used,
        "turn1_sglang_call_ms_median": cand_turn1,
        "turn1_sglang_call_speedup_pct": _speedup(base_turn1, cand_turn1),
        "recipient_forward_ms_median": cand_rec,
        "recipient_forward_speedup_pct": _speedup(base_rec, cand_rec),
        "forward": candidate_forward,
        "generate": candidate_generate,
        "examples": examples,
    }


def _parse_candidate(raw: str) -> tuple[str, Path, str]:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--candidate must be label:log_dir:suffix, got %r" % raw
        )
    label, log_dir, suffix = parts
    return label, Path(log_dir), suffix


def _print_table(result: dict[str, Any]) -> None:
    print("\nSemantic gate")
    print(json.dumps(result["baseline"], indent=2, ensure_ascii=False))
    print(
        "\n| selector | pass | common | ans_changed | c2w | ans_c2w | score_drop | "
        "used | turn1_ms | turn1_spd% | recip_ms | recip_spd% | reused | recomputed | skipped |"
    )
    print(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for row in result["candidates"]:
        fwd = row["forward"]
        print(
            f"| {row['label']} | {str(row['pass']).lower()} | {row['common']} | "
            f"{row['answer_changed']} | {row['correct_to_wrong']} | "
            f"{row['answer_correct_to_wrong']} | {row['score_drop']} | "
            f"{str(row['cacheblend_used']).lower()} | "
            f"{row['turn1_sglang_call_ms_median']} | {row['turn1_sglang_call_speedup_pct']} | "
            f"{row['recipient_forward_ms_median']} | {row['recipient_forward_speedup_pct']} | "
            f"{fwd.get('reused_tokens_sum', 0)} | {fwd.get('recomputed_tokens_sum', 0)} | "
            f"{fwd.get('attention_skipped_tokens_sum', 0)} |"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-log-dir", required=True, type=Path)
    ap.add_argument("--baseline-suffix", required=True)
    ap.add_argument("--candidate", action="append", default=[], type=_parse_candidate)
    ap.add_argument("--min-common", type=int, default=1)
    ap.add_argument("--max-correct-to-wrong", type=int, default=0)
    ap.add_argument("--max-answer-correct-to-wrong", type=int, default=0)
    ap.add_argument("--max-score-drop", type=float, default=0.0)
    ap.add_argument("--max-answer-changed-rate", type=float, default=0.0)
    ap.add_argument("--max-examples", type=int, default=5)
    ap.add_argument("--write-json", type=Path, default=None)
    ap.add_argument("--fail-on-threshold", action="store_true")
    args = ap.parse_args()

    if not args.candidate:
        ap.error("at least one --candidate is required")

    baseline_rollout_dir = args.baseline_log_dir / f"rollout_data_{args.baseline_suffix}"
    baseline_rows = _jsonl_rows(baseline_rollout_dir)
    baseline_forward = _summarize_forward(args.baseline_log_dir, args.baseline_suffix)
    baseline_generate = _summarize_generate(args.baseline_log_dir, args.baseline_suffix)

    result: dict[str, Any] = {
        "thresholds": {
            "min_common": args.min_common,
            "max_correct_to_wrong": args.max_correct_to_wrong,
            "max_answer_correct_to_wrong": args.max_answer_correct_to_wrong,
            "max_score_drop": args.max_score_drop,
            "max_answer_changed_rate": args.max_answer_changed_rate,
        },
        "baseline": {
            "log_dir": str(args.baseline_log_dir),
            "suffix": args.baseline_suffix,
            "rollout": _summarize_rollout(baseline_rows),
            "forward": baseline_forward,
            "generate": baseline_generate,
        },
        "candidates": [],
    }

    for label, log_dir, suffix in args.candidate:
        cand_rows = _jsonl_rows(log_dir / f"rollout_data_{suffix}")
        cand_forward = _summarize_forward(log_dir, suffix)
        cand_generate = _summarize_generate(log_dir, suffix)
        result["candidates"].append(
            _compare_candidate(
                label=label,
                baseline_rows=baseline_rows,
                candidate_rows=cand_rows,
                baseline_forward=baseline_forward,
                candidate_forward=cand_forward,
                baseline_generate=baseline_generate,
                candidate_generate=cand_generate,
                args=args,
            )
        )

    result["pass"] = all(bool(c["pass"]) for c in result["candidates"])
    _print_table(result)
    print("\nJSON")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.fail_on_threshold and not result["pass"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
