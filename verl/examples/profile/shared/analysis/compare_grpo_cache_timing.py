#!/usr/bin/env python3
"""Compare rollout timing across GRPO image-token reuse modes.

Usage (from verl_vision/):
  python3 examples/profile/shared/analysis/compare_grpo_cache_timing.py \\
    --baseline-vision profile_logs_vtool_chart_diversified/vision_encoder_log_*_n4_diversified.csv \\
    --cached-vision profile_logs_vtool_chart_diversified/vision_encoder_log_*_grpo_cache.csv \\
    --baseline-generate profile_logs_vtool_chart_diversified/verl_sglang_generate_log_*_n4_diversified.csv \\
    --cached-generate profile_logs_vtool_chart_diversified/verl_sglang_generate_log_*_grpo_cache.csv

Vision logs explain why the cache hit. Generate logs answer the end-to-end question.
Optional partial logs allow a 3-way comparison:
cache_off vs whole_slot_reuse vs token/window_partial_reuse.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path

_T1_RE = re.compile(r"_t1$")


def _expand_paths(values: list[str] | None) -> list[Path]:
    if not values:
        return []
    out: list[Path] = []
    for value in values:
        matches = sorted(glob.glob(value)) if any(ch in value for ch in "*?[") else []
        out.extend([Path(m) for m in matches] or [Path(value)])
    return out


def _load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _load_many(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        rows.extend(_load_rows(path))
    return rows


def _f(row: dict, key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _is_turn1(row: dict) -> bool:
    if str(row.get("agent_turn", "")) == "1":
        return True
    return bool(_T1_RE.search(str(row.get("request_ids") or row.get("request_id") or "")))


def _stats(vals: list[float]) -> tuple[float, float, int]:
    if not vals:
        return 0.0, 0.0, 0
    s = sum(vals)
    return s, s / len(vals), len(vals)


def _summarize_vision(rows: list[dict], label: str) -> dict:
    turn1 = [r for r in rows if _is_turn1(r)]
    all_ms = [_f(r, "vision_encoder_time_ms") for r in rows]
    t1_ms = [_f(r, "vision_encoder_time_ms") for r in turn1]
    cached = sum(1 for r in rows if str(r.get("cached_image_features", "0")) == "1")
    vit_skipped = sum(int(_f(r, "grpo_sim_vit_skipped", 0)) for r in rows)
    slot1_skipped = sum(int(_f(r, "grpo_sim_slot1_skipped", 0)) for r in rows)
    exact_reuse = sum(int(_f(r, "grpo_sim_exact_reuse", 0)) for r in rows)
    sim_reuse = sum(int(_f(r, "grpo_sim_similarity_reuse", 0)) for r in rows)
    partial_used = sum(int(_f(r, "partial_vit_used", 0)) for r in rows)
    partial_windows = sum(int(_f(r, "total_windows", 0)) for r in rows)
    partial_reused_windows = sum(int(_f(r, "reused_windows", 0)) for r in rows)
    partial_computed_windows = sum(int(_f(r, "computed_windows", 0)) for r in rows)
    partial_reused_layer_windows = sum(int(_f(r, "partial_vit_reused_window_layer_windows", 0)) for r in rows)
    partial_computed_layer_windows = sum(int(_f(r, "partial_vit_computed_window_layer_windows", 0)) for r in rows)

    total, mean, n = _stats(all_ms)
    t1_total, t1_mean, t1_n = _stats(t1_ms)
    return {
        "label": label,
        "kind": "vision",
        "rows": len(rows),
        "turn1_rows": t1_n,
        "vision_total_ms": total,
        "vision_mean_ms": mean,
        "turn1_total_ms": t1_total,
        "turn1_mean_ms": t1_mean,
        "cached_rows": cached,
        "grpo_vit_skipped": vit_skipped,
        "grpo_slot1_skipped": slot1_skipped,
        "grpo_exact_reuse": exact_reuse,
        "grpo_similarity_reuse": sim_reuse,
        "partial_vit_used": partial_used,
        "partial_total_windows": partial_windows,
        "partial_reused_windows": partial_reused_windows,
        "partial_computed_windows": partial_computed_windows,
        "partial_reuse_ratio": partial_reused_windows / partial_windows if partial_windows else 0.0,
        "partial_reused_window_layer_windows": partial_reused_layer_windows,
        "partial_computed_window_layer_windows": partial_computed_layer_windows,
    }


def _summarize_generate(rows: list[dict], label: str) -> dict:
    turn1 = [r for r in rows if _is_turn1(r)]
    e2e = [_f(r, "generate_e2e_ms") for r in rows]
    t1_e2e = [_f(r, "generate_e2e_ms") for r in turn1]
    sglang = [_f(r, "sglang_call_ms") for r in rows]
    t1_sglang = [_f(r, "sglang_call_ms") for r in turn1]
    queue = [_f(r, "queue_ms") for r in rows]
    output_tokens = [_f(r, "output_tokens") for r in rows]
    total, mean, n = _stats(e2e)
    t1_total, t1_mean, t1_n = _stats(t1_e2e)
    s_total, s_mean, _ = _stats(sglang)
    t1_s_total, t1_s_mean, _ = _stats(t1_sglang)
    q_total, q_mean, _ = _stats(queue)
    tok_total, tok_mean, _ = _stats(output_tokens)
    return {
        "label": label,
        "kind": "generate",
        "rows": len(rows),
        "turn1_rows": t1_n,
        "e2e_total_ms": total,
        "e2e_mean_ms": mean,
        "turn1_e2e_total_ms": t1_total,
        "turn1_e2e_mean_ms": t1_mean,
        "sglang_total_ms": s_total,
        "sglang_mean_ms": s_mean,
        "turn1_sglang_total_ms": t1_s_total,
        "turn1_sglang_mean_ms": t1_s_mean,
        "queue_total_ms": q_total,
        "queue_mean_ms": q_mean,
        "output_tokens_total": tok_total,
        "output_tokens_mean": tok_mean,
    }


def _delta_line(name: str, base_ms: float, cached_ms: float) -> str:
    if base_ms <= 0:
        return f">>> {name}: baseline is 0; cannot compute savings"
    saved = base_ms - cached_ms
    pct = 100.0 * saved / base_ms
    return f">>> {name}: saved {saved:.1f}ms ({pct:.1f}%)"


def _print_single_vision(label: str, row: dict) -> None:
    print(f"-- {label}")
    print(
        f"  rows={row['rows']}  turn1={row['turn1_rows']}  "
        f"vision_total={row['vision_total_ms']:.1f}ms  mean={row['vision_mean_ms']:.2f}ms"
    )
    print(
        f"  turn1_total={row['turn1_total_ms']:.1f}ms  turn1_mean={row['turn1_mean_ms']:.2f}ms"
    )
    print(
        f"  cached_rows={row['cached_rows']}  grpo_vit_skipped={row['grpo_vit_skipped']}  "
        f"slot1_skipped={row['grpo_slot1_skipped']}  exact_reuse={row['grpo_exact_reuse']}  "
        f"similarity_reuse={row['grpo_similarity_reuse']}  partial_vit_used={row['partial_vit_used']}  "
        f"partial_windows={row['partial_reused_windows']}/{row['partial_total_windows']} "
        f"({row['partial_reuse_ratio']:.3f})"
    )


def _print_vision_report(base: dict, cached: dict, partial: dict | None = None) -> None:
    print("\n=== Vision Encoder ===")
    _print_single_vision(f"{base['label']} (cache_off)", base)
    _print_single_vision(f"{cached['label']} (whole_slot_reuse)", cached)
    print(_delta_line("Whole-slot total vision time", base["vision_total_ms"], cached["vision_total_ms"]))
    print(_delta_line("Whole-slot turn1 vision time", base["turn1_total_ms"], cached["turn1_total_ms"]))
    if partial is not None:
        _print_single_vision(f"{partial['label']} (token/window_partial_reuse)", partial)
        print(_delta_line("Partial total vision time", base["vision_total_ms"], partial["vision_total_ms"]))
        print(_delta_line("Partial turn1 vision time", base["turn1_total_ms"], partial["turn1_total_ms"]))


def _print_single_generate(label: str, row: dict) -> None:
    print(f"-- {label}")
    print(
        f"  rows={row['rows']}  turn1={row['turn1_rows']}  "
        f"e2e_total={row['e2e_total_ms']:.1f}ms  e2e_mean={row['e2e_mean_ms']:.2f}ms  "
        f"sglang_mean={row['sglang_mean_ms']:.2f}ms"
    )
    print(
        f"  turn1_e2e_total={row['turn1_e2e_total_ms']:.1f}ms  "
        f"turn1_e2e_mean={row['turn1_e2e_mean_ms']:.2f}ms"
    )


def _print_generate_report(base: dict, cached: dict, partial: dict | None = None) -> None:
    print("\n=== End To End Generate ===")
    _print_single_generate(f"{base['label']} (cache_off)", base)
    _print_single_generate(f"{cached['label']} (whole_slot_reuse)", cached)
    print(_delta_line("Whole-slot total generate_e2e", base["e2e_total_ms"], cached["e2e_total_ms"]))
    print(_delta_line("Whole-slot turn1 generate_e2e", base["turn1_e2e_total_ms"], cached["turn1_e2e_total_ms"]))
    print(_delta_line("Whole-slot total sglang_call", base["sglang_total_ms"], cached["sglang_total_ms"]))
    print(_delta_line("Whole-slot turn1 sglang_call", base["turn1_sglang_total_ms"], cached["turn1_sglang_total_ms"]))
    if partial is not None:
        _print_single_generate(f"{partial['label']} (token/window_partial_reuse)", partial)
        print(_delta_line("Partial total generate_e2e", base["e2e_total_ms"], partial["e2e_total_ms"]))
        print(_delta_line("Partial turn1 generate_e2e", base["turn1_e2e_total_ms"], partial["turn1_e2e_total_ms"]))
        print(_delta_line("Partial total sglang_call", base["sglang_total_ms"], partial["sglang_total_ms"]))
        print(_delta_line("Partial turn1 sglang_call", base["turn1_sglang_total_ms"], partial["turn1_sglang_total_ms"]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", nargs="+", help="Deprecated alias for --baseline-vision")
    p.add_argument("--cached", nargs="+", help="Deprecated alias for --cached-vision")
    p.add_argument("--baseline-vision", nargs="+")
    p.add_argument("--cached-vision", nargs="+")
    p.add_argument("--partial-vision", nargs="+")
    p.add_argument("--baseline-generate", nargs="+")
    p.add_argument("--cached-generate", nargs="+")
    p.add_argument("--partial-generate", nargs="+")
    args = p.parse_args()

    base_vision_paths = _expand_paths(args.baseline_vision or args.baseline)
    cached_vision_paths = _expand_paths(args.cached_vision or args.cached)
    partial_vision_paths = _expand_paths(args.partial_vision)
    base_gen_paths = _expand_paths(args.baseline_generate)
    cached_gen_paths = _expand_paths(args.cached_generate)
    partial_gen_paths = _expand_paths(args.partial_generate)

    if base_vision_paths and cached_vision_paths:
        base_rows = _load_many(base_vision_paths)
        cached_rows = _load_many(cached_vision_paths)
        partial = None
        if partial_vision_paths:
            partial_rows = _load_many(partial_vision_paths)
            partial = _summarize_vision(partial_rows, f"{len(partial_vision_paths)} vision file(s)")
        base = _summarize_vision(base_rows, f"{len(base_vision_paths)} vision file(s)")
        cached = _summarize_vision(cached_rows, f"{len(cached_vision_paths)} vision file(s)")
        _print_vision_report(base, cached, partial)

    if base_gen_paths and cached_gen_paths:
        base_rows = _load_many(base_gen_paths)
        cached_rows = _load_many(cached_gen_paths)
        partial = None
        if partial_gen_paths:
            partial_rows = _load_many(partial_gen_paths)
            partial = _summarize_generate(partial_rows, f"{len(partial_gen_paths)} generate file(s)")
        base = _summarize_generate(base_rows, f"{len(base_gen_paths)} generate file(s)")
        cached = _summarize_generate(cached_rows, f"{len(cached_gen_paths)} generate file(s)")
        _print_generate_report(base, cached, partial)

    if not ((base_vision_paths and cached_vision_paths) or (base_gen_paths and cached_gen_paths)):
        p.error("provide baseline/cached vision logs, generate logs, or both")


if __name__ == "__main__":
    main()
