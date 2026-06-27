#!/usr/bin/env python3
"""Sanity-check trainer rollout JSONL before positive discovery.

Checks whether the dump has the fields needed for branch-level parity:
request ids, group uid, rollout_idx, response text, and score aliases. Optional
refocus_chart recomputation verifies whether reward aligns with response_text
or the full decoded output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(path.glob("*.jsonl"))


def _rows(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row["_file"] = str(path)
                    out.append(row)
    return out


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _present(row: dict[str, Any], *keys: str) -> bool:
    return any(row.get(key) not in (None, "") for key in keys)


def _maybe_score(text: str, gt: str, enabled: bool) -> float:
    if not enabled or not gt:
        return -1.0
    from verl.utils.reward_score.refocus_chart import compute_score

    return float(compute_score(text, gt))


def _mean_abs(vals: list[float]) -> float:
    return sum(abs(v) for v in vals) / len(vals) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout-data", required=True, type=Path)
    ap.add_argument("--verify-refocus-score", action="store_true")
    ap.add_argument("--sample", type=int, default=5)
    args = ap.parse_args()

    paths = _paths(args.rollout_data)
    rows = _rows(paths)
    if not rows:
        raise SystemExit(f"no rows in {args.rollout_data}")

    required = {
        "agent_uid": ("agent_uid", "uid"),
        "request_id": ("request_id", "agent_request_id"),
        "rollout_idx": ("rollout_idx", "index"),
        "response_text": ("final_response_text", "response_text", "vtool_final_response_text"),
        "score": ("compute_score", "reward_score", "score"),
        "ground_truth": ("gts",),
    }
    missing = {
        name: sum(1 for row in rows if not _present(row, *keys))
        for name, keys in required.items()
    }

    score_alias_deltas: list[float] = []
    response_score_deltas: list[float] = []
    output_score_deltas: list[float] = []
    positives = 0
    examples: list[dict[str, Any]] = []
    for row in rows:
        scores = [
            _f(row[key])
            for key in ("compute_score", "reward_score", "score")
            if row.get(key) not in (None, "")
        ]
        if scores:
            positives += int(max(scores) > 0.0)
        if len(scores) >= 2:
            score_alias_deltas.append(max(scores) - min(scores))

        logged_score = scores[0] if scores else 0.0
        gt = str(row.get("gts") or "")
        response_text = str(
            row.get("final_response_text")
            or row.get("response_text")
            or row.get("vtool_final_response_text")
            or ""
        )
        output_text = str(row.get("output") or "")
        response_reward = _maybe_score(response_text, gt, args.verify_refocus_score)
        output_reward = _maybe_score(output_text, gt, args.verify_refocus_score)
        if response_reward >= 0:
            response_score_deltas.append(response_reward - logged_score)
        if output_reward >= 0:
            output_score_deltas.append(output_reward - logged_score)
        if len(examples) < args.sample and logged_score > 0:
            examples.append(
                {
                    "file": row.get("_file"),
                    "agent_uid": row.get("agent_uid") or row.get("uid"),
                    "request_id": row.get("request_id") or row.get("agent_request_id"),
                    "rollout_idx": row.get("rollout_idx") or row.get("index"),
                    "score": logged_score,
                    "response_reward": response_reward,
                    "output_reward": output_reward,
                    "response_text_prefix": response_text[:200],
                    "output_prefix": output_text[:200],
                }
            )

    result = {
        "rollout_data": str(args.rollout_data),
        "files": len(paths),
        "rows": len(rows),
        "positive_rows": positives,
        "missing_fields": missing,
        "score_alias_delta_max": max(score_alias_deltas) if score_alias_deltas else 0.0,
        "score_alias_delta_mean_abs": _mean_abs(score_alias_deltas),
        "response_text_score_delta_mean_abs": _mean_abs(response_score_deltas),
        "output_score_delta_mean_abs": _mean_abs(output_score_deltas),
        "positive_examples": examples,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
