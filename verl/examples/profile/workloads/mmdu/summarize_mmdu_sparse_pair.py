#!/usr/bin/env python3
"""Gate one real MMDU dense/query-sparse pair on speed and task reward."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


STEP_RE = re.compile(r"training/global_step:(\d+)")
PROFILE_PREFIX = "VERL_ROLLOUT_PROFILE_STEP "
RAY_SESSION_RE = re.compile(r"/[^\s]+/ray/session_[0-9_-]+")


def _metric(line: str, name: str) -> float | None:
    match = re.search(rf"(?:^|\s){re.escape(name)}:([-+0-9.eE]+)(?:\s|$)", line)
    return float(match.group(1)) if match else None


def _profile_step(line: str) -> tuple[int, dict[str, float]] | None:
    if PROFILE_PREFIX not in line:
        return None
    payload = line.split(PROFILE_PREFIX, 1)[1]
    values, _ = json.JSONDecoder().raw_decode(payload.lstrip())
    return int(values["global_step"]), {
        "timing_s/gen": float(values["rollout_s"]),
        "timing_s/step": float(values["rollout_s"]),
        "critic/score/mean": float(values["reward_mean"]),
        "response/aborted_ratio": float(values["aborted_ratio"]),
        "response_length/mean": float(values["response_length_mean"]),
    }


def _read_steps(path: Path) -> list[dict[str, float]]:
    profile_steps: dict[int, dict[str, float]] = {}
    by_step: dict[int, dict[str, float]] = {}
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        profile = _profile_step(line)
        if profile is not None:
            step, values = profile
            profile_steps[step] = values
            continue
        step_match = STEP_RE.search(line)
        if step_match is None or "timing_s/gen:" not in line:
            continue
        step = int(step_match.group(1))
        values = {
            name: value
            for name in (
                "timing_s/gen",
                "timing_s/step",
                "critic/score/mean",
                "response/aborted_ratio",
                "response_length/mean",
            )
            if (value := _metric(line, name)) is not None
        }
        by_step[step] = values
    # Ray may retain a worker's print in worker-*.out without forwarding it to the
    # driver/tee stream. The launch log contains the exact Ray session directory in
    # dashboard and filesystem-monitor messages, so recover only that arm's events.
    if not profile_steps:
        session_dirs = {Path(value) for value in RAY_SESSION_RE.findall(text)}
        for session_dir in session_dirs:
            for worker_log in (session_dir / "logs").glob("worker-*.out"):
                for line in worker_log.read_text(errors="replace").splitlines():
                    profile = _profile_step(line)
                    if profile is not None:
                        step, values = profile
                        profile_steps[step] = values
    if profile_steps:
        return [profile_steps[step] for step in sorted(profile_steps)]
    if not by_step:
        raise ValueError(f"no completed trainer steps in {path}")
    return [by_step[step] for step in sorted(by_step)]


def _summarize(path: Path) -> dict[str, float]:
    steps = _read_steps(path)
    return {
        "steps": float(len(steps)),
        "rollout_s": sum(step["timing_s/gen"] for step in steps),
        "step_s": sum(step["timing_s/step"] for step in steps),
        "reward_mean": sum(step.get("critic/score/mean", 0.0) for step in steps)
        / len(steps),
        "aborted_ratio_max": max(
            step.get("response/aborted_ratio", 0.0) for step in steps
        ),
        "response_length_mean": sum(
            step.get("response_length/mean", 0.0) for step in steps
        )
        / len(steps),
    }


def _sparse_counters(path: Path) -> dict[str, object]:
    counters: dict[str, object] = {
        "rows": 0,
        "kept_tokens": 0,
        "dropped_tokens": 0,
        "direct_source_rows": 0,
        "incremental_append_rows": 0,
        "decode_local_rows": 0,
        "max_proxy_error_bound": 0.0,
    }
    fallback_reasons: Counter[str] = Counter()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("mode") == "DECODE" and row.get("reuse_action") == "local":
                counters["decode_local_rows"] += 1
                fallback_reasons[row.get("reuse_reason") or "unspecified"] += 1
            if row.get("cacheblend_sparse_decode_used") != "1":
                continue
            counters["rows"] += 1
            counters["kept_tokens"] += int(
                row.get("cacheblend_sparse_decode_kept_tokens") or 0
            )
            counters["dropped_tokens"] += int(
                row.get("cacheblend_sparse_decode_dropped_tokens") or 0
            )
            counters["direct_source_rows"] += int(
                row.get("cacheblend_sparse_decode_direct_source") or 0
            )
            counters["incremental_append_rows"] += int(
                row.get("cacheblend_sparse_decode_incremental_append") or 0
            )
            error_bound = row.get("reuse_error_bound")
            if error_bound not in (None, ""):
                counters["max_proxy_error_bound"] = max(
                    float(counters["max_proxy_error_bound"]), float(error_bound)
                )
    counters["fallback_reasons"] = dict(sorted(fallback_reasons.items()))
    return counters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-log", type=Path, required=True)
    parser.add_argument("--sparse-log", type=Path, required=True)
    parser.add_argument("--sparse-forward-csv", type=Path, required=True)
    parser.add_argument("--min-rollout-speedup", type=float, default=0.05)
    parser.add_argument("--min-token-throughput-speedup", type=float, default=0.0)
    parser.add_argument("--max-reward-drop", type=float, default=0.01)
    parser.add_argument("--max-response-length-drift", type=float, default=0.05)
    args = parser.parse_args()

    dense = _summarize(args.dense_log)
    sparse = _summarize(args.sparse_log)
    counters = _sparse_counters(args.sparse_forward_csv)
    if dense["rollout_s"] <= 0 or dense["response_length_mean"] <= 0:
        raise ValueError("dense metrics must have positive rollout and response length")
    rollout_speedup = 1.0 - sparse["rollout_s"] / dense["rollout_s"]
    reward_delta = sparse["reward_mean"] - dense["reward_mean"]
    response_length_drift = abs(
        sparse["response_length_mean"] / dense["response_length_mean"] - 1.0
    )
    token_throughput_speedup = (
        (sparse["response_length_mean"] / sparse["rollout_s"])
        / (dense["response_length_mean"] / dense["rollout_s"])
        - 1.0
    )
    passed = (
        sparse["steps"] == dense["steps"]
        and rollout_speedup >= args.min_rollout_speedup
        and token_throughput_speedup >= args.min_token_throughput_speedup
        and reward_delta >= -args.max_reward_drop
        and response_length_drift <= args.max_response_length_drift
        and sparse["aborted_ratio_max"] == 0.0
        and counters["dropped_tokens"] > 0
    )
    result = {
        "dense": dense,
        "sparse": sparse,
        "sparse_counters": counters,
        "rollout_speedup": rollout_speedup,
        "reward_delta": reward_delta,
        "response_length_drift": response_length_drift,
        "token_throughput_speedup": token_throughput_speedup,
        "thresholds": {
            "min_rollout_speedup": args.min_rollout_speedup,
            "min_token_throughput_speedup": args.min_token_throughput_speedup,
            "max_reward_drop": args.max_reward_drop,
            "max_response_length_drift": args.max_response_length_drift,
        },
        "passed": passed,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
