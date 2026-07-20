from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[2]
    / "examples/profile/workloads/mmdu/summarize_mmdu_sparse_pair.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_mmdu_sparse_pair", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def test_rollout_only_json_is_preferred_over_progress_text(tmp_path):
    log = tmp_path / "launch.log"
    rows = [
        {
            "global_step": 1,
            "rollout_s": 12.5,
            "reward_mean": 0.7,
            "aborted_ratio": 0.0,
            "response_length_mean": 900.0,
        },
        {
            "global_step": 2,
            "rollout_s": 10.0,
            "reward_mean": 0.5,
            "aborted_ratio": 0.0,
            "response_length_mean": 1000.0,
        },
    ]
    log.write_text(
        "\n".join(
            "ray-prefix VERL_ROLLOUT_PROFILE_STEP " + json.dumps(row) + " suffix"
            for row in rows
        )
    )

    result = SUMMARY._summarize(log)
    assert result["steps"] == 2.0
    assert result["rollout_s"] == 22.5
    assert result["reward_mean"] == 0.6
    assert result["response_length_mean"] == 950.0


def test_sparse_counter_aggregation(tmp_path):
    path = tmp_path / "forward.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "cacheblend_sparse_decode_used",
                "cacheblend_sparse_decode_kept_tokens",
                "cacheblend_sparse_decode_dropped_tokens",
                "cacheblend_sparse_decode_direct_source",
                "cacheblend_sparse_decode_incremental_append",
                "mode",
                "reuse_action",
                "reuse_reason",
                "reuse_error_bound",
            ),
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "cacheblend_sparse_decode_used": "1",
                    "cacheblend_sparse_decode_kept_tokens": "100",
                    "cacheblend_sparse_decode_dropped_tokens": "80",
                    "cacheblend_sparse_decode_direct_source": "1",
                    "cacheblend_sparse_decode_incremental_append": "0",
                    "mode": "DECODE",
                    "reuse_action": "skip",
                    "reuse_reason": "query_ranked_context_blocks",
                    "reuse_error_bound": "0.04",
                },
                {
                    "cacheblend_sparse_decode_used": "0",
                    "cacheblend_sparse_decode_kept_tokens": "100",
                    "cacheblend_sparse_decode_dropped_tokens": "0",
                    "cacheblend_sparse_decode_direct_source": "0",
                    "cacheblend_sparse_decode_incremental_append": "0",
                    "mode": "DECODE",
                    "reuse_action": "local",
                    "reuse_reason": "drop_upper_bound_below_floor",
                    "reuse_error_bound": "",
                },
                {
                    "cacheblend_sparse_decode_used": "1",
                    "cacheblend_sparse_decode_kept_tokens": "90",
                    "cacheblend_sparse_decode_dropped_tokens": "70",
                    "cacheblend_sparse_decode_direct_source": "1",
                    "cacheblend_sparse_decode_incremental_append": "1",
                    "mode": "DECODE",
                    "reuse_action": "skip",
                    "reuse_reason": "query_ranked_context_blocks",
                    "reuse_error_bound": "0.03",
                },
            ]
        )

    assert SUMMARY._sparse_counters(path) == {
        "rows": 2,
        "kept_tokens": 190,
        "dropped_tokens": 150,
        "direct_source_rows": 2,
        "incremental_append_rows": 1,
        "decode_local_rows": 1,
        "max_proxy_error_bound": 0.04,
        "fallback_reasons": {"drop_upper_bound_below_floor": 1},
    }


def test_profile_json_falls_back_to_exact_ray_session(tmp_path):
    session = tmp_path / "ray" / "session_2026_07_20_01_02_03_123"
    logs = session / "logs"
    logs.mkdir(parents=True)
    launch = tmp_path / "launch.log"
    launch.write_text(f"file monitor: {session} is over threshold\n")
    (logs / "worker-task.out").write_text(
        "VERL_ROLLOUT_PROFILE_STEP "
        + json.dumps(
            {
                "global_step": 1,
                "rollout_s": 20.0,
                "reward_mean": 0.4,
                "aborted_ratio": 0.0,
                "response_length_mean": 800.0,
            }
        )
        + "\n"
    )

    assert SUMMARY._summarize(launch)["rollout_s"] == 20.0
