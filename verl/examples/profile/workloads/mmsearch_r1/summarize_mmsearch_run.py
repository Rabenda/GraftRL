#!/usr/bin/env python3
"""Print the small set of signals needed to accept/reject an MMSearch run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _number(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := _number(row, key)) is not None]
    return mean(values) if values else None


def _load_jsonl(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _latest_step(rows: list[dict[str, Any]], key: str = "step") -> list[dict[str, Any]]:
    steps = [value for row in rows if (value := _number(row, key)) is not None]
    if not steps:
        return rows
    latest = max(steps)
    return [row for row in rows if _number(row, key) == latest]


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-data", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--suffix", required=True)
    args = parser.parse_args()

    rollout_rows = _latest_step(_load_jsonl(args.rollout_data))
    candidate_total = sum(_number(row, "mmsearch_context_candidate_count") or 0 for row in rollout_rows)
    selected_total = sum(_number(row, "mmsearch_context_selected_count") or 0 for row in rollout_rows)
    context_samples = sum((_number(row, "mmsearch_context_candidate_count") or 0) > 0 for row in rollout_rows)
    retrieval_hits = sum(_number(row, "mmsearch_retrieval_cache_hits") or 0 for row in rollout_rows)
    selection_hits = sum(_number(row, "mmsearch_selection_cache_hits") or 0 for row in rollout_rows)
    token_hits = sum(_number(row, "mmsearch_tokenization_cache_hits") or 0 for row in rollout_rows)
    exact_reuse = sum(_number(row, "mmsearch_exact_reuse_count") or 0 for row in rollout_rows)
    local_compute = sum(_number(row, "mmsearch_local_compute_count") or 0 for row in rollout_rows)
    skipped = sum(_number(row, "mmsearch_skip_count") or 0 for row in rollout_rows)
    reduction = 100 * (1 - selected_total / candidate_total) if candidate_total else None
    reduction_text = "n/a" if reduction is None else f"{reduction:.1f}%"

    vision_rows = _latest_step(
        _load_csv(args.log_dir / f"vision_encoder_log_{args.suffix}.csv"),
        key="global_step",
    )
    generate_rows = _latest_step(
        _load_csv(args.log_dir / f"verl_sglang_generate_log_{args.suffix}.csv"),
        key="global_step",
    )
    model_rows = _latest_step(
        _load_csv(args.log_dir / f"model_forward_log_{args.suffix}.csv"),
        key="global_step",
    )
    followup_rows = [row for row in generate_rows if _number(row, "agent_turn") == 1]

    worker_owners: dict[str, set[str]] = defaultdict(set)
    for row in rollout_rows:
        group_id = str(row.get("uid", ""))
        worker_pid = str(row.get("agent_worker_pid", ""))
        if group_id and worker_pid:
            worker_owners[group_id].add(worker_pid)
    replica_owners: dict[str, set[str]] = defaultdict(set)
    for row in generate_rows:
        group_id = str(row.get("agent_uid", ""))
        replica = str(row.get("replica_rank", ""))
        if group_id and replica:
            replica_owners[group_id].add(replica)
    split_workers = sum(len(owners) > 1 for owners in worker_owners.values())
    split_replicas = sum(len(owners) > 1 for owners in replica_owners.values())

    cache_counter_names = (
        "entries",
        "current_size",
        "hits",
        "misses",
        "sets",
        "evictions",
        "rejections",
        "same_group_hits",
        "cross_group_hits",
        "cross_branch_hits",
        "unattributed_hits",
        "coalesced_items",
    )
    cache_by_pid: dict[str, dict[str, float]] = defaultdict(dict)
    for row in model_rows:
        pid = str(row.get("pid", "unknown"))
        for name in cache_counter_names:
            value = _number(row, f"mm_embedding_cache_{name}")
            if value is not None:
                cache_by_pid[pid][name] = max(cache_by_pid[pid].get(name, 0), value)
    cache_totals = {
        name: sum(values.get(name, 0) for values in cache_by_pid.values())
        for name in cache_counter_names
    }

    print("MMSearch latest-step acceptance summary")
    print(f"  rollout samples: {len(rollout_rows)}")
    print(
        "  static context: "
        f"{int(candidate_total)} candidates -> {int(selected_total)} selected "
        f"({reduction_text} reduction)"
    )
    print(
        "  execution actions: "
        f"exact={int(exact_reuse)}, local={int(local_compute)}, skip_items={int(skipped)}"
    )
    print(
        "  stage cache hits: "
        f"retrieval={int(retrieval_hits)}/{context_samples}, "
        f"tokenization={int(token_hits)}/{context_samples}; "
        f"selection-policy-cache={int(selection_hits)}/{context_samples}"
    )
    print(
        "  group routing: "
        f"workers={len(worker_owners)} groups/{split_workers} split, "
        f"replicas={len(replica_owners)} groups/{split_replicas} split"
    )
    if cache_by_pid:
        print(
            "  encoder cache: "
            f"hits={int(cache_totals['hits'])}, misses={int(cache_totals['misses'])}, "
            f"sets={int(cache_totals['sets'])}, entries={int(cache_totals['entries'])}, "
            f"same_group={int(cache_totals['same_group_hits'])}, "
            f"cross_group={int(cache_totals['cross_group_hits'])}, "
            f"cross_branch={int(cache_totals['cross_branch_hits'])}, "
            f"coalesced={int(cache_totals['coalesced_items'])}, "
            f"evictions={int(cache_totals['evictions'])}, rejections={int(cache_totals['rejections'])}"
        )
    print(
        "  quality: "
        f"reward={_fmt(_mean(rollout_rows, 'score'))}, "
        f"accuracy={_fmt(_mean(rollout_rows, 'accuracy'))}, "
        f"format={_fmt(_mean(rollout_rows, 'format_score'))}"
    )
    print(
        "  encoder work: "
        f"calls={len(vision_rows)}, images={int(sum(_number(row, 'image_count') or 0 for row in vision_rows))}, "
        f"vision_ms={_fmt(sum(_number(row, 'vision_encoder_time_ms') or 0 for row in vision_rows), 1)}"
    )
    print(
        "  turn1: "
        f"prompt_tokens={_fmt(_mean(followup_rows, 'prompt_tokens'), 1)}, "
        f"image_tokens={_fmt(_mean(followup_rows, 'image_prompt_tokens'), 1)}, "
        f"request_e2e_ms={_fmt(_mean(followup_rows, 'generate_e2e_ms'), 1)}"
    )


if __name__ == "__main__":
    main()
