#!/usr/bin/env python3
"""Sanity-check Sokoban image dump after verl-agent rollout."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    args = parser.parse_args()
    dump_dir = args.dump_dir
    manifest = dump_dir / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"missing {manifest}")

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_branch: dict[tuple[int, int], list[int]] = defaultdict(list)
    missing_png = 0
    for r in rows:
        p = dump_dir / r["path"]
        if not p.exists():
            missing_png += 1
        by_branch[(r["group_idx"], r["branch_idx"])].append(r["step"])

    step_counts = [len(set(steps)) for steps in by_branch.values()]
    print(f"records={len(rows)} branches={len(by_branch)} missing_png={missing_png}")
    if step_counts:
        print(f"steps/branch: min={min(step_counts)} median={sorted(step_counts)[len(step_counts)//2]} max={max(step_counts)}")
    # step0 cross-branch: should exist for each group
    groups = Counter(r["group_idx"] for r in rows if r["step"] == 0)
    branches_per_group = Counter()
    for (g, b), steps in by_branch.items():
        if 0 in steps:
            branches_per_group[g] += 1
    print(f"groups with step0 obs: {len(groups)}")
    print(f"branches@step0 per group (sample): {dict(list(branches_per_group.items())[:5])}")


if __name__ == "__main__":
    main()
