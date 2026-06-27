#!/usr/bin/env python3
"""Sanity-check DeepEyes image dump after zoom rollout."""

from __future__ import annotations

import argparse
import hashlib
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
    by_branch: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    missing_png = 0
    roles = Counter()
    for r in rows:
        p = dump_dir / r["path"]
        if not p.exists():
            missing_png += 1
        uid = r["uid"]
        branch = r.get("request_id") or str(r.get("rollout_idx", "?"))
        turn = int(r["turn"])
        roles[r.get("role", "?")] += 1
        by_branch[(uid, branch)][turn] = r

    uids = {uid for uid, _ in by_branch}
    branches_per_uid = Counter(uid for uid, _ in by_branch)
    turn0_branches = Counter()
    zoom_outputs = 0
    for uid, branch in by_branch:
        if 0 in by_branch[(uid, branch)]:
            turn0_branches[uid] += 1
    for turns in by_branch.values():
        if any(row.get("role") == "zoom_output" for row in turns.values()):
            zoom_outputs += 1

    print(f"records={len(rows)} uids={len(uids)} branches={len(by_branch)} missing_png={missing_png}")
    print(f"roles: {dict(roles)}")
    if branches_per_uid:
        counts = list(branches_per_uid.values())
        print(f"branches/uid: min={min(counts)} median={sorted(counts)[len(counts)//2]} max={max(counts)}")
    print(f"uids with turn0 deepeyes_input: {len(turn0_branches)}")
    print(f"branches with zoom_output: {zoom_outputs}")

    same_branch_same = 0
    same_branch_diff = 0
    t1_hashes_by_uid: dict[str, list[str]] = defaultdict(list)
    for (uid, branch), turns in by_branch.items():
        zoom_turns = sorted(t for t, row in turns.items() if row.get("role") == "zoom_output")
        for t in zoom_turns:
            p = dump_dir / turns[t]["path"]
            if p.exists():
                t1_hashes_by_uid[uid].append(hashlib.sha256(p.read_bytes()).hexdigest())
        if 0 not in turns or not zoom_turns:
            continue
        p0 = dump_dir / turns[0]["path"]
        p1 = dump_dir / turns[zoom_turns[0]]["path"]
        if not p0.exists() or not p1.exists():
            continue
        if hashlib.sha256(p0.read_bytes()).hexdigest() == hashlib.sha256(p1.read_bytes()).hexdigest():
            same_branch_same += 1
        else:
            same_branch_diff += 1
    if same_branch_same or same_branch_diff:
        print(f"same-branch t0->first_zoom: identical={same_branch_same} changed={same_branch_diff}")
    if t1_hashes_by_uid:
        distinct_dist = Counter(len(set(v)) for v in t1_hashes_by_uid.values())
        print(f"distinct first-zoom per uid: {dict(sorted(distinct_dist.items()))}")


if __name__ == "__main__":
    main()
