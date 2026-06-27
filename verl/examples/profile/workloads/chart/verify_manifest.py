#!/usr/bin/env python3
"""Sanity-check VTool Chart image dump after refocus rollout."""

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
    debug_rows = []
    for r in rows:
        p = dump_dir / r["path"]
        if not p.exists():
            missing_png += 1
        uid = r["uid"]
        branch = r.get("request_id") or str(r.get("rollout_idx", "?"))
        turn = int(r["turn"])
        roles[r.get("role", "?")] += 1
        by_branch[(uid, branch)][turn] = r
        if "tool_success" in r or "changed_pixel_frac" in r:
            debug_rows.append(r)

    uids = {uid for uid, _ in by_branch}
    branches_per_uid = Counter(uid for uid, _ in by_branch)
    turn0_branches = Counter()
    turn1_refocus = 0
    for (uid, branch), turns in by_branch.items():
        if 0 in turns:
            turn0_branches[uid] += 1
        if any(row.get("role") == "refocus_output" for row in turns.values()):
            turn1_refocus += 1

    print(f"records={len(rows)} uids={len(uids)} branches={len(by_branch)} missing_png={missing_png}")
    print(f"roles: {dict(roles)}")
    if branches_per_uid:
        counts = list(branches_per_uid.values())
        print(f"branches/uid: min={min(counts)} median={sorted(counts)[len(counts)//2]} max={max(counts)}")
    print(f"uids with turn0 chart_input: {len(turn0_branches)}")
    print(f"branches with refocus_output: {turn1_refocus}")
    if turn0_branches:
        sample = dict(list(turn0_branches.items())[:3])
        print(f"turn0 branches per uid (sample): {sample}")

    same_branch_same = 0
    same_branch_diff = 0
    t1_hashes_by_uid = defaultdict(list)
    for (uid, branch), turns in by_branch.items():
        if 1 in turns:
            p1 = dump_dir / turns[1]["path"]
            if p1.exists():
                t1_hashes_by_uid[uid].append(hashlib.sha256(p1.read_bytes()).hexdigest())
        if 0 not in turns or 1 not in turns:
            continue
        p0 = dump_dir / turns[0]["path"]
        p1 = dump_dir / turns[1]["path"]
        if not p0.exists() or not p1.exists():
            continue
        if hashlib.sha256(p0.read_bytes()).hexdigest() == hashlib.sha256(p1.read_bytes()).hexdigest():
            same_branch_same += 1
        else:
            same_branch_diff += 1
    if same_branch_same or same_branch_diff:
        print(f"same-branch t0->t1: identical={same_branch_same} changed={same_branch_diff}")
    if t1_hashes_by_uid:
        distinct_dist = Counter(len(set(v)) for v in t1_hashes_by_uid.values())
        print(f"distinct turn1 per uid: {dict(sorted(distinct_dist.items()))}")

    if debug_rows:
        success = Counter(bool(r.get("tool_success")) for r in debug_rows)
        unchanged = sum(1 for r in debug_rows if r.get("hash_equal") is True)
        changed_fracs = [float(r["changed_pixel_frac"]) for r in debug_rows if "changed_pixel_frac" in r]
        print(f"debug rows: {len(debug_rows)} tool_success={dict(success)} hash_equal={unchanged}")
        if changed_fracs:
            print(
                "changed_pixel_frac: "
                f"min={min(changed_fracs):.6f} "
                f"mean={sum(changed_fracs) / len(changed_fracs):.6f} "
                f"max={max(changed_fracs):.6f}"
            )


if __name__ == "__main__":
    main()
