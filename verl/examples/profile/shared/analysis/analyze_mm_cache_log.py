#!/usr/bin/env python3
"""Decompose SGLang multimodal-embedding-cache 0% reuse into root causes.

Reads per-replica cache trace files (mm_cache_log_<suffix>_<pid>.csv) emitted by
the instrumented MultiModalStaticCache (SGLANG_MM_CACHE_PROFILE=1), and classifies
why each cache GET missed:

  - cross_replica : first time THIS replica process sees this content hash. With a
                    per-process cache + load balancer, the n identical GRPO copies
                    get spread over replicas, so each replica cold-misses. A single
                    GLOBAL cache would have computed each hash once.
  - concurrent    : a get-miss for this hash while an earlier get-miss on the same
                    replica has not yet completed its set (burst get-before-set).
                    These are the GRPO n-way duplicates co-scheduled in one window.
  - lru_evicted   : this replica had set this hash before, but it was evicted
                    (capacity too small) before being reused.
  - anomaly       : miss while hash is supposedly present (should not happen).

Per-replica event replay (events sorted by timestamp):
  present  : hashes currently resident (add on set_ok, remove on evict)
  ever_set : hashes set at least once on this replica
  pending  : hashes with an outstanding get-miss but no set_ok yet

Usage:
  python3 examples/profile/analyze_mm_cache_log.py \
    --log-dir profile_logs_refocus_chart --suffix refocus_chart_multiturn_bs64_n4
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import Counter, defaultdict


def load_events(log_dir: str, suffix: str) -> dict[str, list[dict]]:
    pattern = os.path.join(log_dir, f"mm_cache_log_{suffix}_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No cache trace files matched {pattern}")
    by_pid: dict[str, list[dict]] = defaultdict(list)
    for fp in files:
        with open(fp, newline="") as f:
            for row in csv.DictReader(f):
                row["timestamp"] = float(row["timestamp"])
                by_pid[row["pid"]].append(row)
    for pid in by_pid:
        by_pid[pid].sort(key=lambda r: r["timestamp"])
    return by_pid, files


def classify(by_pid: dict[str, list[dict]]):
    miss_classes = Counter()
    n_get = n_hit = n_miss = n_evict = 0
    hash_to_pids_missed: dict[str, set] = defaultdict(set)

    for pid, events in by_pid.items():
        present: set = set()
        ever_set: set = set()
        pending: set = set()
        for ev in events:
            etype = ev["event"]
            h = ev["combined_hash"]
            if etype == "get":
                n_get += 1
                if ev["hit"] == "1":
                    n_hit += 1
                    continue
                n_miss += 1
                if h in present:
                    miss_classes["anomaly"] += 1
                elif h in pending:
                    miss_classes["concurrent"] += 1
                elif h in ever_set:
                    miss_classes["lru_evicted"] += 1
                else:
                    miss_classes["cross_replica"] += 1
                    hash_to_pids_missed[h].add(pid)
                pending.add(h)
            elif etype == "set_ok":
                present.add(h)
                ever_set.add(h)
                pending.discard(h)
            elif etype == "set_exists":
                present.add(h)
                ever_set.add(h)
                pending.discard(h)
            elif etype == "evict":
                n_evict += 1
                present.discard(h)
            elif etype == "set_fail":
                pending.discard(h)

    return {
        "n_get": n_get,
        "n_hit": n_hit,
        "n_miss": n_miss,
        "n_evict": n_evict,
        "miss_classes": miss_classes,
        "hash_to_pids_missed": hash_to_pids_missed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--suffix", required=True)
    args = ap.parse_args()

    by_pid, files = load_events(args.log_dir, args.suffix)
    stats = classify(by_pid)

    n_get = stats["n_get"]
    n_hit = stats["n_hit"]
    n_miss = stats["n_miss"]
    mc = stats["miss_classes"]
    h2p = stats["hash_to_pids_missed"]

    distinct_hashes = len(h2p)
    cross = mc.get("cross_replica", 0)
    # Redundant cold misses vs an ideal single global cache (which computes each hash once).
    cross_replica_redundant = max(0, cross - distinct_hashes)
    multi_pid_hashes = sum(1 for pids in h2p.values() if len(pids) > 1)

    print(f"trace files: {len(files)} (replicas/pids: {len(by_pid)})")
    print(f"GET total={n_get}  hit={n_hit}  miss={n_miss}  hit_rate={ (n_hit/n_get*100) if n_get else 0:.2f}%")
    print(f"evictions={stats['n_evict']}")
    print("\nMISS breakdown (root cause of 0 reuse):")
    for k in ("cross_replica", "concurrent", "lru_evicted", "anomaly"):
        v = mc.get(k, 0)
        pct = (v / n_miss * 100) if n_miss else 0
        print(f"  {k:14s}: {v:5d}  ({pct:5.1f}% of misses)")

    print("\nCross-replica detail:")
    print(f"  distinct content hashes (cold-missed): {distinct_hashes}")
    print(f"  hashes computed on >1 replica:         {multi_pid_hashes}")
    print(f"  redundant computes vs 1 global cache:  {cross_replica_redundant}")
    print(
        "  -> a single shared (global) embedding cache would cut cold computes from "
        f"{cross} to ~{distinct_hashes}."
    )

    if n_hit == 0:
        print("\nNOTE: 0 hits confirms the per-process LRU cache provided no reuse in this rollout.")


if __name__ == "__main__":
    main()
