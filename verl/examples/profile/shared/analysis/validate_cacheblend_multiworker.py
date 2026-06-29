#!/usr/bin/env python3
"""Offline checks for multi-worker VLM CacheBlend runs.

This script only inspects CSV logs. It does not require GPUs or a running Ray
cluster.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _path(log_dir: Path, stem: str, suffix: str) -> Path:
    name = f"{stem}_{suffix}.csv" if suffix else f"{stem}.csv"
    return log_dir / name


def _nonempty(values: Iterable[str]) -> set[str]:
    return {str(v).strip() for v in values if str(v).strip()}


def validate(args: argparse.Namespace) -> tuple[list[str], dict[str, object]]:
    log_dir = Path(args.log_dir)
    barrier_path = _path(log_dir, "cacheblend_barrier_log", args.suffix)
    model_path = _path(log_dir, "model_forward_log", args.suffix)
    generate_path = _path(log_dir, "verl_sglang_generate_log", args.suffix)

    barrier_rows = _read_csv(barrier_path)
    model_rows = _read_csv(model_path)
    generate_rows = _read_csv(generate_path)

    failures: list[str] = []
    report: dict[str, object] = {
        "barrier_path": str(barrier_path),
        "model_forward_path": str(model_path),
        "generate_path": str(generate_path),
        "barrier_rows": len(barrier_rows),
        "model_forward_rows": len(model_rows),
        "generate_rows": len(generate_rows),
    }

    if not barrier_rows:
        failures.append(f"missing or empty barrier log: {barrier_path}")
        return failures, report

    target_rows = [
        row
        for row in barrier_rows
        if row.get("warmup_key", "").strip()
        and row.get("barrier_role") in {"donor", "recipient"}
        and (not args.only_enabled or _truthy(row.get("barrier_enabled", "0")))
    ]
    by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in target_rows:
        by_key[row["warmup_key"]].append(row)

    report["warmup_groups"] = len(by_key)
    if len(by_key) < args.min_groups:
        failures.append(f"warmup groups {len(by_key)} < --min-groups {args.min_groups}")

    role_counts = Counter(row.get("barrier_role", "") for row in target_rows)
    report["barrier_role_counts"] = dict(role_counts)

    bad_donor_counts: dict[str, int] = {}
    bad_server_keys: dict[str, list[str]] = {}
    bad_routing_keys: dict[str, list[str]] = {}
    not_ready_recipients: list[str] = []
    for key, rows in by_key.items():
        donor_count = sum(1 for row in rows if row.get("barrier_role") == "donor")
        if donor_count != 1:
            bad_donor_counts[key] = donor_count

        server_ids = sorted(_nonempty(row.get("server_id", "") for row in rows))
        if len(server_ids) > 1:
            bad_server_keys[key] = server_ids

        routing_ids = sorted(_nonempty(row.get("routing_request_id", "") for row in rows))
        if len(routing_ids) > 1:
            bad_routing_keys[key] = routing_ids

        for row in rows:
            if row.get("barrier_role") == "recipient" and not _truthy(row.get("donor_ready", "0")):
                not_ready_recipients.append(row.get("request_id", ""))

    report["bad_donor_counts"] = bad_donor_counts
    report["bad_server_keys"] = bad_server_keys
    report["bad_routing_keys"] = bad_routing_keys
    report["not_ready_recipients"] = not_ready_recipients

    if bad_donor_counts:
        failures.append(f"{len(bad_donor_counts)} warmup groups do not have exactly one donor")
    if bad_server_keys:
        failures.append(f"{len(bad_server_keys)} warmup groups routed to multiple server_id values")
    if bad_routing_keys:
        failures.append(f"{len(bad_routing_keys)} warmup groups used multiple routing_request_id values")
    if not_ready_recipients:
        failures.append(f"{len(not_ready_recipients)} recipients proceeded without donor_ready")

    cacheblend_rows = [row for row in model_rows if row.get("cacheblend_role")]
    cb_role_counts = Counter(row.get("cacheblend_role", "") for row in cacheblend_rows)
    fallback_counts = Counter(row.get("cacheblend_fallback_reason", "") for row in cacheblend_rows)
    recipient_rows = [row for row in cacheblend_rows if row.get("cacheblend_role") == "recipient"]
    recipient_used = [row for row in recipient_rows if _truthy(row.get("cacheblend_used", "0"))]
    used_rate = len(recipient_used) / len(recipient_rows) if recipient_rows else 0.0
    donor_not_ready = fallback_counts.get("donor_not_ready", 0)

    report["cacheblend_role_counts"] = dict(cb_role_counts)
    report["cacheblend_fallback_counts"] = dict(fallback_counts)
    report["recipient_used_rate"] = used_rate
    report["donor_not_ready"] = donor_not_ready

    if donor_not_ready > args.max_donor_not_ready:
        failures.append(
            f"donor_not_ready {donor_not_ready} > --max-donor-not-ready {args.max_donor_not_ready}"
        )
    if used_rate < args.min_recipient_used_rate:
        failures.append(
            f"recipient used rate {used_rate:.4f} < --min-recipient-used-rate {args.min_recipient_used_rate:.4f}"
        )

    return failures, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--min-groups", type=int, default=1)
    parser.add_argument("--max-donor-not-ready", type=int, default=0)
    parser.add_argument("--min-recipient-used-rate", type=float, default=0.0)
    parser.add_argument(
        "--include-disabled",
        action="store_false",
        dest="only_enabled",
        help="Also inspect rows where barrier_enabled is false.",
    )
    parser.set_defaults(only_enabled=True)
    parser.add_argument("--write-json", default="")
    args = parser.parse_args()

    failures, report = validate(args)
    if args.write_json:
        path = Path(args.write_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
