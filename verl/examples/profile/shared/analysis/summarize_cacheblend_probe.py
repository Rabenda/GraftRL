#!/usr/bin/env python3
"""Summarize VLM CacheBlend probe fields from model_forward_log CSV.

The reader accepts old append-mode logs that contain repeated or changed header
rows, so it can diagnose files produced before the fixed-schema logger.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def _is_header(row: list[str]) -> bool:
    return bool(row) and row[0] == "timestamp" and "mode" in row


def iter_rows(path: Path):
    header: list[str] | None = None
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if _is_header(row):
                header = row
                continue
            if header is None:
                continue
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            yield dict(zip(header, row))


def counter_line(title: str, counter: Counter, limit: int = 20) -> None:
    print(title)
    if not counter:
        print("  (none)")
        return
    for key, count in counter.most_common(limit):
        label = key if key != "" else "(empty)"
        print(f"  {label}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=None, help="model_forward_log CSV")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--suffix", default=None)
    args = parser.parse_args()

    if args.log is None:
        if args.log_dir is None or not args.suffix:
            raise SystemExit("Provide --log or both --log-dir and --suffix")
        args.log = args.log_dir / f"model_forward_log_{args.suffix}.csv"

    rows = list(iter_rows(args.log))
    cacheblend_rows = [r for r in rows if "cacheblend_role" in r]
    extend_rows = [r for r in cacheblend_rows if r.get("mode") == "EXTEND"]

    print(f"log: {args.log}")
    print(f"rows: {len(rows)}")
    print(f"rows_with_cacheblend_schema: {len(cacheblend_rows)}")
    print(f"extend_rows_with_cacheblend_schema: {len(extend_rows)}")

    counter_line("cacheblend_role", Counter(r.get("cacheblend_role", "") for r in cacheblend_rows))
    counter_line(
        "cacheblend_fallback_reason",
        Counter(r.get("cacheblend_fallback_reason", "") for r in cacheblend_rows),
    )
    counter_line("EXTEND fallback_reason", Counter(r.get("cacheblend_fallback_reason", "") for r in extend_rows))
    counter_line("EXTEND batch_size", Counter(r.get("batch_size", "") for r in extend_rows), limit=10)
    counter_line("EXTEND prefill_tokens", Counter(r.get("prefill_tokens", "") for r in extend_rows), limit=20)


if __name__ == "__main__":
    main()
