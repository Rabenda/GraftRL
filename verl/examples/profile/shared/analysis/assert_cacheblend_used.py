#!/usr/bin/env python3
"""Fail-fast gate: assert VLM-CacheBlend reuse actually fired in an `on` run.

Reads the per-forward profiling CSV and checks that donor KV was genuinely reused
by recipients. If reuse never triggered (the common failure mode where every
recipient falls back to ``image_token_count_mismatch`` because chunked prefill
fragmented the donor capture), the on/off *timing* comparison is meaningless: the
``on`` path pays donor-capture + per-layer hook overhead for zero benefit. This
script exits non-zero in that case so the workload stops before reporting bogus
"CacheBlend is slower" numbers.

Pass criterion (all required):
  * sum(cacheblend_reused_tokens) > 0
  * at least one row with cacheblend_used == 1

Usage:
  python3 assert_cacheblend_used.py --log-root <DIR> --suffix <SUFFIX>
  python3 assert_cacheblend_used.py --csv <path/to/model_forward_log_*.csv>
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
from typing import Optional


def _resolve_csv(log_root: Optional[str], suffix: Optional[str], csv_path: Optional[str]) -> str:
    if csv_path:
        return csv_path
    if not (log_root and suffix):
        raise SystemExit("error: provide --csv, or both --log-root and --suffix")
    return os.path.join(log_root, f"model_forward_log_{suffix}.csv")


def _to_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root")
    ap.add_argument("--suffix")
    ap.add_argument("--csv")
    args = ap.parse_args()

    path = _resolve_csv(args.log_root, args.suffix, args.csv)
    if not os.path.isfile(path):
        print(f"[assert-cacheblend][FAIL] forward log not found: {path}", file=sys.stderr)
        return 2

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows or "cacheblend_used" not in rows[0]:
        print(
            f"[assert-cacheblend][FAIL] {path} has no cacheblend columns "
            f"(was SGLANG_VLM_CACHEBLEND=1?)",
            file=sys.stderr,
        )
        return 2

    reused_sum = sum(_to_int(r.get("cacheblend_reused_tokens", "0")) for r in rows)
    used_count = sum(1 for r in rows if r.get("cacheblend_used") == "1")
    recompute_sum = sum(_to_int(r.get("cacheblend_recomputed_tokens", "0")) for r in rows)

    donor_tok = sorted({
        _to_int(r["cacheblend_n_image_tokens"])
        for r in rows
        if r.get("cacheblend_role") == "donor"
    })
    recip_tok = sorted({
        _to_int(r["cacheblend_n_image_tokens"])
        for r in rows
        if r.get("cacheblend_role") == "recipient"
    })
    fallbacks = collections.Counter(
        r.get("cacheblend_fallback_reason", "")
        for r in rows
        if r.get("cacheblend_role") in ("donor", "recipient")
    )

    print(f"[assert-cacheblend] csv={path}")
    print(f"[assert-cacheblend] rows={len(rows)} used_rows={used_count} "
          f"reused_tokens_sum={reused_sum} recomputed_tokens_sum={recompute_sum}")
    print(f"[assert-cacheblend] donor n_image_tokens={donor_tok} "
          f"recipient n_image_tokens={recip_tok}")
    for reason, n in fallbacks.most_common():
        print(f"[assert-cacheblend]   fallback {reason or '<none>'}: {n}")

    if reused_sum > 0 and used_count > 0:
        print(f"[assert-cacheblend][PASS] reuse fired: {reused_sum} tokens reused "
              f"across {used_count} forward(s).")
        return 0

    print("[assert-cacheblend][FAIL] CacheBlend reuse never fired "
          "(reused_tokens_sum=0). Timing comparison is meaningless.", file=sys.stderr)
    # Targeted hint for the dominant failure mode.
    mismatch = sum(
        n for reason, n in fallbacks.items() if "image_token_count_mismatch" in reason
    )
    if mismatch:
        donor_vs_recip = (
            "donor captured a FRAGMENT of the span (chunked prefill split it): "
            f"donor={donor_tok} vs recipient={recip_tok}. "
            "Disable chunking: engine_kwargs.sglang.chunked_prefill_size=-1"
        )
        print(f"[assert-cacheblend][HINT] {donor_vs_recip}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
