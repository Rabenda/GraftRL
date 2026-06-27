#!/usr/bin/env python3
"""Diagnose branch/turn image divergence for VTool/Refocus profiling dumps.

This is intentionally lightweight: it only reads the image-dump manifest and PNG
bytes, so it can run without loading a vision model.  If ``--parquet`` is
provided, it also joins each branch back to ``extra_info`` via ``rollout_idx`` and
summarizes the oracle refocus code / bbox metadata behind missing or no-op turn1
outputs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CALL_RE = re.compile(
    r"focus_on_(?P<family>\w+?)_with_(?P<mode>mask|draw|highlight)"
    r"\s*\(\s*image_1\s*,\s*(?P<keys>\[.*?\])\s*,\s*(?P<bbox>\w+)\s*\)",
    re.DOTALL,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(dump_dir: Path) -> list[dict[str, Any]]:
    manifest = dump_dir / "manifest.jsonl"
    if not manifest.is_file():
        raise SystemExit(f"missing manifest: {manifest}")
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def parse_jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def load_extra_info(parquet_path: Path | None) -> list[dict[str, Any]] | None:
    if parquet_path is None:
        return None
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(f"pyarrow is required for --parquet ({exc})")
    table = pq.read_table(parquet_path, columns=["extra_info"])
    return [row["extra_info"] or {} for row in table.to_pylist()]


def parse_refocus_call(code: str | None) -> dict[str, Any]:
    code = code or ""
    match = CALL_RE.search(code)
    if not match:
        return {
            "oracle_code_present": bool(code),
            "focus_call_found": False,
            "focus_function": "",
            "bbox_var": "",
            "requested_keys": [],
        }
    requested_keys: list[str] = []
    try:
        parsed = ast.literal_eval(match.group("keys"))
        if isinstance(parsed, list):
            requested_keys = [str(x) for x in parsed]
    except Exception:
        requested_keys = []
    return {
        "oracle_code_present": bool(code),
        "focus_call_found": True,
        "focus_function": f"focus_on_{match.group('family')}_with_{match.group('mode')}",
        "bbox_var": match.group("bbox"),
        "requested_keys": requested_keys,
    }


def resolve_keys(keys: list[str], bbox: dict[str, Any], fuzzy: bool) -> list[str]:
    resolved = []
    for key in keys:
        if key in bbox:
            resolved.append(key)
            continue
        if fuzzy:
            for candidate in bbox:
                if key == candidate or key in candidate or candidate in key:
                    resolved.append(candidate)
                    break
    return list(dict.fromkeys(resolved))


def parquet_summary(extra_infos: list[dict[str, Any]] | None, rollout_idx: str | None) -> dict[str, Any]:
    if extra_infos is None or rollout_idx is None:
        return {}
    try:
        idx = int(rollout_idx)
    except (TypeError, ValueError):
        return {}
    if idx < 0 or idx >= len(extra_infos):
        return {"dataset_idx": idx, "dataset_idx_found": False}

    extra = extra_infos[idx]
    code = extra.get("oracle_refocus_code") or ""
    call = parse_refocus_call(code)
    metadata = parse_jsonish((extra.get("tools_kwargs") or {}).get("metadata"))
    bbox = metadata.get(call.get("bbox_var")) or {}
    if not isinstance(bbox, dict):
        bbox = {}
    fuzzy = call.get("focus_function") == "focus_on_y_values_with_draw"
    resolved = resolve_keys(call.get("requested_keys", []), bbox, fuzzy=fuzzy)
    return {
        "dataset_idx": idx,
        "dataset_idx_found": True,
        "chart_id": extra.get("chart_id"),
        "source_chart": extra.get("source_chart"),
        "metadata_source": metadata.get("source"),
        "bbox_count": len(bbox),
        "resolved_count": len(resolved),
        "resolved_keys": "|".join(resolved[:8]),
        **{
            k: ("|".join(v) if isinstance(v, list) else v)
            for k, v in call.items()
            if k != "requested_keys"
        },
        "requested_keys": "|".join(call.get("requested_keys", [])[:8]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--parquet", type=Path, default=None, help="Optional train.parquet for oracle/metadata join")
    ap.add_argument("--out-csv", type=Path, default=None, help="Optional per-branch diagnostic CSV")
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()

    rows = load_manifest(args.dump_dir)
    extra_infos = load_extra_info(args.parquet)

    by_uid: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    roles = Counter()
    missing_png = 0
    for row in rows:
        path = args.dump_dir / row["path"]
        if not path.exists():
            missing_png += 1
            continue
        uid = row["uid"]
        branch = row.get("request_id") or str(row.get("rollout_idx", "?"))
        turn = int(row["turn"])
        row = dict(row)
        row["_sha256"] = sha256_file(path)
        by_uid[uid][branch][turn] = row
        roles[row.get("role", "?")] += 1

    branch_rows: list[dict[str, Any]] = []
    missing_t1 = 0
    same_t0_t1 = 0
    changed_t0_t1 = 0
    distinct_t1_dist = Counter()
    t1_complete_dist = Counter()

    for uid, branches in by_uid.items():
        t1_hashes = []
        for branch, turns in branches.items():
            t0 = turns.get(0)
            t1 = turns.get(1)
            status = "missing_t1"
            if t1 is None:
                missing_t1 += 1
            else:
                t1_hashes.append(t1["_sha256"])
                if t0 and t0["_sha256"] == t1["_sha256"]:
                    same_t0_t1 += 1
                    status = "ident_t0_t1"
                else:
                    changed_t0_t1 += 1
                    status = "changed_t0_t1"
            base = {
                "uid": uid,
                "branch": branch,
                "rollout_idx": (t0 or t1 or {}).get("rollout_idx"),
                "status": status,
                "t0_path": t0.get("path") if t0 else "",
                "t1_path": t1.get("path") if t1 else "",
                "t0_sha256": t0.get("_sha256") if t0 else "",
                "t1_sha256": t1.get("_sha256") if t1 else "",
            }
            base.update(parquet_summary(extra_infos, base["rollout_idx"]))
            branch_rows.append(base)
        if t1_hashes:
            distinct_t1_dist[len(set(t1_hashes))] += 1
            t1_complete_dist[len(t1_hashes)] += 1

    print(f"dump_dir: {args.dump_dir}")
    print(f"records={len(rows)} groups={len(by_uid)} branches={sum(len(v) for v in by_uid.values())} missing_png={missing_png}")
    print(f"roles={dict(roles)}")
    print(f"branches: missing_t1={missing_t1} ident_t0_t1={same_t0_t1} changed_t0_t1={changed_t0_t1}")
    print(f"groups by #branches-with-t1: {dict(sorted(t1_complete_dist.items()))}")
    print(f"groups by #distinct-turn1: {dict(sorted(distinct_t1_dist.items()))}")

    problem_rows = [r for r in branch_rows if r["status"] != "changed_t0_t1"]
    if extra_infos is not None and problem_rows:
        by_reason = Counter(
            (
                r["status"],
                bool(r.get("oracle_code_present")),
                r.get("focus_function") or "NO_FOCUS_CALL",
                r.get("bbox_var") or "",
                r.get("bbox_count") or 0,
                r.get("resolved_count") or 0,
            )
            for r in problem_rows
        )
        print("problem reason buckets:")
        for key, count in by_reason.most_common(12):
            print(f"  {count:>5} {key}")

    if args.samples > 0:
        print("sample problem branches:")
        for row in problem_rows[: args.samples]:
            print(
                "  "
                f"{row['status']} uid={row['uid'][:8]} branch={row['branch'][:8]} "
                f"idx={row.get('dataset_idx', row.get('rollout_idx'))} "
                f"chart={row.get('chart_id', '')} func={row.get('focus_function', '')} "
                f"bbox={row.get('bbox_var', '')}:{row.get('bbox_count', '')} "
                f"resolved={row.get('resolved_count', '')}"
            )

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.out_csv, branch_rows)
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
