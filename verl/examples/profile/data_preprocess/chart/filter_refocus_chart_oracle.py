#!/usr/bin/env python3
"""Filter Refocus_Chart parquet for valid turn0 -> turn1 visual transitions.

Keeps only rows where oracle refocus code exists, parses to a real focus_on_* call,
bbox keys resolve against metadata, the tool executes without error, and the edited
image differs from the input (changed_pixel_frac > threshold).

Drop reasons (offline gate, before rollout):
  missing_oracle_code  - no oracle_refocus_code (incl. ACTION 0 / No action needed)
  no_tool_call         - code present but no focus_on_* call matched
  missing_image        - row has no input image
  bad_image            - cannot decode input image
  unresolved_bbox      - requested_keys do not resolve to any bbox entry
  exec_failed          - exec() raised
  no_output_image      - exec succeeded but no edited image produced
  unchanged            - tool output is byte-identical or zero pixel change

Rollout-time checks (unchanged_after_dump, manifest paths) belong in
``diagnose_refocus_divergence.py``, not here.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageChops

CALL_RE = re.compile(
    r"focus_on_(?P<family>\w+?)_with_(?P<mode>mask|draw|highlight)"
    r"\s*\(\s*image_1\s*,\s*(?P<keys>\[.*?\])\s*,\s*(?P<bbox>\w+)\s*\)",
    re.DOTALL,
)


def _ensure_repo_imports() -> None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "verl").is_dir() and (parent / "examples" / "profile").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return
    raise RuntimeError("cannot find verl_vision repo root from filter_refocus_chart_oracle.py")


_ensure_repo_imports()

from examples.profile.shared.agent.vtool_refocus_tools import RefocusCodeParser, inject_refocus_bbox_context  # noqa: E402


def image_from_item(item: Any) -> Image.Image:
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, dict):
        if item.get("bytes"):
            return Image.open(BytesIO(item["bytes"])).convert("RGB")
        if item.get("path"):
            return Image.open(item["path"]).convert("RGB")
    if isinstance(item, str):
        return Image.open(item).convert("RGB")
    raise TypeError(f"unsupported image item: {type(item)!r}")


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


def parse_refocus_call(code: str) -> dict[str, Any]:
    match = CALL_RE.search(code)
    if not match:
        return {
            "focus_call_found": False,
            "tool_name": "",
            "bbox_name": "",
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
        "focus_call_found": True,
        "tool_name": f"focus_on_{match.group('family')}_with_{match.group('mode')}",
        "bbox_name": match.group("bbox"),
        "requested_keys": requested_keys,
    }


def resolve_keys(keys: list[str], bbox: dict[str, Any], *, fuzzy: bool) -> list[str]:
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


def bbox_dict_for_call(metadata: dict[str, Any], call: dict[str, Any]) -> dict[str, Any]:
    bbox_name = call.get("bbox_name") or ""
    raw = metadata.get(bbox_name) or {}
    if isinstance(raw, dict):
        return raw
    return {}


def pixel_sha256(image: Image.Image) -> str:
    image = image.convert("RGB")
    h = hashlib.sha256()
    h.update(str(image.size).encode())
    h.update(image.tobytes())
    return h.hexdigest()


def changed_pixel_frac(before: Image.Image, after: Image.Image) -> float:
    lhs = before.convert("RGB")
    rhs = after.convert("RGB")
    if lhs.size != rhs.size:
        rhs = rhs.resize(lhs.size)
    diff = ImageChops.difference(lhs, rhs)
    total = max(lhs.size[0] * lhs.size[1], 1)
    if diff.getbbox() is None:
        return 0.0
    changed = sum(1 for pixel in diff.getdata() if pixel != (0, 0, 0))
    return changed / total


def question_snippet(row: dict[str, Any], extra: dict[str, Any], max_len: int = 120) -> str:
    for src in (extra.get("question"), row.get("prompt")):
        if not src:
            continue
        text = str(src).replace("\n", " ").strip()
        if text:
            return text[:max_len]
    return ""


def evaluate_row(row: dict[str, Any], min_changed_frac: float) -> tuple[bool, dict[str, Any]]:
    extra = row.get("extra_info") or {}
    code = (extra.get("oracle_refocus_code") or "").strip()
    metadata = parse_jsonish((extra.get("tools_kwargs") or {}).get("metadata"))

    report: dict[str, Any] = {
        "oracle_code_present": int(bool(code)),
        "tool_name": "",
        "requested_keys": "",
        "bbox_name": "",
        "bbox_count": 0,
        "resolved_count": 0,
        "resolved_keys": "",
        "tool_success": 0,
        "changed_pixel_frac": "0.00000000",
        "drop_reason": "missing_oracle_code",
    }

    if not code:
        return False, report

    call = parse_refocus_call(code)
    report["tool_name"] = call.get("tool_name") or ""
    report["bbox_name"] = call.get("bbox_name") or ""
    report["requested_keys"] = "|".join(call.get("requested_keys") or [])

    if not call.get("focus_call_found"):
        report["drop_reason"] = "no_tool_call"
        return False, report

    bbox = bbox_dict_for_call(metadata, call)
    report["bbox_count"] = len(bbox)
    fuzzy = report["tool_name"] == "focus_on_y_values_with_draw"
    resolved = resolve_keys(call.get("requested_keys") or [], bbox, fuzzy=fuzzy)
    report["resolved_count"] = len(resolved)
    report["resolved_keys"] = "|".join(resolved[:8])

    if not resolved:
        report["drop_reason"] = "unresolved_bbox"
        return False, report

    images = row.get("images") or []
    if not images:
        report["drop_reason"] = "missing_image"
        return False, report

    try:
        before = image_from_item(images[0])
    except Exception as exc:
        report["drop_reason"] = f"bad_image:{type(exc).__name__}"
        return False, report

    parser = RefocusCodeParser()
    output: dict[str, Image.Image | None] = {"image": None}

    def display(image: Image.Image) -> None:
        output["image"] = image

    context = parser.get_tool_context(display)
    inject_refocus_bbox_context(context, metadata)
    context["display"] = display
    context["image_1"] = before.copy()
    initial_keys = set(context)
    executable = parser.ensure_display_call(code)

    try:
        exec(executable, context)
    except Exception as exc:
        report["drop_reason"] = f"exec_failed:{type(exc).__name__}"
        return False, report

    edited = output["image"]
    if not isinstance(edited, Image.Image):
        for key, value in reversed(list(context.items())):
            if key not in initial_keys and isinstance(value, Image.Image):
                edited = value
                break
    if not isinstance(edited, Image.Image):
        report["drop_reason"] = "no_output_image"
        return False, report

    report["tool_success"] = 1
    frac = changed_pixel_frac(before, edited)
    report["changed_pixel_frac"] = f"{frac:.8f}"

    if pixel_sha256(before) == pixel_sha256(edited) or frac <= min_changed_frac:
        report["drop_reason"] = "unchanged"
        return False, report

    report["drop_reason"] = "ok"
    return True, report


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def write_summary(path: Path, *, split: str, total: int, kept: int, reason_counts: dict[str, int]) -> None:
    payload = {
        "split": split,
        "total": total,
        "kept": kept,
        "dropped": total - kept,
        "keep_rate": round(kept / total, 6) if total else 0.0,
        "drop_reasons": dict(sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def filter_file(
    src: Path,
    dst: Path,
    report: Path,
    summary: Path,
    min_changed_frac: float,
    max_rows: int | None,
    progress_every: int,
) -> None:
    table = pq.read_table(src)
    rows = table.to_pylist()
    if max_rows is not None:
        rows = rows[:max_rows]

    kept: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}

    for idx, row in enumerate(rows):
        if progress_every > 0 and idx > 0 and idx % progress_every == 0:
            print(f"  {src.name}: processed {idx}/{len(rows)} kept={len(kept)}", flush=True)

        ok, meta = evaluate_row(row, min_changed_frac)
        extra = row.get("extra_info") or {}
        drop_reason = meta["drop_reason"]
        reason_counts[drop_reason] = reason_counts.get(drop_reason, 0) + 1

        report_rows.append(
            {
                "row_idx": idx,
                "keep": int(ok),
                "drop_reason": drop_reason,
                "chart_id": extra.get("chart_id", ""),
                "source_chart": extra.get("source_chart", ""),
                "question": question_snippet(row, extra),
                **meta,
            }
        )
        if ok:
            kept.append(row)

    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(kept, schema=table.schema), dst)
    write_report(report, report_rows)
    write_summary(summary, split=src.stem, total=len(rows), kept=len(kept), reason_counts=reason_counts)

    print(
        f"{src} -> {dst}: kept {len(kept)}/{len(rows)} "
        f"({len(rows) - len(kept)} dropped), reasons={reason_counts}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=Path("/data/refocus_chart_multiturn"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--splits",
        default="train",
        help="Comma-separated splits (default: train only; test has no teacher thoughts/oracle)",
    )
    ap.add_argument(
        "--min-changed-frac",
        type=float,
        default=0.0,
        help="Minimum changed pixel fraction after tool exec (default: any change)",
    )
    ap.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test cap per split")
    ap.add_argument("--progress-every", type=int, default=500)
    args = ap.parse_args()

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        src = args.input_dir / f"{split}.parquet"
        dst = args.output_dir / f"{split}.parquet"
        report = args.output_dir / f"{split}_filter_report.csv"
        summary = args.output_dir / f"{split}_filter_summary.json"
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        filter_file(src, dst, report, summary, args.min_changed_frac, args.max_rows, args.progress_every)


if __name__ == "__main__":
    main()
