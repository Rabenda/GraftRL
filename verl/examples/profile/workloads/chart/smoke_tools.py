#!/usr/bin/env python3
"""Smoke test: oracle refocus tools on one Refocus Chart parquet row (no GPU)."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

VERL_VISION_ROOT = Path(__file__).resolve().parents[2]
if str(VERL_VISION_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_VISION_ROOT))

import json

from examples.profile.shared.agent.vtool_refocus_tools import RefocusCodeParser, inject_refocus_bbox_context


def _load_image(item: dict) -> Image.Image:
    if item.get("bytes"):
        return Image.open(io.BytesIO(item["bytes"])).convert("RGB")
    return Image.open(item["path"]).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("/data/refocus_chart_multiturn/train.parquet"),
    )
    parser.add_argument("--row", type=int, default=0)
    args = parser.parse_args()

    table = pq.read_table(args.parquet, columns=["images", "extra_info"])
    row = table.slice(args.row, 1).to_pydict()
    images = row["images"][0]
    extra = row["extra_info"][0]
    chart = _load_image(images[0])
    code = extra.get("oracle_refocus_code")
    if not code:
        raise SystemExit("row has no oracle_refocus_code")

    tools_kwargs = extra.get("tools_kwargs") or {}
    raw_meta = tools_kwargs.get("metadata") or {}
    metadata = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta)
    metadata.setdefault("source", extra.get("source_chart") or metadata.get("source"))

    print(f"chart_id={extra.get('chart_id')} source={metadata.get('source')} size={chart.size}")
    print(f"oracle code:\n{code[:200]}...")

    parser_ = RefocusCodeParser()
    tool_output: list[Image.Image] = []

    def _display(img):
        if isinstance(img, Image.Image):
            tool_output.append(img)

    ctx = parser_.get_tool_context(_display)
    inject_refocus_bbox_context(ctx, metadata)
    ctx["image_1"] = chart
    exec_code = parser_.ensure_display_call(code)
    try:
        exec(exec_code, ctx)
    except Exception as exc:
        raise SystemExit(f"refocus exec failed: {exc}") from exc

    edited = tool_output[-1] if tool_output else None
    if edited is None:
        for v in ctx.values():
            if isinstance(v, Image.Image) and v.size[0] > 0:
                edited = v
                break
    if edited is None:
        raise SystemExit("refocus produced no edited image")
    print(f"refocus ok: edited size={edited.size}")
    # Pixel change sanity
    import numpy as np

    a = np.asarray(chart.resize(edited.size)).astype(np.float32)
    b = np.asarray(edited).astype(np.float32)
    diff = np.mean(np.abs(a - b))
    print(f"mean |pixel delta|={diff:.2f} (expect >0 for mask/draw/highlight)")


if __name__ == "__main__":
    main()
