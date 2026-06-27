#!/usr/bin/env python3
"""Build a single stress dataset for VLM-CacheBlend turn1 image-KV reuse.

This keeps the Refocus Chart task semantics but enlarges the chart image and all
refocus bbox metadata by the same scale factor. Oracle refocus then produces a large
turn1 refocus image, so the last image slot has enough LLM image tokens for
CacheBlend savings to be measurable.

Default scale=2.0 is chosen to stay runnable with Qwen2.5-VL 7B on 4xH100 while
raising each chart image from roughly 580 to roughly 2.3k LLM image tokens.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from io import BytesIO
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def image_item_to_pil(image_item: Any) -> Image.Image:
    if isinstance(image_item, dict):
        image_bytes = image_item.get("bytes")
        image_path = image_item.get("path")
        if image_bytes is not None:
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        if image_path:
            return Image.open(image_path).convert("RGB")
    if isinstance(image_item, (bytes, bytearray)):
        return Image.open(BytesIO(image_item)).convert("RGB")
    if isinstance(image_item, str):
        return Image.open(image_item).convert("RGB")
    raise ValueError(f"Unsupported image item type: {type(image_item)!r}")


def pil_to_image_item(image: Image.Image) -> dict:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return {"bytes": buf.getvalue(), "path": None}


def resize_image(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resampling = getattr(Image, "Resampling", Image)
    return image.resize(new_size, resampling.BICUBIC)


def _scale_bbox_coord(value: Any, scale: float) -> Any:
    if isinstance(value, list):
        return [_scale_bbox_coord(v, scale) for v in value]
    return float(value) * scale


def _scale_bbox_dict(value: dict, scale: float) -> dict:
    out = dict(value)
    for key in ("x1", "y1", "x2", "y2"):
        if key in out:
            out[key] = _scale_bbox_coord(out[key], scale)
    return out


def scale_bbox_metadata(value: Any, scale: float) -> Any:
    """Scale nested metadata while preserving unrelated fields.

    Refocus Chart metadata stores bboxes in nested dicts such as
    x_values_bbox/key/{x1,y1,x2,y2}. We recursively scale any dict that looks like a
    bbox leaf.
    """
    if isinstance(value, dict):
        if all(k in value for k in ("x1", "y1", "x2", "y2")):
            return _scale_bbox_dict(value, scale)
        return {k: scale_bbox_metadata(v, scale) for k, v in value.items()}
    if isinstance(value, list):
        return [scale_bbox_metadata(v, scale) for v in value]
    return value


def _load_metadata(raw: Any) -> tuple[Any, bool]:
    if isinstance(raw, str):
        try:
            return json.loads(raw), True
        except json.JSONDecodeError:
            return raw, True
    return raw, False


def scale_tools_metadata(extra_info: dict, scale: float) -> dict:
    extra = copy.deepcopy(extra_info or {})
    tools_kwargs = copy.deepcopy(extra.get("tools_kwargs") or {})
    if not tools_kwargs:
        return extra
    metadata, was_json = _load_metadata(tools_kwargs.get("metadata"))
    if isinstance(metadata, dict):
        metadata = scale_bbox_metadata(metadata, scale)
        tools_kwargs["metadata"] = json.dumps(metadata) if was_json else metadata
    extra["tools_kwargs"] = tools_kwargs
    return extra


def add_stress_note(row: dict, *, scale: float, split: str, idx: int) -> dict:
    extra = dict(row.get("extra_info") or {})
    extra["cacheblend_stress"] = True
    extra["cacheblend_stress_scale"] = scale
    extra["cacheblend_stress_split"] = split
    extra["cacheblend_stress_index"] = idx
    extra["cacheblend_stress_note"] = (
        "Chart image and bbox metadata are scaled to enlarge the turn1 refocus "
        "image span for VLM-CacheBlend profiling."
    )
    row["extra_info"] = extra
    return row


def convert_row(row: dict, *, scale: float, split: str, idx: int) -> dict:
    row = copy.deepcopy(row)
    images = list(row.get("images") or [])
    if not images:
        raise ValueError("Expected at least one image in Refocus Chart row")

    # The agent uses images[0] as image_1 and appends the tool output as the turn1
    # refocus image. Scaling image_1 therefore scales both original and refocus slots.
    images[0] = pil_to_image_item(resize_image(image_item_to_pil(images[0]), scale))
    row["images"] = images
    row["extra_info"] = scale_tools_metadata(row.get("extra_info") or {}, scale)
    return add_stress_note(row, scale=scale, split=split, idx=idx)


def iter_rows(src: str):
    parquet_file = pq.ParquetFile(src)
    for batch in parquet_file.iter_batches(batch_size=32):
        table = pa.Table.from_batches([batch])
        for idx in range(table.num_rows):
            yield {name: table.column(name)[idx].as_py() for name in table.column_names}


def convert_parquet(
    src: str,
    dst: str,
    *,
    scale: float,
    split: str,
    max_rows: int | None,
    require_oracle: bool,
) -> int:
    rows_out = []
    for src_idx, row in enumerate(iter_rows(src)):
        if max_rows is not None and len(rows_out) >= max_rows:
            break
        extra_info = row.get("extra_info") or {}
        if require_oracle and not extra_info.get("oracle_refocus_code"):
            continue
        rows_out.append(convert_row(row, scale=scale, split=split, idx=src_idx))

    if not rows_out:
        raise RuntimeError(f"No rows converted from {src}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows_out), dst)
    return len(rows_out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        default="/data/refocus_chart_multiturn_oracle_changed",
        help="Directory with Refocus Chart train.parquet / test.parquet.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="/data/refocus_chart_cacheblend_stress_s2",
        help="Output directory for the CacheBlend stress parquet.",
    )
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--max_train_rows", type=int, default=32)
    parser.add_argument("--max_test_rows", type=int, default=8)
    args = parser.parse_args()

    if args.scale <= 1.0:
        raise ValueError("--scale should be > 1.0 for a stress dataset")

    for split, cap in (("train", args.max_train_rows), ("test", args.max_test_rows)):
        src = os.path.join(args.src_dir, f"{split}.parquet")
        dst = os.path.join(args.local_save_dir, f"{split}.parquet")
        if not os.path.isfile(src):
            raise FileNotFoundError(src)
        if split == "test" and pq.ParquetFile(src).metadata.num_rows == 0:
            src = os.path.join(args.src_dir, "train.parquet")
        count = convert_parquet(
            src,
            dst,
            scale=args.scale,
            split=split,
            max_rows=cap,
            require_oracle=(split == "train"),
        )
        print(f"Wrote {count} rows -> {dst}")


if __name__ == "__main__":
    main()
