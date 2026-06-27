# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
Build a dummy Refocus_Chart crop dataset for smoke-testing the vision path.

This intentionally does not depend on model tool calls.  It takes the
single-turn Refocus_Chart parquet, appends a center crop as Image_1, and
adds a second <image> token to the prompt so rollout/train exercise a
"newly generated image" input.
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def image_item_to_pil(image_item) -> Image.Image:
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


def center_crop(image: Image.Image, crop_ratio: float) -> Image.Image:
    width, height = image.size
    crop_w = max(1, int(width * crop_ratio))
    crop_h = max(1, int(height * crop_ratio))
    left = max(0, (width - crop_w) // 2)
    top = max(0, (height - crop_h) // 2)
    return image.crop((left, top, left + crop_w, top + crop_h))


def add_second_image_token(content: str) -> str:
    prefix = "Original image:\n<image>\nDummy refocused crop:\n<image>\n"
    if "<image>" in content:
        return content.replace("<image>", prefix, 1)
    return f"{prefix}{content}"


def convert_row(row: dict, crop_ratio: float) -> dict:
    images = list(row.get("images") or [])
    if not images:
        raise ValueError("Expected at least one image in Refocus_Chart row")

    crop = center_crop(image_item_to_pil(images[0]), crop_ratio)
    row["images"] = images + [pil_to_image_item(crop)]
    row["prompt"] = [
        {
            **row["prompt"][0],
            "content": add_second_image_token(row["prompt"][0]["content"]),
        }
    ]

    extra_info = dict(row.get("extra_info") or {})
    extra_info.update(
        {
            "dummy_crop": True,
            "dummy_crop_ratio": crop_ratio,
            "dummy_crop_note": "Image_1 is a generated center crop of Image_0 for smoke profiling.",
        }
    )
    row["extra_info"] = extra_info
    return row


def iter_rows(src: str):
    parquet_file = pq.ParquetFile(src)
    for batch in parquet_file.iter_batches(batch_size=64):
        table = pa.Table.from_batches([batch])
        for idx in range(table.num_rows):
            yield {name: table.column(name)[idx].as_py() for name in table.column_names}


def convert_parquet(src: str, dst: str, crop_ratio: float, max_rows: int | None = None) -> int:
    rows_out = []

    for idx, row in enumerate(iter_rows(src)):
        if max_rows is not None and idx >= max_rows:
            break
        rows_out.append(convert_row(row, crop_ratio))

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows_out), dst)
    return len(rows_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        default="/workspace/repo/verl_vision/data/refocus_chart",
        help="Directory with single-turn Refocus_Chart train.parquet / test.parquet.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="/workspace/repo/verl_vision/data/refocus_chart_dummy_crop",
        help="Output directory for dummy crop parquet.",
    )
    parser.add_argument("--crop_ratio", type=float, default=0.5)
    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--max_test_rows", type=int, default=None)
    args = parser.parse_args()

    if not 0 < args.crop_ratio <= 1:
        raise ValueError("--crop_ratio must be in (0, 1]")

    for split, cap in [("train", args.max_train_rows), ("test", args.max_test_rows)]:
        src = os.path.join(args.src_dir, f"{split}.parquet")
        dst = os.path.join(args.local_save_dir, f"{split}.parquet")
        count = convert_parquet(src, dst, args.crop_ratio, max_rows=cap)
        print(f"Wrote {count} rows -> {dst}")
