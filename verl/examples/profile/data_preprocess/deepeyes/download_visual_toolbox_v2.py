#!/usr/bin/env python3
"""Download ChenShawn/DeepEyes-Datasets-47k (visual_toolbox_v2 rows) to verl parquet.

Filters rows where env_name == visual_toolbox_v2 (DeepEyes zoom/crop agent data).
Images are stored as PNG bytes for verl RLHFDataset compatibility.

Usage:
  python examples/profile/data_preprocess/deepeyes/download_visual_toolbox_v2.py \\
    --local_save_dir /data/deepeyes_visual_toolbox_v2 \\
    --max_rows 2000
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from PIL import Image


def _normalize_image_item(item) -> dict:
    if isinstance(item, dict):
        if item.get("bytes"):
            return {"bytes": item["bytes"], "path": item.get("path")}
        if item.get("path"):
            with open(item["path"], "rb") as f:
                return {"bytes": f.read(), "path": None}
    if isinstance(item, Image.Image):
        buf = BytesIO()
        item.convert("RGB").save(buf, format="PNG")
        return {"bytes": buf.getvalue(), "path": None}
    if isinstance(item, str):
        with open(item, "rb") as f:
            return {"bytes": f.read(), "path": None}
    raise TypeError(f"Unsupported image item: {type(item)!r}")


def convert_row(example: dict, idx: int) -> dict:
    extra = dict(example.get("extra_info") or {})
    extra["index"] = idx
    extra.setdefault("split", "train")

    images_raw = example.get("images") or []
    if not isinstance(images_raw, list):
        images_raw = [images_raw]
    images = [_normalize_image_item(img) for img in images_raw]

    reward_model = example.get("reward_model") or {}
    if not reward_model.get("ground_truth") and extra.get("answer"):
        reward_model = {"style": "rule", "ground_truth": str(extra["answer"])}

    return {
        "images": images,
        "data_source": example.get("data_source") or "vstar",
        "prompt": example.get("prompt"),
        "ability": example.get("ability") or "vl",
        "reward_model": reward_model,
        # HF rows use deepeyes_visual_toolbox_v2; profile agent loop registers both names.
        "agent_name": "deepeyes_agent",
        "env_name": example.get("env_name") or "visual_toolbox_v2",
        "extra_info": extra,
    }


def export_split(
    split_name: str,
    out_path: str,
    max_rows: int | None,
    env_name: str,
    cache_dir: str | None,
    skip_rows: int = 0,
) -> dict[str, int]:
    ds = load_dataset(
        "ChenShawn/DeepEyes-Datasets-47k",
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    )
    rows = []
    stats = {"seen": 0, "kept": 0, "skipped_env": 0, "skipped_offset": 0}

    print(
        f"Streaming ChenShawn/DeepEyes-Datasets-47k -> {split_name} "
        f"(env_name={env_name!r}, max_rows={max_rows}, skip_rows={skip_rows})...",
        flush=True,
    )
    env_kept = 0
    for example in ds:
        stats["seen"] += 1
        if example.get("env_name") != env_name:
            stats["skipped_env"] += 1
            continue
        if env_kept < skip_rows:
            env_kept += 1
            stats["skipped_offset"] += 1
            continue
        if max_rows is not None and stats["kept"] >= max_rows:
            break
        rows.append(convert_row(example, stats["kept"]))
        stats["kept"] += 1
        env_kept += 1
        if stats["kept"] % 200 == 0:
            print(f"  {split_name}: kept {stats['kept']} rows...", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_save_dir",
        default="/data/deepeyes_visual_toolbox_v2",
    )
    parser.add_argument("--max_rows", type=int, default=2000)
    parser.add_argument("--max_test_rows", type=int, default=200)
    parser.add_argument(
        "--env_name",
        default="visual_toolbox_v2",
        help="Keep only rows with this env_name",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="HF datasets cache (default: HF_DATASETS_CACHE or /data/huggingface_cache/datasets)",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or os.environ.get("HF_DATASETS_CACHE")
    os.makedirs(args.local_save_dir, exist_ok=True)

    train_path = os.path.join(args.local_save_dir, "train.parquet")
    train_stats = export_split(
        "train", train_path, args.max_rows, args.env_name, cache_dir, skip_rows=0
    )
    print(f"Wrote {train_stats['kept']} rows to {train_path}")
    print(f"Train stats: {train_stats}")

    test_path = os.path.join(args.local_save_dir, "test.parquet")
    test_stats = export_split(
        "test",
        test_path,
        args.max_test_rows,
        args.env_name,
        cache_dir,
        skip_rows=args.max_rows,
    )
    print(f"Wrote {test_stats['kept']} rows to {test_path}")
    print(f"Test stats: {test_stats}")


if __name__ == "__main__":
    main()
