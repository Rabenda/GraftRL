#!/usr/bin/env python3
"""Download VTOOL/Refocus_Chart (streaming) to verl_vision parquet for vtool_agent profiling.

Preserves multiturn prompt + tools_kwargs.metadata (bbox for refocus Python API).
Stores oracle refocus code from dataset thoughts for VTOOL_ORACLE_REFOCUS=1 profiling.

Usage:
  python examples/profile/data_preprocess/chart/download_refocus_chart.py \\
    --local_save_dir data/refocus_chart_multiturn \\
    --max_train_rows 512 --max_test_rows 128
"""

from __future__ import annotations

import argparse
import os
import re
from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from PIL import Image

_ACTION_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_oracle_refocus_code(thoughts: list | None) -> str | None:
    if not thoughts:
        return None
    for text in thoughts:
        if not isinstance(text, str):
            continue
        match = _ACTION_RE.search(text)
        if match and "focus_on_" in match.group(1):
            return match.group(1).strip()
    return None


def _to_pil(image_item) -> Image.Image:
    if isinstance(image_item, Image.Image):
        return image_item.convert("RGB")
    if isinstance(image_item, dict) and image_item.get("bytes"):
        return Image.open(BytesIO(image_item["bytes"])).convert("RGB")
    if isinstance(image_item, str):
        return Image.open(image_item).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image_item)!r}")


def pil_to_parquet_item(image: Image.Image) -> dict:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return {"bytes": buf.getvalue(), "path": None}


def convert_row(example: dict, idx: int, split: str) -> dict:
    extra = dict(example.get("extra_info") or {})
    thoughts = list(example.get("thoughts") or [])
    extra["thoughts"] = thoughts
    extra["oracle_refocus_code"] = extract_oracle_refocus_code(thoughts)
    extra["split"] = split
    extra["index"] = idx
    extra["source_chart"] = example.get("source")
    extra["chart_id"] = example.get("id")
    extra.setdefault("need_tools_kwargs", True)

    edited = example.get("edited_image")
    if edited is not None:
        try:
            extra["gold_edited_image"] = pil_to_parquet_item(_to_pil(edited))
        except Exception:
            pass

    images_raw = example.get("images")
    if images_raw is None:
        images_list = []
    elif isinstance(images_raw, list):
        images_list = images_raw
    else:
        images_list = [images_raw]

    images = [pil_to_parquet_item(_to_pil(img)) for img in images_list]

    reward_model = example.get("reward_model") or {}
    if not reward_model.get("ground_truth"):
        reward_model = {
            "style": "rule",
            "ground_truth": str(extra.get("answer", "")),
        }

    return {
        "images": images,
        "data_source": example.get("data_source") or "VTOOL/Refocus_Chart",
        "prompt": example.get("prompt"),
        "ability": example.get("ability", "math"),
        "reward_model": reward_model,
        "agent_name": example.get("agent_name") or "vtool_agent",
        "extra_info": extra,
    }


def export_split(
    split: str,
    out_path: str,
    max_rows: int | None,
    cache_dir: str | None,
) -> int:
    ds = load_dataset(
        "VTOOL/Refocus_Chart",
        split=split,
        streaming=True,
        cache_dir=cache_dir,
    )
    rows = []
    print(f"Streaming {split} from VTOOL/Refocus_Chart (max_rows={max_rows})...", flush=True)
    for idx, example in enumerate(ds):
        if max_rows is not None and idx >= max_rows:
            break
        rows.append(convert_row(example, idx, split))
        if (idx + 1) % 100 == 0:
            print(f"  {split}: {idx + 1} rows...", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out_path)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_save_dir",
        default="/workspace/repo/verl_vision/data/refocus_chart_multiturn",
    )
    parser.add_argument("--max_train_rows", type=int, default=512)
    parser.add_argument("--max_test_rows", type=int, default=128)
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="HF datasets cache (default: HF_DATASETS_CACHE or /data/huggingface_cache/datasets)",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or os.environ.get("HF_DATASETS_CACHE")
    os.makedirs(args.local_save_dir, exist_ok=True)

    for split, cap in [("train", args.max_train_rows), ("test", args.max_test_rows)]:
        dst = os.path.join(args.local_save_dir, f"{split}.parquet")
        print(f"Export {split} -> {dst} (max_rows={cap})", flush=True)
        n = export_split(split, dst, cap, cache_dir)
        print(f"Wrote {n} rows to {dst}", flush=True)


if __name__ == "__main__":
    main()
