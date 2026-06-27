#!/usr/bin/env python3
"""
Offline prompt token stats (image pad vs text) for Geo3K vs Refocus_Chart single-turn parquet.
Uses Qwen2.5-VL processor; no GPU rollout required.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
from io import BytesIO

import pyarrow.parquet as pq
from PIL import Image
from transformers import AutoProcessor

from verl.utils.tokenizer import build_multimodal_processor_inputs


def load_rows(path: str, n: int):
    table = pq.read_table(path)
    limit = min(table.num_rows, n)
    for i in range(limit):
        yield {name: table.column(name)[i].as_py() for name in table.column_names}


def build_messages(prompt: list, images: list) -> list:
    image_offset = 0
    out = []
    for message in prompt:
        content = message["content"]
        if not isinstance(content, str):
            out.append(message)
            continue
        content_list = []
        for segment in re.split("(<image>)", content):
            if segment == "":
                continue
            if segment == "<image>":
                im = images[image_offset]
                if isinstance(im, dict) and im.get("bytes"):
                    pil = Image.open(BytesIO(im["bytes"])).convert("RGB")
                    content_list.append({"type": "image", "image": pil})
                image_offset += 1
            else:
                content_list.append({"type": "text", "text": segment})
        out.append({"role": message["role"], "content": content_list})
    return out


def stats_for_dataset(path: str, processor, n: int) -> dict:
    image_token_id = processor.image_token_id
    ratios, image_toks, prefill_toks, img_bytes = [], [], [], []

    for row in load_rows(path, n):
        messages = build_messages(row["prompt"], row.get("images") or [])
        raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        pil_images = []
        for im in row.get("images") or []:
            if isinstance(im, dict) and im.get("bytes"):
                pil_images.append(Image.open(BytesIO(im["bytes"])).convert("RGB"))
                img_bytes.append(len(im["bytes"]))

        inputs = build_multimodal_processor_inputs(
            processor,
            text=[raw_prompt],
            images=pil_images,
        )
        ids = inputs["input_ids"][0].tolist()
        prefill = len(ids)
        image_toks_count = sum(1 for t in ids if t == image_token_id)
        ratios.append(image_toks_count / prefill if prefill else 0.0)
        image_toks.append(image_toks_count)
        prefill_toks.append(prefill)

    return {
        "n": len(ratios),
        "image_prompt_tokens_mean": statistics.mean(image_toks),
        "image_prompt_tokens_median": statistics.median(image_toks),
        "prefill_tokens_mean": statistics.mean(prefill_toks),
        "image_prompt_ratio_mean": statistics.mean(ratios),
        "image_prompt_ratio_median": statistics.median(ratios),
        "raw_image_bytes_kb_mean": statistics.mean(img_bytes) / 1024 if img_bytes else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--geo3k", default="/workspace/repo/verl_vision/data/geo3k/train.parquet")
    parser.add_argument(
        "--chartqa",
        default="/workspace/repo/verl_vision/data/refocus_chart/train.parquet",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", "/data/huggingface_cache")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    for name, path in [("geo3k", args.geo3k), ("refocus_chart", args.chartqa)]:
        if not os.path.exists(path):
            print(f"[skip] {name}: missing {path}")
            continue
        s = stats_for_dataset(path, processor, args.n)
        print(f"\n=== {name} ({path}) n={s['n']} ===")
        for k, v in s.items():
            if k == "n":
                continue
            print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
