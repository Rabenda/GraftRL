#!/usr/bin/env python3
"""Report multimodal prompt token lengths using the same path as RLHFDataset filtering."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from transformers import AutoProcessor

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import build_multimodal_processor_inputs, hf_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument(
        "--caps",
        nargs="+",
        type=int,
        default=[4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576],
    )
    return parser.parse_args()


def make_dataset_config() -> OmegaConf:
    return OmegaConf.create(
        {
            "image_key": "images",
            "video_key": "videos",
            "audio_key": "audios",
            "prompt_key": "prompt",
            "max_prompt_length": 10**9,
            "filter_overlong_prompts": False,
            "truncation": "error",
            "trust_remote_code": False,
            "image_patch_size": 14,
            "apply_chat_template_kwargs": {},
            "mm_processor_kwargs": {},
            "return_raw_chat": True,
            "return_multi_modal_inputs": True,
            "return_full_prompt": False,
            "use_shm": False,
            "custom_cls": {"path": None, "name": None},
            "tool_config_path": None,
            "function_tool_path": None,
            "reward_fn_key": "data_source",
            "dataloader_num_workers": 0,
            "shuffle": False,
        }
    )


def prompt_length(dataset: RLHFDataset, row: dict) -> int:
    processor = dataset.processor
    messages = dataset._build_messages(row, key=dataset.prompt_key)
    apply_kwargs = dict(**dataset.apply_chat_template_kwargs)
    if dataset.tool_schemas is not None:
        apply_kwargs["tools"] = dataset.tool_schemas
    raw_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
    )
    images, videos, audios = dataset._process_multi_modal_info(
        messages, dataset.image_patch_size, dataset.config
    )
    if images is None and videos is None and audios is None:
        return len(
            processor.tokenizer(
                text=raw_prompt,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
        )
    return len(
        build_multimodal_processor_inputs(
            processor,
            text=[raw_prompt],
            images=images,
            videos=videos,
            audio=audios,
            mm_processor_kwargs=dataset.mm_processor_kwargs,
        )["input_ids"][0]
    )


def analyze_split(
    dataset: RLHFDataset, split: str, input_dir: Path, caps: list[int]
) -> tuple[np.ndarray, int]:
    table = pq.read_table(input_dir / f"{split}.parquet")
    rows = table.to_pylist()
    lens: list[int] = []
    errors = 0
    for i, row in enumerate(rows):
        try:
            lens.append(prompt_length(dataset, row))
        except Exception:
            errors += 1
            print(f"[{split}] row {i} length error:", flush=True)
            traceback.print_exc()
    arr = np.array(lens, dtype=np.int64)
    print(f"\n=== {split}: rows={len(rows)} measured={len(arr)} errors={errors} ===")
    if len(arr) == 0:
        return arr, errors
    print(
        f"  min={arr.min()} p50={int(np.percentile(arr, 50))} "
        f"p90={int(np.percentile(arr, 90))} p95={int(np.percentile(arr, 95))} "
        f"max={arr.max()} mean={arr.mean():.0f}"
    )
    for cap in caps:
        kept = int((arr <= cap).sum())
        print(f"    <= {cap:6d}: {kept:4d} / {len(arr)}  ({100 * kept / len(arr):.0f}%)")
    return arr, errors


def main() -> None:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
    tokenizer = hf_tokenizer(args.model, trust_remote_code=False)
    config = make_dataset_config()

    all_lens: list[np.ndarray] = []
    for split in args.splits:
        input_file = args.input_dir / f"{split}.parquet"
        if not input_file.exists():
            raise FileNotFoundError(input_file)
        dataset = RLHFDataset(
            data_files=str(input_file), tokenizer=tokenizer, processor=processor, config=config
        )
        arr, _ = analyze_split(dataset, split, args.input_dir, args.caps)
        if len(arr):
            all_lens.append(arr)

    if len(all_lens) > 1:
        merged = np.concatenate(all_lens)
        print(f"\n=== combined: n={len(merged)} ===")
        print(
            f"  min={merged.min()} p50={int(np.percentile(merged, 50))} "
            f"p90={int(np.percentile(merged, 90))} p95={int(np.percentile(merged, 95))} "
            f"max={merged.max()} mean={merged.mean():.0f}"
        )
        for cap in args.caps:
            kept = int((merged <= cap).sum())
            print(f"    <= {cap:6d}: {kept:4d} / {len(merged)}  ({100 * kept / len(merged):.0f}%)")


if __name__ == "__main__":
    main()
