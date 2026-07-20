#!/usr/bin/env python3
"""Offline length filtering for MMDU parquet files.

This uses the same RLHFDataset multimodal length path as training, then writes
the filtered HuggingFace Dataset back to parquet so GPU runs do not spend time
filtering after allocation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf
from transformers import AutoProcessor

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import hf_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-prompt-length", type=int, default=24576)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    return parser.parse_args()


def filter_split(args: argparse.Namespace, split: str, tokenizer, processor) -> None:
    input_file = args.input_dir / f"{split}.parquet"
    output_file = args.output_dir / f"{split}.parquet"
    if not input_file.exists():
        raise FileNotFoundError(input_file)

    config = OmegaConf.create(
        {
            "image_key": "images",
            "video_key": "videos",
            "audio_key": "audios",
            "prompt_key": "prompt",
            "max_prompt_length": args.max_prompt_length,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": args.num_workers,
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

    dataset = RLHFDataset(data_files=str(input_file), tokenizer=tokenizer, processor=processor, config=config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.dataframe.to_parquet(str(output_file))
    print(f"{split}: wrote {len(dataset)} rows to {output_file}")


def main() -> None:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
    tokenizer = hf_tokenizer(args.model, trust_remote_code=False)
    for split in args.splits:
        filter_split(args, split, tokenizer, processor)


if __name__ == "__main__":
    main()
