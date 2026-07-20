#!/usr/bin/env python3
"""Download and convert MMDU benchmark or MMDU-45k subsets to verl parquet.

Sources on HuggingFace (laolao77/MMDU):
  - benchmark: 110 dialogues, benchmark.json + mmdu_pics.zip (max ~550 rows @ turns=5)
  - 45k: ~45k dialogues, mmdu-45k.json + mmdu-45k_pics.zip

Each dialogue can yield multiple rollout rows via --turns_per_dialogue (last N assistant turns).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


DATA_SOURCE = "laolao77/MMDU"
IMAGE_TOKEN_RE = re.compile(r"<ImageHere>|<image>")

DATASET_FILES = {
    "benchmark": {
        "json": "benchmark.json",
        "zip": "mmdu_pics.zip",
        "extra": ("README.md",),
    },
    "45k": {
        "json": "mmdu-45k.json",
        "zip": "mmdu-45k_pics.zip",
        "extra": (),
    },
}


def download_raw(raw_dir: Path, dataset: str, force: bool) -> tuple[Path, Path]:
    if dataset not in DATASET_FILES:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {sorted(DATASET_FILES)}")
    spec = DATASET_FILES[dataset]
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename in (*spec["extra"], spec["json"], spec["zip"]):
        dst = raw_dir / filename
        if dst.exists() and not force:
            continue
        print(f"Downloading {filename} ...", flush=True)
        hf_hub_download(
            repo_id=DATA_SOURCE,
            repo_type="dataset",
            filename=filename,
            local_dir=str(raw_dir),
            force_download=force,
        )
    json_path = raw_dir / spec["json"]
    zip_path = raw_dir / spec["zip"]
    if not json_path.exists() or not zip_path.exists():
        raise FileNotFoundError(f"Missing raw files under {raw_dir}: {json_path}, {zip_path}")
    return json_path, zip_path


def normalize_image_path(path: str) -> str:
    return path.lstrip("/")


def read_image_item(zip_file: ZipFile, image_path: str) -> dict:
    zip_path = normalize_image_path(image_path)
    with zip_file.open(zip_path) as fp:
        return {"bytes": fp.read(), "path": None}


def normalize_text(text: str) -> str:
    return IMAGE_TOKEN_RE.sub("<image>", text)


def convert_message(message: dict) -> dict:
    role = message.get("from")
    if role == "human":
        role = "user"
    if role == "gpt":
        role = "assistant"
    if role not in {"user", "assistant", "system"}:
        raise ValueError(f"Unsupported MMDU role: {role!r}")
    return {"role": role, "content": normalize_text(str(message.get("value", "")))}


def assistant_indices(conversations: list[dict]) -> list[int]:
    indices = []
    for idx, message in enumerate(conversations):
        if message.get("from") in {"assistant", "gpt"}:
            indices.append(idx)
    return indices


def build_rows_for_dialogue(
    example: dict,
    zip_file: ZipFile,
    *,
    split: str,
    turns_per_dialogue: int,
    dataset_name: str,
) -> list[dict]:
    conversations = list(example.get("conversations") or [])
    images = list(example.get("image") or [])
    assistant_idxs = assistant_indices(conversations)
    if not assistant_idxs:
        return []
    selected = assistant_idxs[-turns_per_dialogue:]
    rows = []
    for turn_rank, answer_idx in enumerate(selected):
        if answer_idx <= 0 or conversations[answer_idx - 1].get("from") not in {"user", "human"}:
            continue
        prompt_messages = [convert_message(msg) for msg in conversations[:answer_idx]]
        answer = str(conversations[answer_idx].get("value", ""))
        placeholder_count = sum(str(msg["content"]).count("<image>") for msg in prompt_messages)
        if placeholder_count <= 0:
            continue
        if placeholder_count > len(images):
            continue
        image_paths = images[:placeholder_count]
        try:
            parquet_images = [read_image_item(zip_file, path) for path in image_paths]
        except KeyError:
            continue
        sample_id = f"{example.get('id')}:turn{answer_idx}"
        mmdu_id = example.get("id", -1)
        rows.append(
            {
                "images": parquet_images,
                "data_source": DATA_SOURCE,
                "prompt": prompt_messages,
                "ability": "multimodal_dialogue",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": -1,
                    "sample_id": sample_id,
                    "mmdu_id": mmdu_id,
                    "mmdu_set": str(example.get("set", "")),
                    "mmdu_dataset": dataset_name,
                    "target_answer_turn": int(answer_idx),
                    "target_user_turn": int(answer_idx - 1),
                    "turn_rank_from_end": int(len(selected) - turn_rank),
                    "num_messages_in_prompt": int(len(prompt_messages)),
                    "num_total_messages": int(len(conversations)),
                    "num_images": int(placeholder_count),
                    "num_images_total": int(len(images)),
                    "source_image_paths": [normalize_image_path(path) for path in image_paths],
                    "question": str(conversations[answer_idx - 1].get("value", "")),
                    "answer": answer,
                    "need_tools_kwargs": False,
                },
            }
        )
    return rows


def load_converted_rows(
    json_path: Path,
    zip_path: Path,
    *,
    turns_per_dialogue: int,
    dataset_name: str,
    max_dialogues: int | None = None,
) -> list[dict]:
    with open(json_path, encoding="utf-8") as fp:
        data = json.load(fp)
    if max_dialogues is not None and max_dialogues > 0:
        data = data[:max_dialogues]
    rows = []
    skipped_dialogues = 0
    with ZipFile(zip_path) as zip_file:
        for example in data:
            built = build_rows_for_dialogue(
                example,
                zip_file,
                split="all",
                turns_per_dialogue=turns_per_dialogue,
                dataset_name=dataset_name,
            )
            if built:
                rows.extend(built)
            else:
                skipped_dialogues += 1
    if not rows:
        raise RuntimeError("No MMDU rows were converted")
    print(
        f"Converted {len(rows)} rows from {len(data)} dialogues "
        f"(skipped_dialogues={skipped_dialogues}, turns={turns_per_dialogue})",
        flush=True,
    )
    return rows


def write_split(rows: list[dict], out_path: Path, split: str) -> None:
    fixed = []
    for index, row in enumerate(rows):
        row = dict(row)
        extra = dict(row["extra_info"])
        extra["split"] = split
        extra["index"] = index
        row["extra_info"] = extra
        fixed.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(fixed), out_path)
    print(f"Wrote {len(fixed)} rows -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/mmdu_raw_hf")
    parser.add_argument("--local_save_dir", default="data/mmdu_benchmark_small")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_FILES),
        default="benchmark",
        help="benchmark (110 dialogues) or 45k (~45k dialogues).",
    )
    parser.add_argument("--max_train_rows", type=int, default=64)
    parser.add_argument("--max_test_rows", type=int, default=16)
    parser.add_argument(
        "--max_dialogues",
        type=int,
        default=0,
        help="Cap dialogues scanned from JSON (0 = all). Useful for 45k smoke builds.",
    )
    parser.add_argument(
        "--turns_per_dialogue",
        type=int,
        default=1,
        help="Use the last N assistant turns per dialogue as separate rollout targets.",
    )
    parser.add_argument("--force_download", action="store_true")
    args = parser.parse_args()

    if args.turns_per_dialogue <= 0:
        raise ValueError("--turns_per_dialogue must be positive")
    raw_dir = Path(args.raw_dir)
    save_dir = Path(args.local_save_dir)
    json_path, zip_path = download_raw(raw_dir, args.dataset, args.force_download)

    max_dialogues = args.max_dialogues if args.max_dialogues > 0 else None
    rows = load_converted_rows(
        json_path,
        zip_path,
        turns_per_dialogue=args.turns_per_dialogue,
        dataset_name=args.dataset,
        max_dialogues=max_dialogues,
    )
    needed = args.max_train_rows + args.max_test_rows
    if len(rows) < needed:
        raise RuntimeError(
            f"Converted {len(rows)} rows, fewer than requested {needed}. "
            f"Increase --max_dialogues or --turns_per_dialogue."
        )
    train_rows = rows[: args.max_train_rows]
    test_rows = rows[args.max_train_rows : args.max_train_rows + args.max_test_rows]
    write_split(train_rows, save_dir / "train.parquet", "train")
    write_split(test_rows, save_dir / "test.parquet", "test")

    print(f"Raw files: {raw_dir} ({args.dataset})")
    print(f"Ready: {save_dir}")


if __name__ == "__main__":
    main()
