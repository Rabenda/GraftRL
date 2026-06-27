# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
Convert VTOOL/Refocus_Chart parquet to single-turn VLM RL format for verl_vision profiling.

Strips vtool multiturn tool instructions and keeps: image + question + geo3k-style answer format.
"""

from __future__ import annotations

import argparse
import os
import re

import pyarrow as pa
import pyarrow.parquet as pq

DATA_SOURCE = "VTOOL/Refocus_Chart"
USER_REQUEST_RE = re.compile(r"USER REQUEST:\s*<image>\s*(.+?)\s*$", re.DOTALL)

INSTRUCTION_FOLLOWING = (
    r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    r"The reasoning process MUST BE enclosed within <think> </think> tags. "
    r"The final answer MUST BE put in \boxed{}."
)


def extract_question(content: str, extra_info: dict) -> str:
    question = (extra_info or {}).get("question")
    if question:
        return str(question).strip()

    match = USER_REQUEST_RE.search(content)
    if match:
        return match.group(1).strip()

    if "<image>" in content:
        tail = content.split("<image>")[-1].strip()
        if tail:
            return tail[:2000]

    return content.strip()[:2000]


def convert_row(row: dict, idx: int, split: str) -> dict:
    extra_info = row.get("extra_info") or {}
    reward_model = row.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth") or extra_info.get("answer", "")
    question = extract_question(row["prompt"][0]["content"], extra_info)
    prompt_text = f"<image>{question} {INSTRUCTION_FOLLOWING}"

    return {
        "images": row["images"],
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": row.get("ability", "math"),
        "reward_model": {"style": "rule", "ground_truth": str(ground_truth)},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": str(ground_truth),
            "question": question,
            "source": row.get("source"),
            "chart_id": row.get("id"),
        },
    }


def convert_parquet(src: str, dst: str, max_rows: int | None = None) -> int:
    table = pq.read_table(src)
    rows_out = []
    n = table.num_rows
    limit = min(n, max_rows) if max_rows is not None else n
    split = "train" if "train" in os.path.basename(src) else "test"

    for idx in range(limit):
        row = {name: table.column(name)[idx].as_py() for name in table.column_names}
        rows_out.append(convert_row(row, idx, split))

    out_table = pa.Table.from_pylist(rows_out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pq.write_table(out_table, dst)
    return len(rows_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        default="/workspace/repo/verl_vision/data/refocus_chart_raw",
        help="Directory with train.parquet / test.parquet from VTOOL/Refocus_Chart",
    )
    parser.add_argument(
        "--local_save_dir",
        default="/workspace/repo/verl_vision/data/refocus_chart",
        help="Output directory for single-turn parquet",
    )
    parser.add_argument(
        "--max_train_rows",
        type=int,
        default=None,
        help="Optional cap for smoke tests (e.g. 512)",
    )
    parser.add_argument(
        "--max_test_rows",
        type=int,
        default=None,
        help="Optional cap for smoke tests",
    )
    args = parser.parse_args()

    for split, cap in [("train", args.max_train_rows), ("test", args.max_test_rows)]:
        src = os.path.join(args.src_dir, f"{split}.parquet")
        dst = os.path.join(args.local_save_dir, f"{split}.parquet")
        count = convert_parquet(src, dst, max_rows=cap)
        print(f"Wrote {count} rows -> {dst}")
