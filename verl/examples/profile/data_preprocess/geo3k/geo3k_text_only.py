#!/usr/bin/env python3
"""Build padded text-only Geo3K parquet for fair profiling vs image+text run.

Strips images and <image> tags, then pads each prompt so total token count matches
the corresponding image+text requests in a reference generate log (same text stem,
matched by text_prompt_tokens → target prompt_tokens).

Usage:
  python examples/profile/data_preprocess/geo3k/geo3k_text_only.py \\
    --match-generate-log profile_logs_geo3k_full/verl_sglang_generate_log_geo3k_full_bs64_n4.csv

  python examples/profile/data_preprocess/geo3k/geo3k_text_only.py \\
    --src-dir data/geo3k \\
    --out-dir data/geo3k_text_only_padded \\
    --match-generate-log profile_logs_geo3k_full/verl_sglang_generate_log_geo3k_full_bs64_n4.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import statistics as st
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

_IMAGE_TAG_RE = re.compile(r"<image>|<video>|<audio>")
_PAD_SNIPPET = (
    " Additional neutral padding for controlled profiling: this text does not change "
    "the geometry task. "
)

_DEFAULT_GENERATE_LOG = (
    "/workspace/repo/verl_vision/profile_logs_geo3k_full/"
    "verl_sglang_generate_log_geo3k_full_bs64_n4.csv"
)
_DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def _strip_mm_tags(text: str) -> str:
    s = _IMAGE_TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", s).strip()


def _prompt_content(prompt) -> str:
    if isinstance(prompt, list) and prompt:
        content = prompt[0].get("content", "")
        return content if isinstance(content, str) else str(content)
    return str(prompt)


def _load_tokenizer(model_path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit("transformers required for token-accurate padding") from e
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_target_map(generate_log_path: str) -> tuple[dict[int, int], dict]:
    """Map text_prompt_tokens -> median total prompt_tokens (image+text run)."""
    rows = []
    with open(generate_log_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("has_image") not in ("1", "True", "true"):
                continue
            rows.append(r)
    if not rows:
        raise SystemExit(f"No has_image=1 rows in {generate_log_path}")

    buckets: dict[int, list[int]] = defaultdict(list)
    all_prompt: list[int] = []
    for r in rows:
        tt = int(float(r["text_prompt_tokens"]))
        pt = int(float(r["prompt_tokens"]))
        buckets[tt].append(pt)
        all_prompt.append(pt)

    target_by_text: dict[int, int] = {
        tt: int(round(st.median(pts))) for tt, pts in buckets.items()
    }
    global_target = int(round(st.median(all_prompt)))
    meta = {
        "generate_log": generate_log_path,
        "image_rows": len(rows),
        "unique_text_prompt_tokens": len(target_by_text),
        "global_median_prompt_tokens": global_target,
        "target_by_text_prompt_tokens": target_by_text,
    }
    return target_by_text, meta


def pad_to_token_target(tokenizer, content: str, target_tokens: int) -> str:
    text = content
    n = _count_tokens(tokenizer, text)
    guard = 0
    while n < target_tokens and guard < 5000:
        text += _PAD_SNIPPET
        n = _count_tokens(tokenizer, text)
        guard += 1
    if n > target_tokens:
        # Rare: trim by removing padding suffix in chunks
        while n > target_tokens and _PAD_SNIPPET in text:
            text = text[: -len(_PAD_SNIPPET)]
            n = _count_tokens(tokenizer, text)
    return text


def _text_only_padded_row(
    row: dict,
    tokenizer,
    target_by_text: dict[int, int],
    global_target: int,
) -> dict:
    out = copy.deepcopy(row)
    out["images"] = []

    plain = _strip_mm_tags(_prompt_content(out.get("prompt")))
    text_tok = _count_tokens(tokenizer, plain)
    target_total = target_by_text.get(text_tok, global_target)
    padded = pad_to_token_target(tokenizer, plain, target_total)
    actual_tok = _count_tokens(tokenizer, padded)

    if isinstance(out.get("prompt"), list) and out["prompt"]:
        out["prompt"] = [{**out["prompt"][0], "content": padded}]
    extra = dict(out.get("extra_info") or {})
    if "question" in extra and isinstance(extra["question"], str):
        extra["question"] = _strip_mm_tags(extra["question"])
    extra["text_only"] = True
    extra["padded_for_profiling"] = True
    extra["target_prompt_tokens"] = target_total
    extra["actual_prompt_tokens"] = actual_tok
    extra["text_prompt_tokens_before_pad"] = text_tok
    out["extra_info"] = extra
    return out


def convert_parquet(
    src: str,
    dst: str,
    tokenizer,
    target_by_text: dict[int, int],
    global_target: int,
    max_rows: int | None,
) -> tuple[int, list[int]]:
    table = pq.read_table(src)
    n = table.num_rows
    if max_rows is not None and max_rows > 0:
        n = min(n, max_rows)
        table = table.slice(0, n)

    out_rows = []
    actual_tokens: list[int] = []
    for row in table.to_pylist():
        out = _text_only_padded_row(row, tokenizer, target_by_text, global_target)
        actual_tokens.append(int(out["extra_info"]["actual_prompt_tokens"]))
        out_rows.append(out)

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out_rows), dst)
    return n, actual_tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Geo3K text-only parquet with prompt length matched to image run"
    )
    parser.add_argument("--src-dir", default="/workspace/repo/verl_vision/data/geo3k")
    parser.add_argument(
        "--out-dir",
        default="/workspace/repo/verl_vision/data/geo3k_text_only_padded",
    )
    parser.add_argument(
        "--match-generate-log",
        default=_DEFAULT_GENERATE_LOG,
        help="Image+text profiling generate log for per-text_token target lengths",
    )
    parser.add_argument("--model-path", default=_DEFAULT_MODEL)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isfile(args.match_generate_log):
        raise SystemExit(
            f"Missing {args.match_generate_log}. Run geo3k_full profiling first."
        )

    target_by_text, meta = build_target_map(args.match_generate_log)
    global_target = meta["global_median_prompt_tokens"]
    tokenizer = _load_tokenizer(args.model_path)
    max_rows = args.max_rows if args.max_rows > 0 else None

    all_actual: list[int] = []
    for split in ("train", "test"):
        src = os.path.join(args.src_dir, f"{split}.parquet")
        dst = os.path.join(args.out_dir, f"{split}.parquet")
        if not os.path.isfile(src):
            raise SystemExit(f"Missing {src}")
        count, actual = convert_parquet(
            src, dst, tokenizer, target_by_text, global_target, max_rows
        )
        all_actual.extend(actual)
        print(f"Wrote {dst} ({count} rows)")

    meta["actual_prompt_tokens_train_stats"] = {
        "mean": round(st.mean(all_actual), 1),
        "median": round(st.median(all_actual), 1),
        "min": min(all_actual),
        "max": max(all_actual),
    }
    meta_path = os.path.join(args.out_dir, "padding_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Wrote {meta_path}")
    print(
        f"Padded token stats: mean={meta['actual_prompt_tokens_train_stats']['mean']} "
        f"median={meta['actual_prompt_tokens_train_stats']['median']} "
        f"(image run global median target={global_target})"
    )


if __name__ == "__main__":
    main()
