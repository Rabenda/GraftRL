#!/usr/bin/env python3
"""Export dataset images + prompts to a browsable HTML gallery."""

from __future__ import annotations

import argparse
import html
import io
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def _load_image(entry) -> Image.Image:
    if isinstance(entry, Image.Image):
        return entry.convert("RGB")
    if isinstance(entry, dict):
        if entry.get("bytes"):
            return Image.open(io.BytesIO(entry["bytes"])).convert("RGB")
        path = entry.get("path")
        if path:
            return Image.open(path).convert("RGB")
    if isinstance(entry, (bytes, bytearray)):
        return Image.open(io.BytesIO(entry)).convert("RGB")
    if isinstance(entry, str):
        return Image.open(entry).convert("RGB")
    raise TypeError(f"Unsupported image entry type: {type(entry)!r}")


def _prompt_text(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for msg in prompt:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_bits = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_bits.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_bits.append(block)
                content = "\n".join(text_bits)
            parts.append(f"[{role}]\n{content}")
        return "\n\n".join(parts)
    return str(prompt)


def export_parquet(
    parquet_path: str,
    out_dir: str,
    max_rows: int | None = None,
    split_name: str | None = None,
) -> Path:
    table = pq.read_table(parquet_path)
    out = Path(out_dir)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    n = table.num_rows if max_rows is None else min(max_rows, table.num_rows)
    split = split_name or Path(parquet_path).stem
    rows_meta = []

    for idx in range(n):
        row = {name: table.column(name)[idx].as_py() for name in table.column_names}
        prompt = _prompt_text(row.get("prompt", ""))
        extra = row.get("extra_info") or {}
        images = row.get("images") or []
        image_paths = []
        for img_i, img_entry in enumerate(images):
            img = _load_image(img_entry)
            rel = f"images/{split}_row{idx:04d}_img{img_i}.png"
            img.save(out / rel)
            image_paths.append(rel)

        rows_meta.append(
            {
                "row": idx,
                "split": split,
                "data_source": row.get("data_source"),
                "ability": row.get("ability"),
                "question": extra.get("question"),
                "answer": extra.get("answer") or (row.get("reward_model") or {}).get("ground_truth"),
                "image_count": len(image_paths),
                "image_paths": image_paths,
                "prompt": prompt,
            }
        )

    meta_path = out / f"{split}_samples.json"
    meta_path.write_text(json.dumps(rows_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    cards = []
    for item in rows_meta:
        imgs_html = "".join(
            f'<img src="{html.escape(p)}" alt="row{item["row"]} img{i}" loading="lazy" />'
            for i, p in enumerate(item["image_paths"])
        )
        cards.append(
            f"""
<section class="card">
  <h2>Row {item['row']} · {html.escape(str(item.get('data_source') or ''))} · {item['image_count']} image(s)</h2>
  <div class="imgs">{imgs_html}</div>
  <pre class="prompt">{html.escape(item['prompt'])}</pre>
  <p><b>Q:</b> {html.escape(str(item.get('question') or ''))}</p>
  <p><b>A:</b> {html.escape(str(item.get('answer') or ''))}</p>
</section>
"""
        )

    index_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>Dataset samples · {html.escape(split)}</title>
  <style>
    body {{ font-family: sans-serif; max-width: 1100px; margin: 24px auto; padding: 0 16px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 24px; }}
    .imgs img {{ max-width: 420px; max-height: 420px; border: 1px solid #ccc; margin-right: 8px; }}
    pre.prompt {{ white-space: pre-wrap; background: #f7f7f7; padding: 12px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>{html.escape(split)} · {n} rows</h1>
  <p>Source: {html.escape(parquet_path)}</p>
  {''.join(cards)}
</body>
</html>
"""
    index_path = out / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    print(f"Wrote {n} rows -> {out}")
    print(f"  HTML: {index_path}")
    print(f"  JSON: {meta_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export parquet VLM samples to HTML + PNG.")
    parser.add_argument("--parquet", required=True, help="Path to train/test parquet")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--split-name", default=None)
    args = parser.parse_args()
    export_parquet(args.parquet, args.out_dir, max_rows=args.max_rows, split_name=args.split_name)


if __name__ == "__main__":
    main()
