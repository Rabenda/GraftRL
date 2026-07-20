#!/usr/bin/env python3
"""Build synthetic OSWorld-style GUI trajectories for GraftRL offline replay.

No Docker required. Each trajectory has N screenshots that change locally
(a moving colored panel) so later CacheBlend / visual-reuse experiments have
a controllable adjacent-frame signal. Assistant labels are long Thought+Action
strings to stress decode time.
"""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from convert_osworld_traj import UITARS_SYSTEM_PROMPT  # noqa: E402


def make_frame(step: int, width: int, height: int) -> bytes:
    img = Image.new("RGB", (width, height), (32, 36, 48))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, 48], fill=(48, 52, 64))
    draw.rectangle([16, 14, width - 16, 34], fill=(70, 74, 88))
    x = 80 + (step * 37) % max(1, width - 280)
    y = 80 + (step * 23) % max(1, height - 220)
    draw.rectangle([x, y, x + 200, y + 140], fill=(40, 140, 220))
    draw.text((x + 20, y + 50), f"step-{step}", fill=(255, 255, 255))
    draw.rectangle([0, 48, 64, height], fill=(24, 28, 36))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_prediction(step: int, thought_chars: int) -> str:
    filler = (
        "I need to inspect the current screen carefully, locate the target UI "
        "element, and choose the next GUI action. "
    )
    thought = (filler * ((thought_chars // len(filler)) + 1))[:thought_chars]
    return (
        f"Thought: {thought}\n"
        f"Action: click(start_box='<|box_start|>({100 + step * 10},{200 + step * 5})"
        f"<|box_end|>')"
    )


def build_row(
    idx: int,
    *,
    n_steps: int,
    width: int,
    height: int,
    thought_chars: int,
    instruction: str,
) -> dict:
    screenshots = []
    predictions = []
    actions = []
    for step in range(n_steps):
        screenshots.append({"bytes": make_frame(step, width, height), "path": None})
        predictions.append(make_prediction(step, thought_chars))
        actions.append(f"click(step={step})")

    prompt = [
        {
            "role": "user",
            "content": UITARS_SYSTEM_PROMPT.format(instruction=instruction)
            + "\n\nCurrent screenshot:\n<image>",
        }
    ]
    return {
        "images": [screenshots[0]],
        "data_source": "osworld_gui_synth",
        "prompt": prompt,
        "ability": "gui_agent",
        "reward_model": {"style": "rule", "ground_truth": "1.0"},
        "extra_info": {
            "split": "train",
            "index": idx,
            "sample_id": f"synth:{idx}",
            "domain": "synth",
            "task_id": f"synth_{idx:04d}",
            "instruction": instruction,
            "num_turns": n_steps,
            "step_predictions": predictions,
            "step_actions": actions,
            "screenshots": screenshots,
            "source_dir": "",
            "need_tools_kwargs": False,
        },
    }


def write_split(rows: list[dict], out_path: Path, split: str) -> None:
    fixed = []
    for i, row in enumerate(rows):
        row = dict(row)
        extra = dict(row["extra_info"])
        extra["split"] = split
        extra["index"] = i
        row["extra_info"] = extra
        fixed.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(fixed), out_path)
    print(f"Wrote {len(fixed)} rows -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=256)
    parser.add_argument("--test-rows", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=8)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--thought-chars", type=int, default=800)
    args = parser.parse_args()

    instruction = (
        "Open the Documents folder, find report.xlsx, change the font of cell A1 "
        "to bold, and save the file."
    )
    train = [
        build_row(
            i,
            n_steps=args.n_steps,
            width=args.width,
            height=args.height,
            thought_chars=args.thought_chars,
            instruction=instruction,
        )
        for i in range(args.train_rows)
    ]
    test = [
        build_row(
            i,
            n_steps=args.n_steps,
            width=args.width,
            height=args.height,
            thought_chars=args.thought_chars,
            instruction=instruction,
        )
        for i in range(args.test_rows)
    ]
    write_split(train, args.output_dir / "train.parquet", "train")
    write_split(test, args.output_dir / "test.parquet", "test")
    meta = {
        "n_steps": args.n_steps,
        "thought_chars": args.thought_chars,
        "train_rows": args.train_rows,
        "test_rows": args.test_rows,
        "resolution": [args.width, args.height],
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Synthetic GUI pool ready under {args.output_dir}")


if __name__ == "__main__":
    main()
