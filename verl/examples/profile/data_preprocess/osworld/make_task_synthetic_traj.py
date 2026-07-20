#!/usr/bin/env python3
"""Build OSWorld task-backed synthetic GUI trajectories.

This is a bridge for machines that have OSWorld task JSONs but cannot run the
Docker/VM environment locally. It uses real OSWorld instructions, domains and
task ids, while rendering controllable synthetic screenshots for multi-turn VLM
rollout profiling. For real screenshots, use convert_osworld_traj.py on ARPO /
OSWorld result dumps.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from convert_osworld_traj import UITARS_SYSTEM_PROMPT  # noqa: E402


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.ImageFont:
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _short(text: str, width: int = 86, lines: int = 5) -> list[str]:
    wrapped = textwrap.wrap(" ".join(text.split()), width=width)
    return wrapped[:lines] or [""]


def load_tasks(task_root: Path, limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    roots = [task_root / "examples", task_root / "examples_windows"]
    for base in roots:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            instruction = str(data.get("instruction") or "").strip()
            task_id = str(data.get("id") or path.stem).strip()
            if not instruction or not task_id:
                continue
            rel = path.relative_to(task_root)
            tasks.append(
                {
                    "task_id": task_id,
                    "instruction": instruction,
                    "domain": path.parent.name,
                    "snapshot": str(data.get("snapshot") or ""),
                    "related_apps": [str(x) for x in data.get("related_apps") or []],
                    "source_path": str(rel),
                }
            )
    rng = random.Random(seed)
    rng.shuffle(tasks)
    if limit is not None and limit > 0:
        tasks = tasks[:limit]
    return tasks


def make_frame(task: dict[str, Any], step: int, width: int, height: int) -> bytes:
    bg = (27, 31, 39)
    panel = (245, 247, 250)
    chrome = (58, 65, 76)
    accent = (
        70 + (hash(task["domain"]) % 80),
        110 + (step * 17) % 90,
        150 + (hash(task["task_id"]) % 80),
    )
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    font_sm = _font(18)
    font_md = _font(24)
    font_lg = _font(30)

    draw.rectangle([0, 0, width, 52], fill=chrome)
    draw.text((18, 14), "OSWorld task replay", fill=(250, 250, 250), font=font_md)
    draw.text((width - 180, 16), f"turn {step + 1}", fill=(230, 235, 240), font=font_sm)
    draw.rectangle([0, 52, 78, height], fill=(36, 42, 52))
    for i, app in enumerate((task.get("related_apps") or [task["domain"]])[:6]):
        y = 78 + i * 56
        draw.rounded_rectangle([14, y, 64, y + 40], radius=6, fill=(72, 82, 96))
        draw.text((22, y + 10), app[:3].upper(), fill=(255, 255, 255), font=font_sm)

    margin = 110
    draw.rounded_rectangle([margin, 82, width - 48, height - 40], radius=8, fill=panel)
    draw.rectangle([margin, 82, width - 48, 128], fill=accent)
    title = f"{task['domain']} / {task.get('snapshot') or 'desktop'}"
    draw.text((margin + 20, 92), title[:72], fill=(255, 255, 255), font=font_lg)

    x0 = margin + 24
    y = 154
    draw.text((x0, y), f"Task ID: {task['task_id']}", fill=(30, 34, 42), font=font_sm)
    y += 38
    draw.text((x0, y), "Instruction", fill=(30, 34, 42), font=font_md)
    y += 36
    for line in _short(task["instruction"], width=92, lines=6):
        draw.text((x0, y), line, fill=(55, 62, 73), font=font_sm)
        y += 27

    box_x = margin + 40 + (step * 43) % max(1, width - margin - 360)
    box_y = height - 220 + (step * 19) % 80
    draw.rounded_rectangle([box_x, box_y, box_x + 260, box_y + 120], radius=8, fill=(238, 242, 246), outline=accent, width=4)
    draw.text((box_x + 18, box_y + 18), "Active window", fill=(35, 40, 48), font=font_md)
    draw.text((box_x + 18, box_y + 58), f"local change #{step}", fill=(80, 88, 100), font=font_sm)

    cursor_x = min(width - 42, box_x + 60 + step * 13)
    cursor_y = min(height - 50, box_y + 92)
    draw.polygon(
        [(cursor_x, cursor_y), (cursor_x + 28, cursor_y + 12), (cursor_x + 12, cursor_y + 18), (cursor_x + 20, cursor_y + 42)],
        fill=(20, 22, 28),
    )

    buf = BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def make_prediction(task: dict[str, Any], step: int, thought_chars: int) -> str:
    seed = (
        f"I need to solve the OSWorld task in the {task['domain']} environment. "
        f"The user instruction is: {task['instruction']} "
        "I will inspect visible controls, preserve context from previous turns, "
        "and choose a concrete GUI action that advances the task. "
    )
    thought = (seed * ((thought_chars // max(1, len(seed))) + 2))[:thought_chars]
    x = 140 + step * 23
    y = 180 + step * 17
    return f"Thought: {thought}\nAction: click(start_box='<|box_start|>({x},{y})<|box_end|>')"


def build_row(task: dict[str, Any], idx: int, *, split: str, n_steps: int, width: int, height: int, thought_chars: int) -> dict:
    screenshots = []
    predictions = []
    actions = []
    for step in range(n_steps):
        screenshots.append({"bytes": make_frame(task, step, width, height), "path": None})
        predictions.append(make_prediction(task, step, thought_chars))
        actions.append(f"click(step={step})")

    prompt = [
        {
            "role": "user",
            "content": UITARS_SYSTEM_PROMPT.format(instruction=task["instruction"])
            + "\n\nCurrent screenshot:\n<image>",
        }
    ]
    return {
        "images": [screenshots[0]],
        "data_source": "osworld_task_synth",
        "prompt": prompt,
        "ability": "gui_agent",
        "reward_model": {"style": "rule", "ground_truth": "1.0"},
        "extra_info": {
            "split": split,
            "index": idx,
            "sample_id": f"{task['domain']}:{task['task_id']}",
            "domain": task["domain"],
            "task_id": task["task_id"],
            "instruction": task["instruction"],
            "num_turns": n_steps,
            "step_predictions": predictions,
            "step_actions": actions,
            "screenshots": screenshots,
            "source_dir": task["source_path"],
            "snapshot": task.get("snapshot", ""),
            "related_apps": task.get("related_apps", []),
            "need_tools_kwargs": False,
        },
    }


def write_split(tasks: list[dict[str, Any]], out_path: Path, split: str, *, n_steps: int, width: int, height: int, thought_chars: int) -> None:
    rows = [
        build_row(task, i, split=split, n_steps=n_steps, width=width, height=height, thought_chars=thought_chars)
        for i, task in enumerate(tasks)
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out_path)
    print(f"Wrote {len(rows)} rows -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=256)
    parser.add_argument("--test-rows", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=8)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--thought-chars", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    needed = args.train_rows + args.test_rows
    tasks = load_tasks(args.task_root, limit=needed, seed=args.seed)
    if not tasks:
        raise SystemExit(f"No OSWorld task JSONs with instructions under {args.task_root}")
    if len(tasks) < needed:
        print(f"Warning: only {len(tasks)} tasks < requested {needed}; reusing tasks to fill splits.")
        repeated = []
        while len(repeated) < needed:
            repeated.extend(tasks)
        tasks = repeated[:needed]

    train = tasks[: args.train_rows]
    test = tasks[args.train_rows : args.train_rows + args.test_rows]
    write_split(train, args.output_dir / "train.parquet", "train", n_steps=args.n_steps, width=args.width, height=args.height, thought_chars=args.thought_chars)
    write_split(test, args.output_dir / "test.parquet", "test", n_steps=args.n_steps, width=args.width, height=args.height, thought_chars=args.thought_chars)
    meta = {
        "source": "osworld_task_synth",
        "task_root": str(args.task_root),
        "n_steps": args.n_steps,
        "thought_chars": args.thought_chars,
        "train_rows": len(train),
        "test_rows": len(test),
        "resolution": [args.width, args.height],
        "seed": args.seed,
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"OSWorld task-backed GUI pool ready under {args.output_dir}")


if __name__ == "__main__":
    main()
