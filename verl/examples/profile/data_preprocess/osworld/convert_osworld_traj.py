#!/usr/bin/env python3
"""Convert OSWorld / ARPO result directories into GraftRL parquet.

Expected per-task result layout (from OSWorld ``lib_run_single.py``)::

    <task_dir>/
      traj.jsonl
      step_reset_*.png          # step_num=0 screenshot
      step_1_*.png              # post-action screenshots
      plan_result*-step_*.txt   # optional Thought/Action text

Each parquet row is one trajectory:
  - ``images``: ordered screenshots used as per-turn observations
  - ``prompt``: initial user message with instruction + first ``<image>``
  - ``extra_info.step_actions`` / ``step_predictions``: recorded labels
  - ``extra_info.screenshots``: same images referenced by turn index

The GraftRL ``osworld_gui_agent`` loop then snowballs: model generates
Thought+Action, appends it, then feeds the next screenshot as a new user turn.
"""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


UITARS_SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='xxx')
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait()
finished(content='xxx')

## Note
- Use English in `Thought` and `Action` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""


def load_png_bytes(path: Path, *, max_side: int | None = None) -> dict:
    img = Image.open(path).convert("RGB")
    if max_side is not None:
        w, h = img.size
        scale = min(1.0, float(max_side) / max(w, h))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return {"bytes": buf.getvalue(), "path": None}


def parse_traj_jsonl(traj_path: Path) -> list[dict[str, Any]]:
    steps = []
    with open(traj_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            steps.append(json.loads(line))
    steps.sort(key=lambda s: int(s.get("step_num", 0)))
    return steps


def find_result_dirs(results_root: Path) -> list[Path]:
    """Find leaf task dirs that contain traj.jsonl."""
    return sorted({p.parent for p in results_root.rglob("traj.jsonl")})


def convert_one_task(
    task_dir: Path,
    *,
    max_steps: int,
    max_side: int | None,
    domain: str = "",
    task_id: str = "",
) -> dict[str, Any] | None:
    traj_path = task_dir / "traj.jsonl"
    if not traj_path.exists():
        return None
    steps = parse_traj_jsonl(traj_path)
    if not steps:
        return None

    reset = next((s for s in steps if int(s.get("step_num", -1)) == 0), None)
    action_steps = [s for s in steps if int(s.get("step_num", -1)) > 0]
    if reset is None:
        return None

    instruction = str(reset.get("instruction") or "").strip()
    if not instruction:
        # Some dumps only put instruction on step 0; fall back to sibling meta.
        meta = task_dir / "instruction.txt"
        if meta.exists():
            instruction = meta.read_text(encoding="utf-8").strip()
    if not instruction:
        instruction = f"OSWorld task {task_dir.name}"

    # Observation for turn t is the screenshot *before* action t.
    # step_0 = initial obs; step_k screenshot is post-action state → obs for turn k+1.
    obs_files: list[str] = [str(reset.get("screenshot_file") or "")]
    predictions: list[str] = []
    actions: list[str] = []
    for s in action_steps:
        pred = str(s.get("prediction") or s.get("plan_result_full") or s.get("plan_result") or "")
        if not pred:
            # optional sidecar text files
            for pattern in (
                f"plan_result_full-step_{s['step_num']}_*.txt",
                f"plan_result-step_{s['step_num']}_*.txt",
            ):
                hits = list(task_dir.glob(pattern))
                if hits:
                    pred = hits[0].read_text(encoding="utf-8")
                    break
        predictions.append(pred)
        actions.append(str(s.get("action") or ""))
        # next observation comes from this step's screenshot (after env.step)
        obs_files.append(str(s.get("screenshot_file") or ""))

    # Keep at most max_steps observations (turns). Need N obs for N generates.
    obs_files = [f for f in obs_files if f][:max_steps]
    if len(obs_files) < 2:
        # allow single-obs trajectories (1 turn)
        if len(obs_files) < 1:
            return None

    images = []
    for name in obs_files:
        path = task_dir / name
        if not path.exists():
            # fuzzy match by prefix
            hits = list(task_dir.glob(name.split("@")[0] + "*.png")) if "@" in name else []
            if not hits:
                hits = list(task_dir.glob(Path(name).stem + "*.png"))
            if not hits:
                return None
            path = hits[0]
        images.append(load_png_bytes(path, max_side=max_side))

    n_turns = len(images)
    predictions = predictions[: max(0, n_turns)]
    actions = actions[: max(0, n_turns)]

    # Reward from result.txt if present
    reward = 0.0
    result_file = task_dir / "result.txt"
    if result_file.exists():
        try:
            reward = float(result_file.read_text(encoding="utf-8").strip().split()[0])
        except Exception:
            reward = 0.0

    prompt = [
        {
            "role": "user",
            "content": UITARS_SYSTEM_PROMPT.format(instruction=instruction)
            + "\n\nCurrent screenshot:\n<image>",
        }
    ]

    # RLHFDataset requires len(images) == number of <image> placeholders in prompt.
    # Keep only the first screenshot in ``images``; all frames live in extra_info
    # for the GUI agent loop to consume turn-by-turn.
    return {
        "images": [images[0]],
        "data_source": "osworld_gui",
        "prompt": prompt,
        "ability": "gui_agent",
        "reward_model": {"style": "rule", "ground_truth": str(reward)},
        "extra_info": {
            "split": "train",
            "index": -1,
            "sample_id": f"{domain}:{task_id or task_dir.name}",
            "domain": domain,
            "task_id": task_id or task_dir.name,
            "instruction": instruction,
            "num_turns": n_turns,
            "step_predictions": predictions,
            "step_actions": actions,
            "screenshots": images,
            "source_dir": str(task_dir),
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
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root of OSWorld/ARPO result dirs containing traj.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--max-side", type=int, default=1280, help="Resize long side; 0=keep")
    parser.add_argument("--max-train-rows", type=int, default=256)
    parser.add_argument("--max-test-rows", type=int, default=64)
    args = parser.parse_args()

    max_side = None if int(args.max_side) <= 0 else int(args.max_side)
    task_dirs = find_result_dirs(args.results_root)
    if not task_dirs:
        raise SystemExit(f"No traj.jsonl under {args.results_root}")

    rows: list[dict] = []
    for task_dir in task_dirs:
        # Heuristic domain/task_id from path: .../<domain>/<task_id>/
        parts = task_dir.parts
        domain = parts[-2] if len(parts) >= 2 else ""
        task_id = parts[-1]
        row = convert_one_task(
            task_dir,
            max_steps=args.max_steps,
            max_side=max_side,
            domain=domain,
            task_id=task_id,
        )
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit("Converted 0 trajectories (check screenshot paths in traj.jsonl)")

    needed = args.max_train_rows + args.max_test_rows
    if len(rows) < needed:
        print(
            f"Warning: only {len(rows)} trajectories < requested {needed}; "
            f"will split what we have."
        )
    train_n = min(args.max_train_rows, len(rows))
    test_n = min(args.max_test_rows, max(0, len(rows) - train_n))
    write_split(rows[:train_n], args.output_dir / "train.parquet", "train")
    write_split(
        rows[train_n : train_n + test_n] if test_n else rows[:1],
        args.output_dir / "test.parquet",
        "test",
    )
    print(f"Done. train={train_n} test={test_n or 1} from {len(task_dirs)} result dirs")


if __name__ == "__main__":
    main()
