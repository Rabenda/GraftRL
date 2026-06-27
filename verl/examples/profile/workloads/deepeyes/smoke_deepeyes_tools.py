#!/usr/bin/env python3
"""Dry-run DeepEyes zoom tool on a parquet sample (no GPU / no verl rollout)."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from examples.profile.shared.agent.deepeyes_tools import parse_tool_response, zoom_in_image


def _load_image(item: dict) -> Image.Image:
    return Image.open(BytesIO(item["bytes"])).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        default="/data/deepeyes_visual_toolbox_v2/train.parquet",
        type=Path,
    )
    parser.add_argument("--row", type=int, default=0)
    args = parser.parse_args()

    row = pq.read_table(args.parquet).slice(args.row, 1).to_pylist()[0]
    image = _load_image(row["images"][0])
    w, h = image.size
    print(f"row={args.row} data_source={row.get('data_source')} size={w}x{h}")
    print(f"question={(row.get('extra_info') or {}).get('question', '')[:120]}")

    bbox = [w * 0.2, h * 0.2, w * 0.6, h * 0.6]
    tool_text = (
        "<think>zoom center</think>\n"
        "<tool_call>\n"
        + json.dumps({"name": "image_zoom_in_tool", "arguments": {"bbox_2d": bbox}})
        + "\n</tool_call>"
    )
    parsed = parse_tool_response(tool_text)
    assert parsed.status, parsed.message
    cropped, msg = zoom_in_image(image, bbox)
    assert cropped is not None, msg
    print(f"zoom ok: input={image.size} crop={cropped.size}")

    bad = parse_tool_response("<answer>no tool</answer>")
    print(f"NOTOOL parse: {bad.error_code}")

    from examples.profile.workloads.deepeyes.deepeyes_agent_loop import DeepEyesAgentLoop

    print(f"agent registered: DeepEyesAgentLoop={DeepEyesAgentLoop.__name__}")
    print("smoke OK")


if __name__ == "__main__":
    main()
