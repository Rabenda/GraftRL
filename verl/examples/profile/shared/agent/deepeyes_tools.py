"""DeepEyes visual_toolbox_v2 tool helpers for verl_vision profile agent loop.

Ported from DeepEyes/verl/workers/agent/envs/mm_process_engine/visual_toolbox_v2.py
(zoom/crop only — no rotate in profile v1).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from math import ceil, floor
from typing import Any

from PIL import Image

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ToolParseResult:
    status: bool
    tool_name: str
    arguments: dict[str, Any]
    message: str
    error_code: str


def extract_answer(text: str) -> str | None:
    matches = _ANSWER_RE.findall(text or "")
    return matches[-1].strip() if matches else None


def extract_tool_call_raw(text: str) -> str | None:
    matches = _TOOL_CALL_RE.findall(text or "")
    return matches[-1].strip() if matches else None


def parse_tool_response(text: str) -> ToolParseResult:
    raw = extract_tool_call_raw(text)
    if not raw:
        return ToolParseResult(False, "", {}, "No tool call", "NOTOOL")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ToolParseResult(False, "", {}, f"Invalid JSON: {exc}", "BAD_JSON")

    name = str(payload.get("name") or "")
    args = payload.get("arguments") or {}
    if not isinstance(args, dict):
        return ToolParseResult(False, name, {}, "arguments must be an object", "BAD_ARGS")
    if name != "image_zoom_in_tool":
        return ToolParseResult(False, name, args, f"Unsupported tool: {name}", "UNKNOWN_TOOL")
    return ToolParseResult(True, name, args, "ok", "")


def validate_bbox(left: int, top: int, right: int, bottom: int) -> bool:
    try:
        if not (left < right and bottom > top):
            return False
        height = bottom - top
        width = right - left
        if max(height, width) / max(min(height, width), 1) > 100:
            return False
        if min(height, width) <= 30:
            return False
        return True
    except Exception:
        return False


def maybe_resize_bbox(
    left: int,
    top: int,
    right: int,
    bottom: int,
    *,
    width: int,
    height: int,
) -> list[int] | None:
    left = max(0, int(left))
    top = max(0, int(top))
    right = min(width, int(right))
    bottom = min(height, int(bottom))
    if not validate_bbox(left, top, right, bottom):
        return None

    box_h = bottom - top
    box_w = right - left
    if box_h >= 28 and box_w >= 28:
        return [left, top, right, bottom]

    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    ratio = 28 / min(box_h, box_w)
    new_half_height = ceil(box_h * ratio * 0.5)
    new_half_width = ceil(box_w * ratio * 0.5)
    new_left = floor(center_x - new_half_width)
    new_right = ceil(center_x + new_half_width)
    new_top = floor(center_y - new_half_height)
    new_bottom = ceil(center_y + new_half_height)
    if not validate_bbox(new_left, new_top, new_right, new_bottom):
        return None
    return [new_left, new_top, new_right, new_bottom]


def zoom_in_image(image: Image.Image, bbox_2d: list[Any]) -> tuple[Image.Image | None, str]:
    if len(bbox_2d) != 4:
        return None, "bbox_2d must have 4 numbers"
    width, height = image.size
    resized = maybe_resize_bbox(
        int(bbox_2d[0]),
        int(bbox_2d[1]),
        int(bbox_2d[2]),
        int(bbox_2d[3]),
        width=width,
        height=height,
    )
    if not resized:
        return None, "invalid bbox_2d"
    cropped = image.crop(tuple(resized))
    return cropped.convert("RGB"), "ok"
