# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""Geo3K refocus-style two-turn agent loop for E+P-heavy profiling."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image, ImageDraw, ImageEnhance

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _image_dump_dir() -> Path | None:
    raw = os.environ.get("PROFILE_IMAGE_DUMP_DIR", "").strip()
    return Path(raw) if raw else None


def _dump_image(
    *,
    dump_dir: Path,
    uid: str,
    request_id: str,
    turn: int,
    image_idx: int,
    role: str,
    image: Image.Image,
    rollout_idx: str | None = None,
    variant: str | None = None,
) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    rel = f"{uid[:8]}_{request_id[:8]}_t{turn}_i{image_idx}_{role}.png"
    image.convert("RGB").save(dump_dir / rel)
    record = {
        "uid": uid,
        "request_id": request_id,
        "turn": turn,
        "image_idx": image_idx,
        "role": role,
        "path": rel,
    }
    if rollout_idx is not None:
        record["rollout_idx"] = rollout_idx
    if variant:
        record["variant"] = variant
    with (dump_dir / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _center_roi(width: int, height: int, ratio: float) -> tuple[int, int, int, int]:
    roi_w = max(1, int(width * ratio))
    roi_h = max(1, int(height * ratio))
    left = max(0, (width - roi_w) // 2)
    top = max(0, (height - roi_h) // 2)
    return left, top, left + roi_w, top + roi_h


def build_full_canvas_refocus(
    image: Image.Image,
    *,
    mode: str,
    rollout_idx: str | None,
    roi_ratio: float,
    dim_alpha: int,
) -> Image.Image:
    """Return a same-size image with a highlighted center ROI.

    exact: identical for every branch of the same input.
    diversified: branch-dependent color/width but still visually close.
    """

    base = image.convert("RGB")
    width, height = base.size
    box = _center_roi(width, height, roi_ratio)

    dimmed = ImageEnhance.Brightness(base).enhance(0.45)
    out = dimmed.copy()
    out.paste(base.crop(box), box)

    draw = ImageDraw.Draw(out, "RGBA")
    palette = [(255, 48, 48, 210), (255, 176, 0, 200), (40, 180, 110, 200), (45, 125, 255, 200)]
    idx = 0
    if mode == "diversified" and rollout_idx is not None:
        try:
            idx = int(rollout_idx) % len(palette)
        except ValueError:
            idx = sum(ord(ch) for ch in rollout_idx) % len(palette)
    color = palette[idx if mode == "diversified" else 0]
    width_px = max(3, min(width, height) // (90 if mode == "diversified" else 80))

    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay, "RGBA")
    overlay_draw.rectangle((0, 0, width, height), fill=(0, 0, 0, max(0, min(255, dim_alpha))))
    overlay_draw.rectangle(box, fill=(0, 0, 0, 0))
    out = Image.alpha_composite(out.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(out, "RGBA")

    for i in range(width_px):
        inset = i
        draw.rectangle((box[0] - inset, box[1] - inset, box[2] + inset, box[3] + inset), outline=color)
    return out.convert("RGB")


@register("geo3k_refocus_agent")
class Geo3KRefocusAgentLoop(AgentLoopBase):
    """Generate a short first turn, append a full-canvas refocus image, then answer."""

    def __init__(
        self,
        *args,
        refocus_mode: str = "exact",
        roi_ratio: float = 0.72,
        dim_alpha: int = 0,
        first_turn_max_new_tokens: int = 16,
        final_turn_max_new_tokens: int = 64,
        second_user_text: str = "A refocused version of the same geometry image is provided. Use it with the original image and give the final boxed answer.",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.refocus_mode = os.environ.get("GEO3K_REFOCUS_MODE", refocus_mode).strip() or refocus_mode
        self.roi_ratio = float(os.environ.get("GEO3K_REFOCUS_ROI_RATIO", roi_ratio))
        self.dim_alpha = int(os.environ.get("GEO3K_REFOCUS_DIM_ALPHA", dim_alpha))
        self.first_turn_max_new_tokens = int(
            os.environ.get("GEO3K_REFOCUS_FIRST_TURN_MAX_NEW_TOKENS", first_turn_max_new_tokens)
        )
        self.final_turn_max_new_tokens = int(
            os.environ.get("GEO3K_REFOCUS_FINAL_TURN_MAX_NEW_TOKENS", final_turn_max_new_tokens)
        )
        self.second_user_text = os.environ.get("GEO3K_REFOCUS_SECOND_USER_TEXT", second_user_text)

    def _sampling_params(self, sampling_params: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
        params = dict(sampling_params)
        params.pop("max_tokens", None)
        params["max_new_tokens"] = max(1, int(max_new_tokens))
        return params

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        uid = str(kwargs.get("uid") or uuid4())
        request_id = uuid4().hex
        rollout_idx = str(kwargs.get("index")) if kwargs.get("index") is not None else None
        dump_dir = _image_dump_dir()
        extra_info = kwargs.get("extra_info") or {}
        mode = str(extra_info.get("geo3k_refocus_mode") or self.refocus_mode)
        variant = str(extra_info.get("geo3k_refocus_variant") or mode)

        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        if not images:
            raise ValueError("geo3k_refocus_agent requires at least one input image")
        if not isinstance(images, list):
            images = [images]

        if dump_dir is not None:
            for img_i, img in enumerate(images):
                _dump_image(
                    dump_dir=dump_dir,
                    uid=uid,
                    request_id=request_id,
                    turn=0,
                    image_idx=img_i,
                    role="input",
                    image=img,
                    rollout_idx=rollout_idx,
                    variant=variant,
                )

        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        metrics: dict[str, Any] = {"geo3k_refocus": 1.0}
        response_mask: list[int] = []
        response_logprobs: list[float] | None = [] if sampling_params.get("logprobs") else None
        first_budget = min(self.first_turn_max_new_tokens, max(1, self.response_length // 4))

        with simple_timer("generate_sequences", metrics):
            first: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=self._sampling_params(sampling_params, first_budget),
                image_data=images,
                video_data=videos,
                audio_data=audios,
                mm_processor_kwargs=mm_processor_kwargs,
                agent_turn=0,
                agent_uid=uid,
                rollout_idx=rollout_idx,
            )

        full_ids = prompt_ids + first.token_ids
        response_mask += [1] * len(first.token_ids)
        if response_logprobs is not None:
            response_logprobs += first.log_probs if first.log_probs else [0.0] * len(first.token_ids)
        extra_fields = dict(first.extra_fields or {})
        metrics["num_preempted"] = first.num_preempted if first.num_preempted is not None else -1

        refocus = build_full_canvas_refocus(
            images[0],
            mode=mode,
            rollout_idx=rollout_idx,
            roi_ratio=self.roi_ratio,
            dim_alpha=self.dim_alpha,
        )
        if dump_dir is not None:
            _dump_image(
                dump_dir=dump_dir,
                uid=uid,
                request_id=request_id,
                turn=1,
                image_idx=len(images),
                role="refocus",
                image=refocus,
                rollout_idx=rollout_idx,
                variant=variant,
            )

        images = images + [refocus]
        add_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.second_user_text},
                ],
            }
        ]
        add_ids = await self.apply_chat_template(
            add_messages,
            images=[refocus],
            videos=None,
            audios=None,
            remove_system_prompt=True,
        )
        full_ids += add_ids
        response_mask += [0] * len(add_ids)
        if response_logprobs is not None:
            response_logprobs += [0.0] * len(add_ids)

        remaining_budget = min(self.final_turn_max_new_tokens, self.response_length - len(response_mask))
        if remaining_budget > 0:
            with simple_timer("generate_sequences_second_turn", metrics):
                second: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=full_ids,
                    sampling_params=self._sampling_params(sampling_params, remaining_budget),
                    image_data=images,
                    video_data=videos,
                    audio_data=audios,
                    mm_processor_kwargs=mm_processor_kwargs,
                    agent_turn=1,
                    agent_uid=uid,
                    rollout_idx=rollout_idx,
                )
            full_ids += second.token_ids
            response_mask += [1] * len(second.token_ids)
            if response_logprobs is not None:
                response_logprobs += second.log_probs if second.log_probs else [0.0] * len(second.token_ids)
            extra_fields.update(second.extra_fields or {})
            metrics["num_preempted"] += second.num_preempted if second.num_preempted is not None else 0

        response_ids = full_ids[-len(response_mask) :]
        prompt_ids = full_ids[: len(full_ids) - len(response_mask)]
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data={"images": images, **({"videos": videos} if videos is not None else {})},
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=metrics,
            extra_fields=extra_fields,
        )
        output.extra_fields.update(
            {
                "geo3k_refocus_agent": True,
                "geo3k_refocus_variant": variant,
                "geo3k_refocus_mode": mode,
                "uid": uid,
                "request_id": request_id,
            }
        )
        return output
