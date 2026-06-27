# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""Forced two-turn VLM agent loop for dummy crop smoke profiling."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _image_dump_dir() -> Path | None:
    raw = os.environ.get("PROFILE_IMAGE_DUMP_DIR", "").strip()
    return Path(raw) if raw else None


def _dump_rollout_image(
    *,
    dump_dir: Path,
    uid: str,
    request_id: str,
    turn: int,
    image_idx: int,
    role: str,
    image: Image.Image,
    rollout_idx: str | None = None,
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
    with (dump_dir / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def center_crop(image: Image.Image, crop_ratio: float) -> Image.Image:
    width, height = image.size
    crop_w = max(1, int(width * crop_ratio))
    crop_h = max(1, int(height * crop_ratio))
    left = max(0, (width - crop_w) // 2)
    top = max(0, (height - crop_h) // 2)
    return image.crop((left, top, left + crop_w, top + crop_h)).convert("RGB")


@register("dummy_crop_agent")
class DummyCropAgentLoop(AgentLoopBase):
    """Generate once, append a center crop as Image_1, then generate again."""

    def __init__(
        self,
        *args,
        crop_ratio: float = 0.5,
        first_turn_max_new_tokens: int = 32,
        second_user_text: str = "A dummy refocused crop has been generated. Use it with the original image and answer.",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.crop_ratio = crop_ratio
        self.first_turn_max_new_tokens = first_turn_max_new_tokens
        self.second_user_text = second_user_text
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    def _sampling_params(self, sampling_params: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
        params = dict(sampling_params)
        params.pop("max_tokens", None)
        params["max_new_tokens"] = max(1, max_new_tokens)
        return params

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        uid = str(kwargs.get("uid") or uuid4())
        dump_dir = _image_dump_dir()

        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        if not images:
            raise ValueError("dummy_crop_agent requires at least one input image")
        if not isinstance(images, list):
            images = [images]

        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        metrics = {}
        request_id = uuid4().hex
        rollout_idx = kwargs.get("index")
        if dump_dir is not None:
            for img_i, img in enumerate(images):
                _dump_rollout_image(
                    dump_dir=dump_dir,
                    uid=uid,
                    request_id=request_id,
                    turn=0,
                    image_idx=img_i,
                    role="input",
                    image=img,
                    rollout_idx=str(rollout_idx) if rollout_idx is not None else None,
                )

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
            )

        metrics["num_preempted"] = first.num_preempted if first.num_preempted is not None else -1
        full_ids = prompt_ids + first.token_ids
        response_mask = [1] * len(first.token_ids)
        response_logprobs = list(first.log_probs) if first.log_probs else None
        extra_fields = dict(first.extra_fields or {})

        crop = center_crop(images[0], self.crop_ratio)
        if dump_dir is not None:
            _dump_rollout_image(
                dump_dir=dump_dir,
                uid=uid,
                request_id=request_id,
                turn=1,
                image_idx=len(images),
                role="crop",
                image=crop,
                rollout_idx=str(rollout_idx) if rollout_idx is not None else None,
            )
        images = images + [crop]
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
            images=[crop],
            videos=None,
            audios=None,
            remove_system_prompt=True,
        )

        full_ids += add_ids
        response_mask += [0] * len(add_ids)
        if response_logprobs is not None:
            response_logprobs += [0.0] * len(add_ids)

        remaining_budget = self.response_length - len(response_mask)
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
                )

            metrics["num_preempted"] += second.num_preempted if second.num_preempted is not None else 0
            full_ids += second.token_ids
            response_mask += [1] * len(second.token_ids)
            if response_logprobs is not None:
                response_logprobs += second.log_probs if second.log_probs else [0.0] * len(second.token_ids)
            extra_fields.update(second.extra_fields or {})

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
                "turn_scores": [],
                "tool_rewards": [],
                "dummy_crop_agent": True,
                "uid": uid,
                "request_id": request_id,
            }
        )
        return output
