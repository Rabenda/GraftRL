# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""DeepEyes visual_toolbox_v2 agent loop for verl_vision + SGLang profiling."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from examples.profile.shared.agent.deepeyes_tools import (
    ToolParseResult,
    extract_answer,
    parse_tool_response,
    zoom_in_image,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

ZOOM_SUCCESS_OBS = (
    "Here is the cropped image returned after you called image_zoom_in_tool.\n"
    "If the image is sufficient, answer in <answer></answer>. "
    "Otherwise you may call the tool again."
)
ZOOM_FAILURE_OBS = (
    "OBSERVATION: image_zoom_in_tool failed. "
    "Please fix the bbox or answer directly in <answer></answer>."
)


def _image_dump_dir() -> Path | None:
    raw = os.environ.get("PROFILE_IMAGE_DUMP_DIR", "").strip()
    return Path(raw) if raw else None


def _dump_rollout_image(
    *,
    dump_dir: Path,
    uid: str,
    request_id: str,
    turn: int,
    role: str,
    image: Image.Image,
    rollout_idx: str | None = None,
    tool_success: bool | None = None,
) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    rel = f"{uid[:8]}_{request_id[:8]}_t{turn}_{role}.png"
    image.convert("RGB").save(dump_dir / rel)
    record = {
        "uid": uid,
        "request_id": request_id,
        "turn": turn,
        "role": role,
        "path": rel,
    }
    if rollout_idx is not None:
        record["rollout_idx"] = rollout_idx
    if tool_success is not None:
        record["tool_success"] = tool_success
    with (dump_dir / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@register("deepeyes_visual_toolbox_v2")
@register("deepeyes_agent")
class DeepEyesAgentLoop(AgentLoopBase):
    """Multi-turn DeepEyes zoom: parse <tool_call> JSON -> crop -> append cropped image."""

    def __init__(
        self,
        *args,
        tool_timeout_seconds: float = 10.0,
        max_tool_rounds: int = 5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.tool_timeout_seconds = tool_timeout_seconds
        self.max_tool_rounds = max(1, int(max_tool_rounds))

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        uid = str(kwargs.get("uid") or uuid4())
        global_step = kwargs.get("global_step")
        request_id = uuid4().hex
        rollout_idx = kwargs.get("index")
        dump_dir = _image_dump_dir()

        multi_modal_data = await self.process_multi_modal_info(messages)
        images = list(multi_modal_data.get("images") or [])
        videos = list(multi_modal_data.get("videos") or [])
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)
        if not images:
            raise ValueError("DeepEyes sample missing input image")

        origin_image = images[0]
        if dump_dir is not None:
            _dump_rollout_image(
                dump_dir=dump_dir,
                uid=uid,
                request_id=request_id,
                turn=0,
                role="deepeyes_input",
                image=origin_image,
                rollout_idx=str(rollout_idx) if rollout_idx is not None else None,
            )

        metrics: dict[str, Any] = {
            "tool_use_attempted": 0.0,
            "tool_use_success": 0.0,
        }
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images or None,
            videos=videos or None,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        response_mask: list[int] = []
        response_logprobs: list[float] | None = [] if sampling_params.get("logprobs") else None
        assistant_turns = 0
        user_turns = 0
        tool_rounds = 0
        tool_attempted = False
        tool_success = False
        zoom_source = "none"

        while True:
            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=images or None,
                    video_data=videos or None,
                    audio_data=audios,
                    mm_processor_kwargs=mm_processor_kwargs,
                    agent_turn=assistant_turns,
                    agent_uid=uid,
                    rollout_idx=str(rollout_idx) if rollout_idx is not None else None,
                    training_global_step=global_step,
                )

            current_resp_start = len(response_mask)
            response_ids_chunk = output.token_ids
            prompt_ids += response_ids_chunk
            response_mask += [1] * len(response_ids_chunk)
            response_logprobs = self._extend_logprobs(
                response_logprobs, current_resp_start, len(response_ids_chunk), output.log_probs
            )
            assistant_turns += 1

            raw_response = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(response_ids_chunk, skip_special_tokens=False),
            )

            if extract_answer(raw_response):
                break

            parse_result = parse_tool_response(raw_response)
            if parse_result.error_code == "NOTOOL":
                break

            if tool_rounds >= self.max_tool_rounds:
                break

            tool_attempted = True
            metrics["tool_use_attempted"] = 1.0
            for index in range(current_resp_start, current_resp_start + len(response_ids_chunk)):
                response_mask[index] = 0
            if response_logprobs:
                for index in range(current_resp_start, current_resp_start + len(response_ids_chunk)):
                    response_logprobs[index] = 0.0

            with simple_timer("tool_calls", metrics):
                edited_image, tool_ok = await self._run_tool_round(
                    parse_result=parse_result,
                    origin_image=origin_image,
                )
            tool_rounds += 1
            tool_success = tool_ok
            metrics["tool_use_success"] = float(tool_ok)
            zoom_source = "model" if tool_ok else "failed"

            observation_ids = await self._build_observation_ids(edited_image if tool_ok else None)
            if len(response_mask) + len(observation_ids) >= self.response_length:
                break

            prompt_ids += observation_ids
            response_mask += [0] * len(observation_ids)
            if response_logprobs:
                response_logprobs += [0.0] * len(observation_ids)

            if edited_image is not None and tool_ok:
                images.append(edited_image)
                if dump_dir is not None:
                    _dump_rollout_image(
                        dump_dir=dump_dir,
                        uid=uid,
                        request_id=request_id,
                        turn=len(images) - 1,
                        role="zoom_output",
                        image=edited_image,
                        rollout_idx=str(rollout_idx) if rollout_idx is not None else None,
                        tool_success=True,
                    )

            user_turns += 1

            if len(response_mask) >= self.response_length:
                break
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break

        if response_mask:
            response_ids = prompt_ids[-len(response_mask) :]
            prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]
        else:
            response_ids = []

        multi_modal_out: dict[str, Any] = {}
        if images:
            multi_modal_out["images"] = images
        if videos:
            multi_modal_out["videos"] = videos

        extra_fields = {
            "deepeyes_tool_attempted": tool_attempted,
            "deepeyes_tool_success": tool_success,
            "deepeyes_tool_rounds": tool_rounds,
            "deepeyes_zoom_source": zoom_source,
            "uid": uid,
            "request_id": request_id,
        }

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data=multi_modal_out,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
            extra_fields=extra_fields,
        )

    @staticmethod
    def _extend_logprobs(
        existing_logprobs: list[float] | None,
        current_response_len: int,
        chunk_len: int,
        chunk_logprobs: list[float] | None,
    ) -> list[float] | None:
        if existing_logprobs is None:
            return None
        if chunk_logprobs:
            if len(existing_logprobs) < current_response_len:
                existing_logprobs.extend([0.0] * (current_response_len - len(existing_logprobs)))
            existing_logprobs.extend(chunk_logprobs)
        else:
            existing_logprobs.extend([0.0] * chunk_len)
        return existing_logprobs

    async def _run_tool_round(
        self,
        *,
        parse_result: ToolParseResult,
        origin_image: Image.Image,
    ) -> tuple[Image.Image | None, bool]:
        if not parse_result.status:
            return None, False

        bbox = parse_result.arguments.get("bbox_2d")
        if bbox is None:
            bbox = parse_result.arguments.get("bbox")

        def _exec() -> tuple[Image.Image | None, bool]:
            cropped, msg = zoom_in_image(origin_image, list(bbox or []))
            if cropped is None:
                logger.warning("DeepEyes zoom failed: %s args=%s", msg, parse_result.arguments)
                return None, False
            return cropped, True

        try:
            return await asyncio.wait_for(
                self.loop.run_in_executor(None, _exec),
                timeout=self.tool_timeout_seconds,
            )
        except Exception as exc:
            logger.warning("DeepEyes tool exec failed: %s", exc)
            return None, False

    async def _build_observation_ids(self, edited_image: Image.Image | None) -> list[int]:
        if edited_image is not None:
            return await self.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": ZOOM_SUCCESS_OBS},
                        ],
                    }
                ],
                images=[edited_image],
                remove_system_prompt=True,
            )
        return await self.apply_chat_template(
            [{"role": "user", "content": ZOOM_FAILURE_OBS}],
            remove_system_prompt=True,
        )
