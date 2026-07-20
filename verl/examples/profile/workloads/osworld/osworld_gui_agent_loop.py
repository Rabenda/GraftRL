# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""OSWorld / ARPO-style GUI snowball agent loop for GraftRL profiling.

Offline rows carry an instruction + ordered ``extra_info.screenshots`` (one
observation per turn).  At runtime:

1. Turn 0: instruction user message (already in ``raw_prompt``) with screenshot 0.
2. Model generates Thought+Action (long decode).
3. Append model text; append next screenshot as a new user message.
4. Repeat until ``min(num_screenshots, max_runtime_turns)``.

Returned ``prompt_ids`` / ``response_ids`` are **final-turn only** (reward uses
trajectory GT). Intermediate turns still call ``server_manager.generate`` with
``agent_turn`` for CacheBlend / long-context prefill profiling.
"""

from __future__ import annotations

import logging
import os
from io import BytesIO
from typing import Any
from uuid import uuid4

from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _to_pil(item: Any) -> Image.Image:
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, dict):
        if item.get("bytes") is not None:
            return Image.open(BytesIO(item["bytes"])).convert("RGB")
        if item.get("image") is not None:
            return _to_pil(item["image"])
        if item.get("path"):
            return Image.open(item["path"]).convert("RGB")
    if isinstance(item, (bytes, bytearray)):
        return Image.open(BytesIO(item)).convert("RGB")
    raise TypeError(f"Unsupported screenshot type: {type(item)}")


def load_screenshots(extra_info: dict, fallback_images: list | None) -> list[Image.Image]:
    raw = extra_info.get("screenshots")
    if raw is None:
        raw = fallback_images or []
    # numpy object arrays from DataProto
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    return [_to_pil(x) for x in list(raw)]


@register("osworld_gui_agent")
class OSWorldGUIAgentLoop(AgentLoopBase):
    """Snowball GUI rollout over ordered screenshots (offline OSWorld dump)."""

    def __init__(
        self,
        *args,
        max_runtime_turns: int = 8,
        intermediate_max_new_tokens: int = 512,
        final_max_new_tokens: int = 1024,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length
        self.max_runtime_turns = int(
            os.environ.get("OSWORLD_RUNTIME_TURNS", max_runtime_turns)
        )
        self.intermediate_max_new_tokens = int(
            os.environ.get(
                "OSWORLD_INTERMEDIATE_MAX_NEW_TOKENS", intermediate_max_new_tokens
            )
        )
        self.final_max_new_tokens = int(
            os.environ.get("OSWORLD_FINAL_MAX_NEW_TOKENS", final_max_new_tokens)
        )

    def _sampling_params(self, sampling_params: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
        params = dict(sampling_params)
        params.pop("max_tokens", None)
        params["max_new_tokens"] = max(1, int(max_new_tokens))
        return params

    async def _decode_assistant(self, token_ids: list[int]) -> str:
        return await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(token_ids, skip_special_tokens=True),
        )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        uid = str(kwargs.get("uid") or uuid4())
        request_id = uuid4().hex
        rollout_idx = str(kwargs.get("index")) if kwargs.get("index") is not None else None
        global_step = kwargs.get("global_step")
        extra_info = kwargs.get("extra_info") or {}
        if hasattr(extra_info, "item"):
            extra_info = extra_info.item()
        if not isinstance(extra_info, dict):
            extra_info = dict(extra_info)

        screenshots = load_screenshots(extra_info, kwargs.get("images"))
        if not screenshots:
            raise ValueError("osworld_gui_agent requires at least one screenshot")

        n_turns = min(len(screenshots), max(1, self.max_runtime_turns))
        live_messages: list[dict] = [dict(m) for m in messages]
        # Ensure first user message references the first screenshot via structured content
        # if dataset already converted <image>; otherwise keep string form.
        full_ids: list[int] = []
        extra_fields: dict[str, Any] = {}
        turn_stats: list[dict[str, Any]] = []
        mm_processor_kwargs: dict[str, Any] = {}
        images_for_mm: list[Image.Image] = []

        final_prompt_ids: list[int] | None = None
        final_response_ids: list[int] | None = None
        final_response_logprobs: list[float] | None = None
        response_logprobs_enabled = bool(sampling_params.get("logprobs"))
        metrics: dict[str, Any] = {
            "osworld_gui": 1.0,
            "osworld_runtime_turns": float(n_turns),
        }

        for turn_rank in range(n_turns):
            is_final = turn_rank == n_turns - 1
            if is_final:
                budget = min(self.final_max_new_tokens, self.response_length)
            else:
                # Intermediate generations become part of the next prompt, not the
                # trainer-visible response. Keep their decode budget independent so
                # long-horizon profiling really exercises long decode on every turn.
                budget = self.intermediate_max_new_tokens
            if budget <= 0:
                break

            if turn_rank == 0:
                images_for_mm = [screenshots[0]]
                multi_modal_data = await self.process_multi_modal_info(live_messages)
                # Prefer explicit screenshot list so we control binding.
                images = images_for_mm
                videos = multi_modal_data.get("videos")
                audios = multi_modal_data.get("audios")
                mm_processor_kwargs = self._get_mm_processor_kwargs(audios)
                prompt_ids = await self.apply_chat_template(
                    live_messages,
                    images=images,
                    videos=videos,
                    audios=audios,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
                full_ids = list(prompt_ids)
            else:
                images_for_mm = screenshots[: turn_rank + 1]
                new_user = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "New screenshot after the last action:"},
                            {"type": "image", "image": screenshots[turn_rank]},
                        ],
                    }
                ]
                live_messages.append(new_user[0])
                add_ids = await self.apply_chat_template(
                    new_user,
                    images=[screenshots[turn_rank]],
                    videos=None,
                    audios=None,
                    mm_processor_kwargs=mm_processor_kwargs,
                    remove_system_prompt=True,
                )
                full_ids += add_ids
                images = images_for_mm

            if is_final:
                final_prompt_ids = list(full_ids)

            with simple_timer(
                "generate_sequences_final" if is_final else "generate_sequences",
                metrics,
            ):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=full_ids,
                    sampling_params=self._sampling_params(sampling_params, budget),
                    image_data=images,
                    video_data=None,
                    audio_data=None,
                    mm_processor_kwargs=mm_processor_kwargs,
                    agent_turn=turn_rank,
                    agent_uid=uid,
                    rollout_idx=rollout_idx,
                    training_global_step=global_step,
                )

            n_gen = len(output.token_ids)
            full_ids += output.token_ids
            extra_fields.update(output.extra_fields or {})
            assistant_text = await self._decode_assistant(output.token_ids)
            live_messages.append({"role": "assistant", "content": assistant_text})

            turn_stats.append(
                {
                    "turn": turn_rank,
                    "prompt_tokens": len(full_ids) - n_gen,
                    "output_tokens": n_gen,
                    "is_final": is_final,
                    "screenshot_index": turn_rank,
                    "assistant_chars": len(assistant_text),
                }
            )
            if is_final:
                final_response_ids = list(output.token_ids)
                if response_logprobs_enabled:
                    final_response_logprobs = (
                        list(output.log_probs)
                        if output.log_probs
                        else [0.0] * n_gen
                    )

        if final_prompt_ids is None or final_response_ids is None:
            raise RuntimeError("osworld_gui_agent produced no final turn output")

        out = AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=final_response_ids[: self.response_length],
            response_mask=[1] * min(len(final_response_ids), self.response_length),
            response_logprobs=(
                final_response_logprobs[: self.response_length]
                if final_response_logprobs is not None
                else None
            ),
            multi_modal_data={"images": images_for_mm},
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=len(turn_stats),
            metrics=metrics,
            extra_fields=extra_fields,
        )
        out.extra_fields.update(
            {
                "osworld_gui_agent": True,
                "osworld_turn_stats": turn_stats,
                "uid": uid,
                "request_id": request_id,
                "osworld_sample_id": extra_info.get("sample_id"),
                "osworld_instruction": extra_info.get("instruction"),
            }
        )
        return out
