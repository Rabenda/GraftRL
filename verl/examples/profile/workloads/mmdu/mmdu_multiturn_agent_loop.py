# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""MMDU natural multi-turn agent loop for rollout profiling.

Each dataset row keeps its multi-turn *user* questions and images unchanged.
At runtime the loop:

1. Starts at a selected user turn (prefer new-image rounds + final question).
2. Lets the **model generate** the assistant reply.
3. Appends that generated assistant text into the live conversation.
4. Feeds the next dataset user message (skipping dataset assistant history).
5. Repeats until the final selected user turn.

Returned ``prompt_ids`` / ``response_ids`` are the **final turn only** (so reward
matches the last answer / dataset GT). Intermediate turns still call
``server_manager.generate`` with ``agent_turn`` for CacheBlend profiling.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def paired_sampling_seed(
    base_seed: int,
    *,
    global_step: Any,
    sample_index: Any,
    rollout_index: Any,
    turn_index: int,
) -> int:
    """Return a stable per-trajectory seed independent of worker scheduling."""

    coordinates = ":".join(
        str(value)
        for value in (
            int(base_seed),
            global_step,
            sample_index,
            rollout_index,
            int(turn_index),
        )
    )
    digest = hashlib.blake2s(
        coordinates.encode("utf-8"), digest_size=4, person=b"mmdu-ab"
    ).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def message_has_image(message: dict) -> bool:
    """True if a user/assistant message carries an image (string or structured)."""
    content = message.get("content")
    if content is None:
        return False
    if isinstance(content, str):
        return "<image>" in content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                return True
            if isinstance(item, str) and "<image>" in item:
                return True
    return False


def select_user_turn_indices(messages: list[dict], max_runtime_turns: int) -> list[int]:
    """Pick user-message indices for progressive rollout turns.

    Prefer user turns that introduce new images, always keep the final user
    question, and fill remaining slots with other user turns.
    When ``max_runtime_turns==1``, return only the final user turn.
    """
    max_runtime_turns = max(1, int(max_runtime_turns))
    user_idxs = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not user_idxs:
        return [0]

    last_user = user_idxs[-1]
    if max_runtime_turns == 1:
        return [last_user]

    image_users = [i for i in user_idxs if message_has_image(messages[i])]
    picks = sorted(set(image_users + [last_user]))

    if len(picks) <= max_runtime_turns:
        if len(picks) < max_runtime_turns:
            extras = [u for u in user_idxs if u not in picks]
            need = max_runtime_turns - len(picks)
            if extras:
                stride = max(1, len(extras) // need)
                picks = sorted(set(picks + extras[::stride][:need]))
        return picks[:max_runtime_turns]

    keep = {image_users[0] if image_users else user_idxs[0], last_user}
    middle = [u for u in picks if u not in keep]
    slots = max_runtime_turns - len(keep)
    if slots > 0 and middle:
        stride = max(1, len(middle) // slots)
        keep.update(middle[::stride][:slots])
    return sorted(keep)


@register("mmdu_multiturn_agent")
class MMDUMultiturnAgentLoop(AgentLoopBase):
    """Snowball multi-turn rollout: model assistant replies feed later turns."""

    def __init__(
        self,
        *args,
        max_runtime_turns: int = 4,
        intermediate_max_new_tokens: int = 128,
        final_max_new_tokens: int = 512,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length
        self.max_runtime_turns = int(
            os.environ.get("MMDU_RUNTIME_TURNS", max_runtime_turns)
        )
        self.intermediate_max_new_tokens = int(
            os.environ.get(
                "MMDU_INTERMEDIATE_MAX_NEW_TOKENS", intermediate_max_new_tokens
            )
        )
        self.final_max_new_tokens = int(
            os.environ.get("MMDU_FINAL_MAX_NEW_TOKENS", final_max_new_tokens)
        )
        paired_seed = os.environ.get("MMDU_PAIRED_SAMPLING_SEED")
        self.paired_sampling_seed = (
            None if paired_seed is None or paired_seed == "" else int(paired_seed)
        )

    def _sampling_params(
        self,
        sampling_params: dict[str, Any],
        max_new_tokens: int,
        *,
        seed: int | None = None,
    ) -> dict[str, Any]:
        params = dict(sampling_params)
        params.pop("max_tokens", None)
        params["max_new_tokens"] = max(1, int(max_new_tokens))
        if seed is not None:
            # SGLang's internal SamplingParams names the per-request deterministic
            # stream ``sampling_seed``. ``seed`` is only the OpenAI-compatible API
            # spelling and is not accepted by the direct TokenizerManager path used
            # by verl's colocated rollout server.
            params["sampling_seed"] = int(seed)
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
        sample_idx = kwargs.get("index")
        branch_idx = kwargs.get("rollout_idx")
        rollout_idx = str(branch_idx) if branch_idx is not None else None
        global_step = kwargs.get("global_step")
        extra_info = kwargs.get("extra_info") or {}

        turn_user_indices = select_user_turn_indices(messages, self.max_runtime_turns)
        metrics: dict[str, Any] = {
            "mmdu_multiturn": 1.0,
            "mmdu_runtime_turns": float(len(turn_user_indices)),
            "mmdu_snowball": 1.0,
        }

        # Live conversation: dataset user turns + model-generated assistants.
        live_messages: list[dict] = list(messages[: turn_user_indices[0] + 1])
        # Growing token sequence used for SGLang generate (all turns).
        full_ids: list[int] = []
        extra_fields: dict[str, Any] = {}
        turn_stats: list[dict[str, Any]] = []
        images = None
        videos = None
        audios = None
        mm_processor_kwargs: dict[str, Any] = {}

        # Final-turn packing for reward / trainer: only last assistant tokens.
        final_prompt_ids: list[int] | None = None
        final_response_ids: list[int] | None = None
        final_response_logprobs: list[float] | None = None
        response_logprobs_enabled = bool(sampling_params.get("logprobs"))
        # Soft budget across intermediate turns; final turn gets its own cap.
        intermediate_spent = 0

        for turn_rank, user_idx in enumerate(turn_user_indices):
            is_final = turn_rank == len(turn_user_indices) - 1
            if is_final:
                budget = min(self.final_max_new_tokens, self.response_length)
            else:
                budget = min(
                    self.intermediate_max_new_tokens,
                    max(1, self.response_length - intermediate_spent),
                )
            if budget <= 0:
                break

            if turn_rank > 0:
                # Append next dataset user message only (skip dataset assistants).
                live_messages.append(dict(messages[user_idx]))

            multi_modal_data = await self.process_multi_modal_info(live_messages)
            images = multi_modal_data.get("images")
            videos = multi_modal_data.get("videos")
            audios = multi_modal_data.get("audios")
            mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

            if turn_rank == 0:
                prompt_ids = await self.apply_chat_template(
                    live_messages,
                    images=images,
                    videos=videos,
                    audios=audios,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
                full_ids = list(prompt_ids)
            else:
                # Tokenize only the newly appended user turn (and its own images).
                new_user = [live_messages[-1]]
                new_mm = await self.process_multi_modal_info(new_user)
                add_ids = await self.apply_chat_template(
                    new_user,
                    images=new_mm.get("images"),
                    videos=new_mm.get("videos"),
                    audios=new_mm.get("audios"),
                    mm_processor_kwargs=self._get_mm_processor_kwargs(new_mm.get("audios")),
                    remove_system_prompt=True,
                )
                full_ids += add_ids

            if is_final:
                final_prompt_ids = list(full_ids)

            with simple_timer(
                "generate_sequences_final" if is_final else "generate_sequences",
                metrics,
            ):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=full_ids,
                    sampling_params=self._sampling_params(
                        sampling_params,
                        budget,
                        seed=(
                            None
                            if self.paired_sampling_seed is None
                            else paired_sampling_seed(
                                self.paired_sampling_seed,
                                global_step=global_step,
                                sample_index=sample_idx,
                                rollout_index=branch_idx,
                                turn_index=turn_rank,
                            )
                        ),
                    ),
                    image_data=images,
                    video_data=videos,
                    audio_data=audios,
                    mm_processor_kwargs=mm_processor_kwargs,
                    agent_turn=turn_rank,
                    agent_uid=uid,
                    rollout_idx=rollout_idx,
                    training_global_step=global_step,
                )

            n_gen = len(output.token_ids)
            full_ids += output.token_ids
            extra_fields.update(output.extra_fields or {})
            if not is_final:
                intermediate_spent += n_gen

            assistant_text = await self._decode_assistant(output.token_ids)
            live_messages.append({"role": "assistant", "content": assistant_text})

            turn_stats.append(
                {
                    "turn": turn_rank,
                    "user_msg_index": int(user_idx),
                    "has_image": bool(message_has_image(messages[user_idx])),
                    "prompt_tokens": len(full_ids) - n_gen,
                    "output_tokens": n_gen,
                    "is_final": is_final,
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
            raise RuntimeError("MMDU multiturn agent produced no final turn output")

        out = AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=final_response_ids[: self.response_length],
            response_mask=[1] * min(len(final_response_ids), self.response_length),
            response_logprobs=(
                final_response_logprobs[: self.response_length]
                if final_response_logprobs is not None
                else None
            ),
            multi_modal_data={
                "images": images,
                **({"videos": videos} if videos is not None else {}),
            },
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=len(turn_stats),
            metrics=metrics,
            extra_fields=extra_fields,
        )
        out.extra_fields.update(
            {
                "mmdu_multiturn_agent": True,
                "mmdu_snowball": True,
                "mmdu_turn_user_indices": turn_user_indices,
                "mmdu_turn_stats": turn_stats,
                "uid": uid,
                "request_id": request_id,
                "mmdu_sample_id": extra_info.get("sample_id"),
            }
        )
        return out
