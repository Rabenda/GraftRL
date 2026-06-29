# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""Sokoban multi-step VLM agent loop on verl_vision + SGLang (refocus-aligned stack)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SOKOBAN_VISUAL_TEMPLATE = """
You are an expert agent operating in the Sokoban environment. Your goal is to push all the boxes onto the target spots. Once all boxes are on the targets, you win!

# Rules
You can only push boxes. You can't pull them, so plan ahead to avoid getting stuck.
You can't walk through or push boxes into walls.
To avoid traps, do not push boxes into corners or against walls where they can't be moved again.

# Visual Elements in the Image:
Character: A small, green alien-like figure with two antennae and black eyes. It represents you.
Box: A yellow crate marked with an orange "X" across its front. It is the box you need to push.
Target: A black tile outlined in red, with a small red diamond shape in the center. It marks the destination where a box should be pushed.

# Current Step
Your current observation is shown in the image: <image>
Your admissible actions are ["up", "down", "left", "right"].

Now it's your turn to make a move (choose ONE action only for the current step).
You should first reason step-by-step about the current situation — observe the positions of boxes and targets, plan a path to push a box toward a target, and avoid traps like corners or walls. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
""".strip()

SOKOBAN_STEP_TEMPLATE = """
Your new observation after the last action is shown in the image: <image>
Your admissible actions are ["up", "down", "left", "right"].

Reason step-by-step about the updated layout in <think> </think>, then choose ONE action in <action> </action> tags.
""".strip()

_ACTION_POOL = {"up": 1, "down": 2, "left": 3, "right": 4, "still": 0}

_branch_lock = asyncio.Lock()
_branch_in_use: dict[int, set[int]] = defaultdict(set)


def _ensure_verl_agent_on_path() -> None:
    root = os.environ.get("VERL_AGENT_ROOT", "/workspace/repo/verl-agent")
    if root not in sys.path:
        sys.path.insert(0, root)


def _image_dump_dir() -> Path | None:
    raw = os.environ.get("PROFILE_IMAGE_DUMP_DIR", "").strip()
    return Path(raw) if raw else None


def _dump_obs(
    *,
    dump_dir: Path,
    group_idx: int,
    branch_idx: int,
    step: int,
    image: Image.Image,
) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    rel = f"g{group_idx:04d}_b{branch_idx}_s{step:02d}_obs.png"
    image.convert("RGB").save(dump_dir / rel)
    record = {
        "group_idx": group_idx,
        "branch_idx": branch_idx,
        "step": step,
        "role": "obs",
        "path": rel,
        "uid": f"group_{group_idx:04d}",
        "request_id": f"branch_{branch_idx}",
    }
    with (dump_dir / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def _allocate_branch(group_idx: int, rollout_n: int) -> int:
    async with _branch_lock:
        used = _branch_in_use[group_idx]
        for branch in range(rollout_n):
            if branch not in used:
                used.add(branch)
                return branch
        branch = len(used) % max(rollout_n, 1)
        used.add(branch)
        return branch


async def _release_branch(group_idx: int, branch_idx: int) -> None:
    async with _branch_lock:
        _branch_in_use[group_idx].discard(branch_idx)


def _parse_action(text: str) -> int:
    lowered = text.lower()
    start_tag, end_tag = "<action>", "</action>"
    start_idx = lowered.find(start_tag)
    end_idx = lowered.find(end_tag)
    if start_idx == -1 or end_idx == -1:
        return 0
    extracted = lowered[start_idx + len(start_tag) : end_idx].strip()
    for name, code in _ACTION_POOL.items():
        if name in extracted:
            return code
    return 0


def _rgb_to_pil(obs: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(obs)).convert("RGB")


def _build_multimodal_user_message(text: str, num_images: int) -> list[dict]:
    """Split <image> placeholders into structured content for processor tokenization."""
    content_list: list[dict] = []
    image_count = 0
    for segment in [item for item in re.split("(<image>)", text) if item != ""]:
        if segment == "<image>":
            content_list.append({"type": "image"})
            image_count += 1
        else:
            content_list.append({"type": "text", "text": segment})
    if image_count != num_images:
        raise ValueError(f"expected {num_images} <image> placeholders, found {image_count}")
    return [{"role": "user", "content": content_list}]


@register("sokoban_agent")
class SokobanAgentLoop(AgentLoopBase):
    """Multi-turn Sokoban: gym env steps + VLM actions, with optional image dump."""

    def __init__(
        self,
        *args,
        max_env_steps: int = 15,
        env_seed_base: int = 0,
        dim_room: tuple[int, int] = (6, 6),
        num_boxes: int = 1,
        search_depth: int = 30,
        response_length_per_turn: int = 128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_env_steps = max_env_steps
        self.env_seed_base = env_seed_base
        self.dim_room = tuple(dim_room)
        self.num_boxes = num_boxes
        self.search_depth = search_depth
        self.response_length_per_turn = min(
            max(int(response_length_per_turn or self.rollout_config.response_length), 1),
            self.rollout_config.response_length,
        )
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.rollout_n = max(int(self.rollout_config.n or 1), 1)

    def _make_env(self):
        _ensure_verl_agent_on_path()
        from agent_system.environments.env_package.sokoban.sokoban.env import SokobanEnv

        return SokobanEnv(
            mode="rgb_array",
            dim_room=self.dim_room,
            num_boxes=self.num_boxes,
            max_steps=self.max_env_steps,
            search_depth=self.search_depth,
        )

    def _initial_user_messages(self) -> list[dict]:
        return _build_multimodal_user_message(SOKOBAN_VISUAL_TEMPLATE, num_images=1)

    def _step_user_messages(self) -> list[dict]:
        return _build_multimodal_user_message(SOKOBAN_STEP_TEMPLATE, num_images=1)

    def _budget(self, sampling_params: dict[str, Any], used: int) -> dict[str, Any]:
        params = dict(sampling_params)
        params.pop("max_tokens", None)
        remaining = self.response_length - used
        per_turn = min(self.response_length_per_turn, remaining)
        params["max_new_tokens"] = max(1, per_turn)
        return params

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        group_idx = int(kwargs.get("index", 0))
        uid = str(kwargs.get("uid") or uuid4())
        global_step = kwargs.get("global_step")
        branch_idx = await _allocate_branch(group_idx, self.rollout_n)
        dump_dir = _image_dump_dir()
        request_id = uuid4().hex

        env = await self.loop.run_in_executor(None, self._make_env)
        env_seed = self.env_seed_base + group_idx
        obs, _info = await self.loop.run_in_executor(None, lambda: env.reset(seed=env_seed))
        images = [_rgb_to_pil(obs)]
        if dump_dir is not None:
            _dump_obs(
                dump_dir=dump_dir,
                group_idx=group_idx,
                branch_idx=branch_idx,
                step=0,
                image=images[-1],
            )

        metrics: dict[str, Any] = {}
        env_rewards: list[float] = []
        prompt_ids = await self.apply_chat_template(
            self._initial_user_messages(),
            images=[images[0]],
        )
        # Count images actually tokenized into prompt_ids (one per <image> placeholder).
        # Starts at 1 for the initial observation; +1 for every appended step turn.
        tokenized_images = 1
        response_mask: list[int] = []
        response_logprobs: list[float] | None = [] if sampling_params.get("logprobs") else None
        assistant_turns = 0
        env_steps = 0
        done = False

        try:
            while not done and env_steps < self.max_env_steps and len(response_mask) < self.response_length:
                remaining = self.response_length - len(response_mask)
                if remaining <= 0:
                    break

                with simple_timer("generate_sequences", metrics):
                    output: TokenOutput = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=self._budget(sampling_params, len(response_mask)),
                        image_data=images,
                        mm_processor_kwargs=self._get_mm_processor_kwargs(None),
                        agent_turn=assistant_turns,
                        agent_uid=uid,
                        rollout_idx=str(group_idx),
                        training_global_step=global_step,
                    )

                remaining = self.response_length - len(response_mask)
                chunk = output.token_ids[:remaining]
                prompt_ids += chunk
                response_mask += [1] * len(chunk)
                if response_logprobs is not None:
                    log_probs = output.log_probs[: len(chunk)] if output.log_probs else [0.0] * len(chunk)
                    response_logprobs += log_probs
                assistant_turns += 1

                text = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.decode(chunk, skip_special_tokens=False),
                )
                action = _parse_action(text)
                obs, reward, terminated, _info = await self.loop.run_in_executor(
                    None, lambda: env.step(action)
                )
                env_rewards.append(float(reward))
                env_steps += 1
                done = bool(terminated) or env.success()
                images.append(_rgb_to_pil(obs))
                if dump_dir is not None:
                    _dump_obs(
                        dump_dir=dump_dir,
                        group_idx=group_idx,
                        branch_idx=branch_idx,
                        step=env_steps,
                        image=images[-1],
                    )

                if done or env_steps >= self.max_env_steps:
                    break

                add_ids = await self.apply_chat_template(
                    self._step_user_messages(),
                    images=[images[-1]],
                    remove_system_prompt=True,
                )
                if len(response_mask) + len(add_ids) > self.response_length:
                    # Do not let the final AgentLoopOutput truncate a multimodal user
                    # turn. Cutting through Qwen image-pad tokens desynchronizes
                    # input_ids from image_grid_thw in get_rope_index.
                    break

                prompt_ids += add_ids
                response_mask += [0] * len(add_ids)
                tokenized_images += 1
                if response_logprobs is not None:
                    response_logprobs += [0.0] * len(add_ids)

            split_idx = len(prompt_ids) - len(response_mask)
            response_ids = prompt_ids[split_idx:]
            final_prompt_ids = prompt_ids[:split_idx]
            if len(response_ids) != len(response_mask) or len(response_mask) > self.response_length:
                raise RuntimeError(
                    "Sokoban rollout produced inconsistent response lengths: "
                    f"response_ids={len(response_ids)}, response_mask={len(response_mask)}, "
                    f"limit={self.response_length}"
                )
            # multi_modal_data must align with the <image> placeholders actually present
            # in prompt_ids. The final env.step appends an observation that is NOT tokenized
            # when the loop ends (done/max_steps), so `images` can be longer than the
            # tokenized count. Slice to the exact number of tokenized observations.
            prompt_images = images[:tokenized_images]
            output = AgentLoopOutput(
                prompt_ids=final_prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=response_logprobs if response_logprobs else None,
                multi_modal_data={"images": prompt_images},
                reward_score=sum(env_rewards),
                num_turns=assistant_turns,
                metrics=metrics,
                extra_fields={
                    "sokoban_agent": True,
                    "group_idx": group_idx,
                    "branch_idx": branch_idx,
                    "env_steps": env_steps,
                    "env_success": bool(env.success()),
                    "env_rewards": env_rewards,
                    "uid": uid,
                    "request_id": request_id,
                },
            )
            return output
        finally:
            await self.loop.run_in_executor(None, env.close)
            await _release_branch(group_idx, branch_idx)
