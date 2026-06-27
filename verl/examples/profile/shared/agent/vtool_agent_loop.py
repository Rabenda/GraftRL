# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""VTool refocus agent loop for verl_vision + SGLang profiling (ported from training-v2/recipe/vtool)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from examples.profile.shared.agent.vtool_refocus_tools import (
    ParseResult,
    RefocusCodeParser,
    inject_refocus_bbox_context,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SUCCESS_OBSERVATION = (
    "OBSERVATION: Execution success. The output is as follows:\n"
    "<the image outputs of the code is added as the second image>"
)
FAILURE_OBSERVATION = (
    "OBSERVATION: Execution failed. "
    "The code did not produce a valid edited image. "
    "Please regenerate your final answer."
)

_ACTION_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)


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
    refocus_source: str | None = None,
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
    if refocus_source:
        record["refocus_source"] = refocus_source
    with (dump_dir / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


_ORACLE_CALL_RE = re.compile(
    r"focus_on_(?P<family>\w+?)_with_(?P<mode>mask|draw|highlight)\s*\(\s*image_1\s*,"
    r"\s*\[.*?\]\s*,\s*(?P<bbox>\w+)\s*\)",
    re.DOTALL,
)
_DEFAULT_DIVERSIFY_MODES = ("draw",)


def _refocus_debug_enabled() -> bool:
    return os.environ.get("VTOOL_REFOCUS_DEBUG", "").strip().lower() in ("1", "true", "yes")


def _metadata_bbox_for(metadata: dict, bbox_var: str) -> dict:
    x = metadata.get("x_values_bbox") or {}
    y = metadata.get("y_values_bbox") or {}
    name = (bbox_var or "").lower()
    if "y" in name or "row" in name:
        return y or x
    return x or y


def build_diversified_oracle_code(
    original_code: str | None,
    metadata: dict,
    branch_seed: str,
    modes: tuple[str, ...] = _DEFAULT_DIVERSIFY_MODES,
) -> tuple[str | None, str | None]:
    if not original_code:
        return None, None
    m = _ORACLE_CALL_RE.search(original_code)
    if not m:
        return None, None
    family = m.group("family")
    bbox_var = m.group("bbox")
    bbox = _metadata_bbox_for(metadata, bbox_var)
    keys = list(bbox.keys())
    if not keys:
        return None, None
    modes = tuple(modes) or _DEFAULT_DIVERSIFY_MODES
    variants = [(k, mode) for k in keys for mode in modes]
    if not variants:
        return None, None
    h = int(hashlib.sha256(str(branch_seed).encode()).hexdigest(), 16)
    key, mode = variants[h % len(variants)]
    func = f"focus_on_{family}_with_{mode}"
    code = f"_diversified = {func}(image_1, [{key!r}], {bbox_var})\ndisplay(_diversified)\n"
    return code, f"{func}:{key}"


def extract_oracle_refocus_code(thoughts: list[str] | None) -> str | None:
    if not thoughts:
        return None
    for text in thoughts:
        if not isinstance(text, str):
            continue
        match = _ACTION_RE.search(text)
        if match and "focus_on_" in match.group(1):
            return match.group(1).strip()
    return None


@register("vtool_agent")
class VToolAgentLoop(AgentLoopBase):
    """Two-phase refocus: assistant emits Python refocus code -> exec -> second generate."""

    def __init__(
        self,
        *args,
        tool_timeout_seconds: float = 10.0,
        force_oracle_refocus: bool = False,
        oracle_diversify_by_branch: bool = False,
        oracle_diversify_modes: str = "draw",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.tool_timeout_seconds = tool_timeout_seconds
        self.code_parser = RefocusCodeParser()
        self._tool_output: Image.Image | None = None
        env_oracle = os.environ.get("VTOOL_ORACLE_REFOCUS", "").strip().lower() in ("1", "true", "yes")
        self._use_oracle = bool(force_oracle_refocus or env_oracle)
        env_diversify = os.environ.get("VTOOL_ORACLE_DIVERSIFY", "").strip().lower() in ("1", "true", "yes")
        self._diversify_oracle = bool(oracle_diversify_by_branch or env_diversify)
        env_modes = os.environ.get("VTOOL_ORACLE_DIVERSIFY_MODES", "").strip()
        modes_src = env_modes or oracle_diversify_modes or "draw"
        self._diversify_modes = tuple(m.strip() for m in modes_src.split(",") if m.strip()) or ("draw",)
        self._oracle_first_turn_max_new_tokens = int(
            os.environ.get("VTOOL_ORACLE_FIRST_TURN_MAX_NEW_TOKENS", "0")
        )
        self._oracle_final_turn_max_new_tokens = int(
            os.environ.get("VTOOL_ORACLE_FINAL_TURN_MAX_NEW_TOKENS", "0")
        )

    def _sampling_params(self, sampling_params: dict[str, Any], max_new_tokens: int | None = None) -> dict[str, Any]:
        params = dict(sampling_params)
        if max_new_tokens is not None:
            params.pop("max_tokens", None)
            params["max_new_tokens"] = max(1, int(max_new_tokens))
        return params

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        uid = str(kwargs.get("uid") or uuid4())
        extra_info = kwargs.get("extra_info") or {}
        tools_kwargs = kwargs.get("tools_kwargs") or {}
        if not tools_kwargs.get("metadata"):
            ei_tools = (kwargs.get("extra_info") or {}).get("tools_kwargs")
            if ei_tools:
                tools_kwargs = ei_tools
        dump_dir = _image_dump_dir()
        request_id = uuid4().hex
        rollout_idx = kwargs.get("index")

        multi_modal_data = await self.process_multi_modal_info(messages)
        images = list(multi_modal_data.get("images") or [])
        videos = list(multi_modal_data.get("videos") or [])
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        if dump_dir is not None:
            for img in images:
                _dump_rollout_image(
                    dump_dir=dump_dir,
                    uid=uid,
                    request_id=request_id,
                    turn=0,
                    role="chart_input",
                    image=img,
                    rollout_idx=str(rollout_idx) if rollout_idx is not None else None,
                )

        metrics: dict[str, Any] = {"tool_use_attempted": 0.0, "tool_use_success": 0.0}
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
        waiting_final_after_tool = False
        tool_attempted = False
        tool_success = False
        diversified_applied = False
        diversified_variant = None
        refocus_source = "none"
        model_refocus_code: str | None = None
        oracle_code = extra_info.get("oracle_refocus_code") or extract_oracle_refocus_code(
            extra_info.get("thoughts")
        )

        while True:
            with simple_timer("generate_sequences", metrics):
                turn_max_new_tokens = None
                if (
                    assistant_turns == 0
                    and self._use_oracle
                    and oracle_code
                    and self._oracle_first_turn_max_new_tokens > 0
                ):
                    turn_max_new_tokens = self._oracle_first_turn_max_new_tokens
                elif waiting_final_after_tool and self._oracle_final_turn_max_new_tokens > 0:
                    turn_max_new_tokens = self._oracle_final_turn_max_new_tokens
                turn_sampling_params = self._sampling_params(sampling_params, turn_max_new_tokens)
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=turn_sampling_params,
                    image_data=images or None,
                    video_data=videos or None,
                    audio_data=audios,
                    mm_processor_kwargs=mm_processor_kwargs,
                    agent_turn=assistant_turns,
                    agent_uid=uid,
                    rollout_idx=str(rollout_idx) if rollout_idx is not None else None,
                )

            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
            else:
                metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

            current_resp_start = len(response_mask)
            response_ids_chunk = output.token_ids
            prompt_ids += response_ids_chunk
            response_mask += [1] * len(response_ids_chunk)
            response_logprobs = self._extend_logprobs(
                response_logprobs, current_resp_start, len(response_ids_chunk), output.log_probs
            )
            assistant_turns += 1

            if not waiting_final_after_tool and assistant_turns == 1:
                if self._use_oracle and oracle_code:
                    code_to_use = oracle_code
                    if self._diversify_oracle:
                        div_code, div_label = build_diversified_oracle_code(
                            oracle_code,
                            self._parse_metadata(tools_kwargs.get("metadata")),
                            request_id,
                            modes=self._diversify_modes,
                        )
                        if div_code:
                            code_to_use = div_code
                            diversified_applied = True
                            diversified_variant = div_label
                    parse_result = ParseResult(
                        status=True, code=code_to_use, message="oracle", error_code=""
                    )
                    refocus_source = "oracle_diversified" if diversified_applied else "oracle"
                else:
                    raw_response = await self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer.decode(response_ids_chunk, skip_special_tokens=False),
                    )
                    parse_result = self.code_parser.parse(raw_response)
                    if parse_result.status and parse_result.code:
                        model_refocus_code = parse_result.code
                        refocus_source = "model"

                if parse_result.error_code == "NOTOOL":
                    break

                tool_attempted = True
                metrics["tool_use_attempted"] = 1.0
                for index in range(current_resp_start, current_resp_start + len(response_ids_chunk)):
                    response_mask[index] = 0
                if response_logprobs:
                    for index in range(current_resp_start, current_resp_start + len(response_ids_chunk)):
                        response_logprobs[index] = 0.0

                with simple_timer("tool_calls", metrics):
                    edited_image, tool_success = await self._run_tool_round(
                        parse_result=parse_result,
                        images=images,
                        tools_kwargs=tools_kwargs,
                    )
                metrics["tool_use_success"] = float(tool_success)

                observation_ids = await self._build_observation_ids(edited_image if tool_success else None)
                if len(response_mask) + len(observation_ids) >= self.response_length:
                    break

                prompt_ids += observation_ids
                response_mask += [0] * len(observation_ids)
                if response_logprobs:
                    response_logprobs += [0.0] * len(observation_ids)

                if edited_image is not None:
                    images.append(edited_image)
                    if dump_dir is not None:
                        _dump_rollout_image(
                            dump_dir=dump_dir,
                            uid=uid,
                            request_id=request_id,
                            turn=1,
                            role="refocus_output",
                            image=edited_image,
                            rollout_idx=str(rollout_idx) if rollout_idx is not None else None,
                            refocus_source=refocus_source,
                        )

                user_turns += 1
                waiting_final_after_tool = True
                continue

            if waiting_final_after_tool:
                break
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
            "turn_scores": [],
            "tool_rewards": [],
            "vtool_tool_attempted": tool_attempted,
            "vtool_tool_success": tool_success,
            "uid": uid,
            "request_id": request_id,
            "rollout_idx": str(rollout_idx) if rollout_idx is not None else "",
            "oracle_refocus": bool(self._use_oracle and oracle_code),
            "vtool_refocus_source": refocus_source,
            "vtool_model_refocus_code": model_refocus_code or "",
            "vtool_oracle_diversify": self._diversify_oracle,
            "vtool_diversified_applied": diversified_applied,
            "vtool_diversified_variant": diversified_variant,
        }
        final_response_ids = [t for t, keep in zip(response_ids, response_mask, strict=False) if keep]
        if final_response_ids:
            extra_fields["vtool_final_response_text"] = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(final_response_ids, skip_special_tokens=True),
            )
        else:
            extra_fields["vtool_final_response_text"] = ""

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

    def _display(self, image: Image.Image) -> None:
        self._tool_output = image

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
        parse_result: ParseResult,
        images: list[Image.Image],
        tools_kwargs: dict[str, Any],
    ) -> tuple[Image.Image | None, bool]:
        edited_image = None
        success = False
        if not (parse_result.status and images):
            return None, False

        metadata = self._parse_metadata(tools_kwargs.get("metadata"))
        context = self.code_parser.get_tool_context(self._display)
        inject_refocus_bbox_context(context, metadata)
        context["display"] = self._display
        context["image_1"] = images[0]
        initial_context_keys = set(context)
        self._tool_output = None
        executable_code = self.code_parser.ensure_display_call(parse_result.code)

        def _exec_tool():
            exec(executable_code, context)

        try:
            await asyncio.wait_for(
                self.loop.run_in_executor(None, _exec_tool),
                timeout=self.tool_timeout_seconds,
            )
            if isinstance(self._tool_output, Image.Image) and self._tool_output.size[0] > 0:
                edited_image = self._tool_output
                success = True
            else:
                for key, value in reversed(list(context.items())):
                    if key in initial_context_keys:
                        continue
                    if isinstance(value, Image.Image) and value.size[0] > 0:
                        edited_image = value
                        success = True
                        break
        except Exception as exc:
            logger.warning("VTool refocus exec failed: %s", exc)

        return edited_image, success

    async def _build_observation_ids(self, edited_image: Image.Image | None) -> list[int]:
        if edited_image is not None and self.processor is not None:
            return await self.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": SUCCESS_OBSERVATION},
                        ],
                    }
                ],
                images=[edited_image],
                remove_system_prompt=True,
            )
        observation = FAILURE_OBSERVATION if edited_image is None else (
            "OBSERVATION: Execution success. The tool produced an edited image. "
            "Please regenerate your final answer based on this observation."
        )
        return await self.apply_chat_template(
            [{"role": "user", "content": observation}],
            remove_system_prompt=True,
        )

    @staticmethod
    def _parse_metadata(metadata: Any) -> dict[str, Any]:
        if isinstance(metadata, dict):
            return metadata
        if isinstance(metadata, str) and metadata:
            try:
                return json.loads(metadata)
            except json.JSONDecodeError:
                return {}
        return {}
