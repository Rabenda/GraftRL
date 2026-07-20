# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""MMSearch-R1-style VLM search loop with dependency-aware exact reuse.

This loop is intentionally local and deterministic. It mirrors the shape of
MMSearch-R1 rollouts, but avoids live search dependencies so prefill/image
reuse can be profiled repeatably:

Search candidates become content-addressed artifacts. Retrieval and
tokenization are exact-reuse stages, while optional selection and suppressing
already-materialized content are tracked separately as SKIP. The decoder still
receives a normal contiguous prompt with recipient-local positions.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
from uuid import uuid4

from PIL import Image, ImageDraw, ImageEnhance

from examples.profile.workloads.mmsearch_r1.context_cache import (
    ContentArtifact,
    ContentSelectionCache,
    image_content_hash,
    text_content_hash,
)
from verl.experimental.rollout_reuse import (
    ArtifactIdentity,
    ExecutionAction,
    ReuseContext,
    ReuseRegistry,
    RolloutReuseRuntime,
    SharingScope,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_IMAGE_SEARCH_RE = re.compile(r"<search>\s*<img>\s*</search>", re.IGNORECASE)
_TEXT_SEARCH_RE = re.compile(r"<text_search>(.*?)</text_search>", re.IGNORECASE | re.DOTALL)


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return max(minimum, int(default))


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, default)
    return str(value).strip() or default


def _decode_safe(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


_SELECTION_CACHE = ContentSelectionCache(max_entries=4096)
_ARTIFACT_REGISTRY: ReuseRegistry[Any] = ReuseRegistry(max_entries=8192)
_REUSE_RUNTIME: RolloutReuseRuntime[Any] = RolloutReuseRuntime(_ARTIFACT_REGISTRY)


def _selection_decision_fields(
    *,
    modality: str,
    candidate_count: int,
    deduplicated_count: int,
    ranked_skip_count: int,
    materialized_skip_count: int,
) -> dict[str, Any]:
    """Emit separate exact and approximate SKIP decisions for context selection."""

    exact_skip_count = max(0, deduplicated_count + materialized_skip_count)
    exact = _REUSE_RUNTIME.decide(
        action=(ExecutionAction.SKIP if exact_skip_count else ExecutionAction.LOCAL),
        operator_id="mmsearch.context_materialization",
        representation_stage=f"{modality}_observation",
        reason=("duplicate_or_already_materialized" if exact_skip_count else "no_exact_skip"),
        eligible_units=max(0, candidate_count),
        applied_units=exact_skip_count,
        approximate=False,
        policy="content-identity-v1",
    )
    ranked = _REUSE_RUNTIME.decide(
        action=(ExecutionAction.SKIP if ranked_skip_count else ExecutionAction.LOCAL),
        operator_id="mmsearch.context_rank_prune",
        representation_stage=f"{modality}_observation",
        reason=("outside_context_budget" if ranked_skip_count else "within_context_budget"),
        eligible_units=max(0, candidate_count - deduplicated_count),
        applied_units=max(0, ranked_skip_count),
        approximate=bool(ranked_skip_count),
        policy="retrieval-rank-v1",
    )
    return {
        **exact.event_fields("context_exact_skip"),
        **ranked.event_fields("context_rank_skip"),
    }


def _content_hash(text: str) -> str:
    return text_content_hash(text)


def _policy_epoch(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _tokenizer_signature(tokenizer: Any) -> dict[str, str]:
    chat_template = str(getattr(tokenizer, "chat_template", "") or "")
    return {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "name_or_path": str(getattr(tokenizer, "name_or_path", "") or ""),
        "chat_template_sha256": text_content_hash(chat_template),
    }


def _candidate_answers(reward_model: Any) -> list[str]:
    if not isinstance(reward_model, dict):
        return []
    value = reward_model.get("candidate_answers", [])
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(answer) for answer in value if answer is not None]


def _make_search_thumbnail(image: Image.Image, idx: int) -> Image.Image:
    base = image.convert("RGB")
    max_side = _env_int("MMSEARCH_R1_THUMBNAIL_MAX_SIDE", 448, minimum=64)
    if max(base.size) > max_side:
        base.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
    width, height = base.size
    crop_ratio = 0.86 - min(idx, 4) * 0.06
    crop_w = max(1, int(width * crop_ratio))
    crop_h = max(1, int(height * crop_ratio))
    left = max(0, (width - crop_w) // 2)
    top = max(0, (height - crop_h) // 2)
    thumb = base.crop((left, top, left + crop_w, top + crop_h)).resize(base.size)

    if idx % 3 == 1:
        thumb = ImageEnhance.Contrast(thumb).enhance(1.18)
    elif idx % 3 == 2:
        thumb = ImageEnhance.Brightness(thumb).enhance(0.88)

    draw = ImageDraw.Draw(thumb, "RGBA")
    palette = [(33, 116, 255, 210), (230, 96, 32, 210), (30, 150, 100, 210), (168, 72, 186, 210)]
    color = palette[idx % len(palette)]
    line = max(3, min(width, height) // 80)
    inset = max(3, line * 2)
    draw.rectangle((inset, inset, width - inset, height - inset), outline=color, width=line)
    draw.rectangle((inset, inset, min(width - inset, inset + 130), inset + 40), fill=color)
    draw.text((inset + 10, inset + 10), f"R{idx + 1}", fill=(255, 255, 255, 255))
    return thumb.convert("RGB")


def _synthetic_image_search(images: list[Image.Image], topk: int) -> tuple[list[Image.Image], list[str]]:
    if not images:
        raise ValueError("MMSearch-R1 image search profiling requires at least one input image")
    topk = max(1, int(topk))
    returned_images = [_make_search_thumbnail(images[0], idx) for idx in range(topk)]
    titles = [f"[Image Search Results] Result {idx + 1}: visually related web image." for idx in range(topk)]
    return returned_images, titles


def _synthetic_text_search(query: str, topk: int) -> list[str]:
    clean_query = " ".join((query or "visual question").split())
    topk = max(1, int(topk))
    return [
        (
            f"[Text Search Results] Result {idx + 1}: summary for query '{clean_query}'. "
            "This deterministic passage stands in for a retrieved web document."
        )
        for idx in range(topk)
    ]


@register("mmsearch_r1_agent")
class MMSearchR1AgentLoop(AgentLoopBase):
    """MMSearch-R1-shaped search rollout using SGLang multi-turn generation."""

    def __init__(
        self,
        *args,
        max_gen_round: int = 3,
        first_turn_max_new_tokens: int = 128,
        followup_max_new_tokens: int = 512,
        image_search_topk: int = 5,
        text_search_topk: int = 5,
        context_topk: int | None = None,
        force_tool: str = "image",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length
        self.max_gen_round = _env_int("MMSEARCH_R1_MAX_GEN_ROUND", max_gen_round, minimum=1)
        self.first_turn_max_new_tokens = _env_int(
            "MMSEARCH_R1_FIRST_TURN_MAX_NEW_TOKENS", first_turn_max_new_tokens, minimum=1
        )
        self.followup_max_new_tokens = _env_int(
            "MMSEARCH_R1_FOLLOWUP_MAX_NEW_TOKENS", followup_max_new_tokens, minimum=1
        )
        self.image_search_topk = _env_int("MMSEARCH_R1_IMAGE_SEARCH_TOPK", image_search_topk, minimum=1)
        self.text_search_topk = _env_int("MMSEARCH_R1_TEXT_SEARCH_TOPK", text_search_topk, minimum=1)
        default_context_topk = context_topk or max(self.image_search_topk, self.text_search_topk)
        self.context_topk = _env_int("MMSEARCH_R1_CONTEXT_TOPK", default_context_topk, minimum=1)
        self.force_tool = _env_str("MMSEARCH_R1_PROFILE_FORCE_TOOL", force_tool).lower()

    def _sampling_params(self, sampling_params: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
        params = dict(sampling_params)
        params.pop("max_tokens", None)
        params["max_new_tokens"] = max(1, int(max_new_tokens))
        return params

    def _select_tool(self, assistant_text: str, round_idx: int) -> tuple[str | None, str]:
        """Return (tool_kind, query). tool_kind is image/text/None."""
        if round_idx == 0 and self.force_tool in ("image", "img"):
            return "image", ""
        if round_idx == 0 and self.force_tool in ("text", "doc"):
            match = _TEXT_SEARCH_RE.search(assistant_text)
            return "text", match.group(1).strip() if match else "visual question"
        if self.force_tool in ("none", "off", "disable", "disabled"):
            return None, ""

        text_match = _TEXT_SEARCH_RE.search(assistant_text)
        if text_match:
            return "text", text_match.group(1).strip()
        if _IMAGE_SEARCH_RE.search(assistant_text):
            return "image", ""
        return None, ""

    async def _decode_assistant(self, token_ids: list[int]) -> str:
        return await self.loop.run_in_executor(None, lambda: _decode_safe(self.tokenizer, token_ids))

    async def _append_image_observation(
        self,
        *,
        full_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float] | None,
        images: list[Image.Image],
        seen_context_hashes: set[str],
        reuse_context: ReuseContext,
    ) -> tuple[list[Image.Image], list[int], dict[str, Any]]:
        query_hash = image_content_hash(images[0])
        thumbnail_max_side = _env_int("MMSEARCH_R1_THUMBNAIL_MAX_SIDE", 448, minimum=64)
        retrieval_identity = ArtifactIdentity.from_dependencies(
            operator_id="mmsearch.synthetic_image_search",
            representation_stage="tool_result",
            content_id=f"image-query:{query_hash}",
            dependencies={
                "query_image_content_id": query_hash,
                "topk": self.image_search_topk,
                "thumbnail_max_side": thumbnail_max_side,
                "operator_version": "synthetic-image-search-v1",
            },
        )

        def compute_results():
            returned_images, titles = _synthetic_image_search(images, self.image_search_topk)
            return tuple(returned_images), tuple(titles)

        retrieval = await _REUSE_RUNTIME.get_or_compute(
            identity=retrieval_identity,
            scope=SharingScope.GROUP,
            context=reuse_context,
            compute=compute_results,
        )
        returned_images, titles = map(list, retrieval.value)
        artifacts = [
            ContentArtifact(
                modality="image",
                content_hash=image_content_hash(image),
                rank=index,
                payload=image,
                label=titles[index],
            )
            for index, image in enumerate(returned_images)
        ]
        selection = _SELECTION_CACHE.select(
            artifacts,
            query_hash=query_hash,
            topk=self.context_topk,
        )
        selected_artifacts = [
            artifact
            for artifact in selection.artifacts
            if artifact.content_hash not in seen_context_hashes
        ]
        deduplicated_candidate_count = selection.candidate_count - selection.unique_count
        skipped_by_selection_count = selection.unique_count - selection.selected_count
        skipped_already_materialized_count = selection.selected_count - len(selected_artifacts)
        seen_context_hashes.update(artifact.content_hash for artifact in selected_artifacts)
        selected_images = [artifact.payload for artifact in selected_artifacts]
        content: list[dict[str, Any]] = [{"type": "text", "text": "Searched results: <information>\n"}]
        for artifact in selected_artifacts:
            content.append({"type": "image"})
            content.append({"type": "text", "text": artifact.label + "\n"})
        if not selected_artifacts:
            content.append({"type": "text", "text": "No new results; reuse the previously supplied context.\n"})
        content.append(
            {
                "type": "text",
                "text": "</information>\nUse the searched images and answer the original user's question.",
            }
        )
        token_content_id = text_content_hash(
            "\0".join(
                f"{artifact.content_hash}:{artifact.label}"
                for artifact in selected_artifacts
            )
        )
        token_identity = ArtifactIdentity.from_dependencies(
            operator_id="mmsearch.apply_chat_template",
            representation_stage="observation_token_ids",
            content_id=f"image-observation:{token_content_id}",
            dependencies={
                "artifacts": [
                    {"content_id": artifact.content_hash, "label": artifact.label}
                    for artifact in selected_artifacts
                ],
                "template_version": "image-observation-v1",
                "tokenizer": _tokenizer_signature(self.tokenizer),
            },
        )

        async def compute_token_ids():
            add_ids = await self.apply_chat_template(
                [{"role": "user", "content": content}],
                images=selected_images,
                videos=None,
                audios=None,
                remove_system_prompt=True,
            )
            return tuple(add_ids)

        tokenization = await _REUSE_RUNTIME.get_or_compute(
            identity=token_identity,
            scope=SharingScope.GROUP,
            context=reuse_context,
            compute=compute_token_ids,
        )
        add_ids = list(tokenization.value)
        full_ids += add_ids
        response_mask += [0] * len(add_ids)
        if response_logprobs is not None:
            response_logprobs += [0.0] * len(add_ids)
        event = {
            "modality": "image",
            "query_hash": query_hash,
            "candidate_count": selection.candidate_count,
            "unique_count": selection.unique_count,
            "ranked_selected_count": selection.selected_count,
            "selected_count": len(selected_artifacts),
            "deduplicated_candidate_count": deduplicated_candidate_count,
            "skipped_by_selection_count": skipped_by_selection_count,
            "skipped_already_materialized_count": skipped_already_materialized_count,
            "skip_count": (
                deduplicated_candidate_count
                + skipped_by_selection_count
                + skipped_already_materialized_count
            ),
            "selected_hashes": [artifact.content_hash for artifact in selected_artifacts],
            "context_reduction_ratio": 1.0 - len(selected_artifacts) / selection.candidate_count,
            "retrieval_cache_hit": retrieval.exact_reuse,
            "selection_cache_hit": selection.cache_hit,
            "tokenization_cache_hit": tokenization.exact_reuse,
            "observation_tokens": len(add_ids),
            **retrieval.event_fields("retrieval"),
            **tokenization.event_fields("tokenization"),
            **_selection_decision_fields(
                modality="image",
                candidate_count=selection.candidate_count,
                deduplicated_count=deduplicated_candidate_count,
                ranked_skip_count=skipped_by_selection_count,
                materialized_skip_count=skipped_already_materialized_count,
            ),
        }
        return images + selected_images, add_ids, event

    async def _append_text_observation(
        self,
        *,
        full_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float] | None,
        query: str,
        seen_context_hashes: set[str],
        reuse_context: ReuseContext,
    ) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
        normalized_query = " ".join((query or "visual question").split())
        query_hash = _content_hash(normalized_query)
        retrieval_identity = ArtifactIdentity.from_dependencies(
            operator_id="mmsearch.synthetic_text_search",
            representation_stage="tool_result",
            content_id=f"text-query:{query_hash}",
            dependencies={
                "normalized_query": normalized_query,
                "topk": self.text_search_topk,
                "operator_version": "synthetic-text-search-v1",
                "external_snapshot": "synthetic-static-v1",
            },
        )
        retrieval = await _REUSE_RUNTIME.get_or_compute(
            identity=retrieval_identity,
            scope=SharingScope.GROUP,
            context=reuse_context,
            compute=lambda: tuple(_synthetic_text_search(query, self.text_search_topk)),
        )
        passages = list(retrieval.value)
        artifacts = [
            ContentArtifact(
                modality="document",
                content_hash=_content_hash(passage),
                rank=index,
                payload=passage,
                label=f"Result {index + 1}",
            )
            for index, passage in enumerate(passages)
        ]
        selection = _SELECTION_CACHE.select(
            artifacts,
            query_hash=query_hash,
            topk=self.context_topk,
        )
        selected_artifacts = [
            artifact
            for artifact in selection.artifacts
            if artifact.content_hash not in seen_context_hashes
        ]
        deduplicated_candidate_count = selection.candidate_count - selection.unique_count
        skipped_by_selection_count = selection.unique_count - selection.selected_count
        skipped_already_materialized_count = selection.selected_count - len(selected_artifacts)
        seen_context_hashes.update(artifact.content_hash for artifact in selected_artifacts)
        selected_passages = [artifact.payload for artifact in selected_artifacts]
        if not selected_passages:
            selected_passages = ["No new results; reuse the previously supplied context."]
        text = (
            "Searched results: <information>\n"
            + "\n".join(selected_passages)
            + "\n</information>\nUse the searched text and answer the original user's question."
        )
        start = len(full_ids)
        token_identity = ArtifactIdentity.from_dependencies(
            operator_id="mmsearch.apply_chat_template",
            representation_stage="observation_token_ids",
            content_id=f"text-observation:{_content_hash(text)}",
            dependencies={
                "selected_content_ids": [
                    artifact.content_hash for artifact in selected_artifacts
                ],
                "rendered_text_sha256": _content_hash(text),
                "template_version": "text-observation-v1",
                "tokenizer": _tokenizer_signature(self.tokenizer),
            },
        )

        async def compute_token_ids():
            add_ids = await self.apply_chat_template(
                [{"role": "user", "content": text}],
                remove_system_prompt=True,
            )
            return tuple(add_ids)

        tokenization = await _REUSE_RUNTIME.get_or_compute(
            identity=token_identity,
            scope=SharingScope.GROUP,
            context=reuse_context,
            compute=compute_token_ids,
        )
        add_ids = list(tokenization.value)
        full_ids += add_ids
        response_mask += [0] * len(add_ids)
        if response_logprobs is not None:
            response_logprobs += [0.0] * len(add_ids)
        doc_spans = [
            {
                "kind": "text_search_observation",
                "query_hash": query_hash,
                "content_hash": _content_hash(text),
                "selected_hashes": [artifact.content_hash for artifact in selected_artifacts],
                "token_start": start,
                "token_end": start + len(add_ids),
                "n_tokens": len(add_ids),
                "candidate_count": selection.candidate_count,
                "ranked_selected_count": selection.selected_count,
                "selected_count": len(selected_artifacts),
            }
        ]
        event = {
            "modality": "document",
            "query_hash": query_hash,
            "candidate_count": selection.candidate_count,
            "unique_count": selection.unique_count,
            "ranked_selected_count": selection.selected_count,
            "selected_count": len(selected_artifacts),
            "deduplicated_candidate_count": deduplicated_candidate_count,
            "skipped_by_selection_count": skipped_by_selection_count,
            "skipped_already_materialized_count": skipped_already_materialized_count,
            "skip_count": (
                deduplicated_candidate_count
                + skipped_by_selection_count
                + skipped_already_materialized_count
            ),
            "selected_hashes": [artifact.content_hash for artifact in selected_artifacts],
            "context_reduction_ratio": 1.0 - len(selected_artifacts) / selection.candidate_count,
            "retrieval_cache_hit": retrieval.exact_reuse,
            "selection_cache_hit": selection.cache_hit,
            "tokenization_cache_hit": tokenization.exact_reuse,
            "observation_tokens": len(add_ids),
            **retrieval.event_fields("retrieval"),
            **tokenization.event_fields("tokenization"),
            **_selection_decision_fields(
                modality="document",
                candidate_count=selection.candidate_count,
                deduplicated_count=deduplicated_candidate_count,
                ranked_skip_count=skipped_by_selection_count,
                materialized_skip_count=skipped_already_materialized_count,
            ),
        }
        return doc_spans, add_ids, event

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        uid = str(kwargs.get("uid") or uuid4())
        request_id = uuid4().hex
        rollout_idx_value = kwargs.get("rollout_idx", kwargs.get("index"))
        rollout_idx = str(rollout_idx_value) if rollout_idx_value is not None else "0"
        global_step = kwargs.get("global_step")
        policy_epoch = _policy_epoch(global_step)

        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images") or []
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        full_ids = list(prompt_ids)
        response_mask: list[int] = []
        response_logprobs: list[float] | None = [] if sampling_params.get("logprobs") else None
        extra_fields: dict[str, Any] = {}
        metrics: dict[str, Any] = {"mmsearch_r1_agent": 1.0}
        doc_reuse_spans: list[dict[str, Any]] = []
        tool_events: list[dict[str, Any]] = []
        context_events: list[dict[str, Any]] = []
        assistant_turns: list[str] = []
        seen_context_hashes: set[str] = set()

        for round_idx in range(self.max_gen_round):
            remaining = self.response_length - len(response_mask)
            if remaining <= 0:
                break
            budget = self.first_turn_max_new_tokens if round_idx == 0 else self.followup_max_new_tokens
            budget = min(budget, remaining)

            with simple_timer("generate_sequences" if round_idx == 0 else "generate_sequences_followup", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=full_ids,
                    sampling_params=self._sampling_params(sampling_params, budget),
                    image_data=images,
                    video_data=videos,
                    audio_data=audios,
                    mm_processor_kwargs=mm_processor_kwargs,
                    agent_turn=round_idx,
                    agent_uid=uid,
                    rollout_idx=rollout_idx,
                    training_global_step=global_step,
                )

            full_ids += output.token_ids
            response_mask += [1] * len(output.token_ids)
            if response_logprobs is not None:
                response_logprobs += output.log_probs if output.log_probs else [0.0] * len(output.token_ids)
            extra_fields.update(output.extra_fields or {})
            metrics["num_preempted"] = metrics.get("num_preempted", 0) + (
                output.num_preempted if output.num_preempted is not None else 0
            )
            assistant_text = await self._decode_assistant(output.token_ids)
            assistant_turns.append(assistant_text)

            if round_idx >= self.max_gen_round - 1:
                break

            tool_kind, query = self._select_tool(assistant_text, round_idx)
            if tool_kind is None:
                break

            if tool_kind == "image":
                reuse_context = ReuseContext(
                    group_id=uid,
                    policy_epoch=policy_epoch,
                    branch_id=rollout_idx,
                    turn_id=round_idx,
                )
                before_images = len(images)
                images, add_ids, context_event = await self._append_image_observation(
                    full_ids=full_ids,
                    response_mask=response_mask,
                    response_logprobs=response_logprobs,
                    images=images,
                    seen_context_hashes=seen_context_hashes,
                    reuse_context=reuse_context,
                )
                context_events.append(context_event)
                tool_events.append(
                    {
                        "round": round_idx,
                        "tool": "image_search",
                        "n_images_before": before_images,
                        "n_images_after": len(images),
                        "observation_tokens": len(add_ids),
                        "candidate_count": context_event["candidate_count"],
                        "selected_count": context_event["selected_count"],
                    }
                )
            elif tool_kind == "text":
                reuse_context = ReuseContext(
                    group_id=uid,
                    policy_epoch=policy_epoch,
                    branch_id=rollout_idx,
                    turn_id=round_idx,
                )
                spans, add_ids, context_event = await self._append_text_observation(
                    full_ids=full_ids,
                    response_mask=response_mask,
                    response_logprobs=response_logprobs,
                    query=query,
                    seen_context_hashes=seen_context_hashes,
                    reuse_context=reuse_context,
                )
                doc_reuse_spans.extend(spans)
                context_events.append(context_event)
                tool_events.append(
                    {
                        "round": round_idx,
                        "tool": "text_search",
                        "observation_tokens": len(add_ids),
                        "query_hash": _content_hash(query or ""),
                        "candidate_count": context_event["candidate_count"],
                        "selected_count": context_event["selected_count"],
                    }
                )
            else:
                break

        response_ids = full_ids[-len(response_mask) :] if response_mask else []
        out_prompt_ids = full_ids[: len(full_ids) - len(response_mask)] if response_mask else full_ids
        out = AgentLoopOutput(
            prompt_ids=out_prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data={"images": images, **({"videos": videos} if videos is not None else {})},
            mm_processor_kwargs=mm_processor_kwargs,
            reward_score=None,
            num_turns=max(1, len(tool_events) + 1),
            metrics=metrics,
            extra_fields=extra_fields,
        )
        candidate_count = sum(event["candidate_count"] for event in context_events)
        selected_count = sum(event["selected_count"] for event in context_events)
        exact_reuse_count = sum(
            int(event[stage + "_action"] == ExecutionAction.EXACT.value)
            for event in context_events
            for stage in ("retrieval", "tokenization")
        )
        local_compute_count = sum(
            int(event[stage + "_action"] == ExecutionAction.LOCAL.value)
            for event in context_events
            for stage in ("retrieval", "tokenization")
        )
        skip_count = sum(
            int(event["skip_count"])
            for event in context_events
        )
        scorer_extra_info = dict(kwargs.get("extra_info") or {})
        scorer_extra_info.update(
            {
                "mmsearch_assistant_turns": assistant_turns,
                "mmsearch_candidate_answers": _candidate_answers(kwargs.get("reward_model")),
            }
        )
        out.extra_fields.update(
            {
                "mmsearch_r1_agent": True,
                "mmsearch_r1_force_tool": self.force_tool,
                "mmsearch_r1_tool_events": tool_events,
                "mmsearch_context_events": context_events,
                "mmsearch_context_candidate_count": candidate_count,
                "mmsearch_context_selected_count": selected_count,
                "mmsearch_context_reduction_ratio": (
                    1.0 - selected_count / candidate_count if candidate_count else 0.0
                ),
                "mmsearch_exact_reuse_count": exact_reuse_count,
                "mmsearch_local_compute_count": local_compute_count,
                "mmsearch_skip_count": skip_count,
                "mmsearch_retrieval_cache_hits": sum(
                    bool(event["retrieval_cache_hit"]) for event in context_events
                ),
                "mmsearch_selection_cache_hits": sum(
                    bool(event["selection_cache_hit"]) for event in context_events
                ),
                "mmsearch_tokenization_cache_hits": sum(
                    bool(event["tokenization_cache_hit"]) for event in context_events
                ),
                "mmsearch_assistant_turns": assistant_turns,
                "mmsearch_candidate_answers": _candidate_answers(kwargs.get("reward_model")),
                # Reward inputs derived during generation are branch-local. Use
                # the explicit replacement channel consumed by ray_trainer rather
                # than colliding with the original dataset ``extra_info`` field.
                "rollout_extra_info": scorer_extra_info,
                "mmsearch_doc_reuse_spans": doc_reuse_spans,
                "uid": uid,
                "request_id": request_id,
                "rollout_idx": rollout_idx,
                "policy_epoch": policy_epoch,
                "agent_worker_pid": os.getpid(),
            }
        )
        return out
