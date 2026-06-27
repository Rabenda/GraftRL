#!/usr/bin/env python3
"""Phase 2: functional token-replacement ablation (minimal v1).

Proves whether high-similarity image tokens are *functionally* reusable:
replace donor tokens into target, then teacher-force a fixed reference answer
and compare logprob / top-1 stability vs random / low-sim controls.

Default minimal experiment (group_36cf828b):
  target A = branch0 refocus, donor B = branch1 refocus, reuse_ratio = 40%

Usage (from verl_vision/):
  # sanity-check context + image paths (no model load)
  PYTHONPATH=. python examples/profile/run_phase2_token_replacement.py --dry-run

  # run minimal v1 on GPU
  PYTHONPATH=. python examples/profile/run_phase2_token_replacement.py \\
    --device cuda:0 --dtype bfloat16

  # later: sweep reuse ratios
  PYTHONPATH=. python examples/profile/run_phase2_token_replacement.py \\
    --reuse-ratios 0.1,0.2,0.3,0.4,0.5,0.6,0.8 --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Constants / defaults for group_36cf828b minimal case
# ---------------------------------------------------------------------------

SUCCESS_OBSERVATION = (
    "OBSERVATION: Execution success. The output is as follows:\n"
    "<the image outputs of the code is added as the second image>"
)

DEFAULT_GROUP_UID = "36cf828b-5311-4cc3-8b25-38d91b1e0322"
DEFAULT_DATASET_ROW = 691
DEFAULT_TARGET_REQUEST_ID = "3aea15b5c7e9491c87a3fbbd44aca875"  # branch0
DEFAULT_DONOR_REQUEST_ID = "a7cdb0cafb8a499e841a9b5a52ca74cb"  # branch1
DEFAULT_REUSE_RATIO = 0.40
DEFAULT_MAX_NEW_TOKENS = 128

_ORACLE_CALL_RE = re.compile(
    r"focus_on_(?P<family>\w+?)_with_(?P<mode>mask|draw|highlight)\s*\(\s*image_1\s*,"
    r"\s*\[.*?\]\s*,\s*(?P<bbox>\w+)\s*\)",
    re.DOTALL,
)
_DEFAULT_DIVERSIFY_MODES = ("draw",)

ReplacementMode = Literal["high_sim", "random", "low_sim"]


# ---------------------------------------------------------------------------
# Diversified oracle (ported from vtool_agent_loop; file may be absent in repo)
# ---------------------------------------------------------------------------


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
    """Deterministic per-branch oracle variant (draw-only by default)."""
    if not original_code:
        return None, None
    match = _ORACLE_CALL_RE.search(original_code)
    if not match:
        return None, None
    family = match.group("family")
    bbox_var = match.group("bbox")
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


# ---------------------------------------------------------------------------
# Data loading / message building
# ---------------------------------------------------------------------------


def _parse_metadata(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        return json.loads(raw)
    return {}


def load_parquet_row(parquet_path: str | Path, row_idx: int) -> dict:
    table = pq.read_table(str(parquet_path))
    if row_idx < 0 or row_idx >= table.num_rows:
        raise IndexError(f"row_idx {row_idx} out of range [0, {table.num_rows})")
    return {name: table.column(name)[row_idx].as_py() for name in table.column_names}


def _pil_from_parquet_image(image_item: dict | Image.Image) -> Image.Image:
    if isinstance(image_item, Image.Image):
        return image_item.convert("RGB")
    if isinstance(image_item, dict) and image_item.get("bytes"):
        return Image.open(io.BytesIO(image_item["bytes"])).convert("RGB")
    raise TypeError(f"unsupported image item: {type(image_item)!r}")


def inject_images_into_prompt(messages: list[dict], images: list[Image.Image]) -> list[dict]:
    """Mirror RLHFDataset._build_messages: replace <image> placeholders."""
    out = [dict(m) for m in messages]
    image_offset = 0
    for message in out:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        content_list: list[dict] = []
        for segment in re.split(r"(<image>|<video>|<audio>)", content):
            if segment == "":
                continue
            if segment == "<image>":
                if image_offset >= len(images):
                    raise ValueError(f"not enough images for <image> placeholder ({image_offset})")
                content_list.append({"type": "image", "image": images[image_offset]})
                image_offset += 1
            else:
                content_list.append({"type": "text", "text": segment})
        message["content"] = content_list
    if image_offset != len(images):
        raise ValueError(f"image placeholder count {image_offset} != len(images) {len(images)}")
    return out


def image_dump_paths(
    dump_dir: Path,
    group_uid: str,
    request_id: str,
) -> tuple[Path, Path]:
    prefix = group_uid.split("-")[0]
    rid = request_id[:8]
    chart = dump_dir / f"{prefix}_{rid}_t0_chart_input.png"
    refocus = dump_dir / f"{prefix}_{rid}_t1_refocus_output.png"
    return chart, refocus


@dataclass
class GroupSpec:
    """A GRPO group with >=2 turn1 refocus branches usable for replacement."""

    group_uid: str
    dataset_row: int
    refocus_request_ids: list[str]  # branches that produced a turn1 refocus image


def load_manifest(dump_dir: Path) -> list[dict]:
    path = dump_dir / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def discover_groups(dump_dir: Path, *, min_branches: int = 2) -> list[GroupSpec]:
    """Enumerate groups whose turn1 refocus PNGs are on disk.

    ``rollout_idx`` in the manifest is the parquet row index, so each group maps
    deterministically to a dataset row without any verl re-run.
    """
    manifest = load_manifest(dump_dir)
    by_uid: dict[str, dict] = {}
    for r in manifest:
        uid = r.get("uid")
        if not uid:
            continue
        entry = by_uid.setdefault(uid, {"rollout_idx": r.get("rollout_idx"), "refocus": {}})
        if int(r.get("turn", 0)) == 1 and r.get("role") == "refocus_output":
            rid = str(r.get("request_id"))
            # confirm the PNG actually exists before trusting it
            _, refocus = image_dump_paths(dump_dir, uid, rid)
            if refocus.is_file():
                entry["refocus"][rid] = True

    specs: list[GroupSpec] = []
    for uid, info in by_uid.items():
        rids = sorted(info["refocus"].keys())
        if len(rids) < min_branches or info["rollout_idx"] is None:
            continue
        specs.append(
            GroupSpec(
                group_uid=uid,
                dataset_row=int(info["rollout_idx"]),
                refocus_request_ids=rids,
            )
        )
    specs.sort(key=lambda s: s.group_uid)
    return specs


def build_pairs(
    request_ids: list[str],
    *,
    pairs_per_group: int,
    seed: int,
) -> list[tuple[str, str]]:
    """Sample directed (target, donor) pairs from a group's refocus branches."""
    all_pairs = [(a, b) for a in request_ids for b in request_ids if a != b]
    if pairs_per_group <= 0 or pairs_per_group >= len(all_pairs):
        return all_pairs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_pairs), size=pairs_per_group, replace=False)
    return [all_pairs[i] for i in sorted(idx)]


def build_turn1_messages(
    row: dict,
    *,
    chart_image: Image.Image,
    refocus_image: Image.Image,
    request_id: str,
    use_diversified_oracle: bool = True,
) -> tuple[list[dict], str]:
    """Reconstruct turn-1 prompt: user₀ + assistant₀ (oracle) + user₁ (obs + refocus)."""
    extra = row.get("extra_info") or {}
    tools_kwargs = extra.get("tools_kwargs") or {}
    metadata = _parse_metadata(tools_kwargs.get("metadata"))
    oracle = (extra.get("oracle_refocus_code") or "").strip()
    if not oracle:
        raise ValueError("parquet row missing extra_info.oracle_refocus_code")

    assistant_code = oracle
    variant_label = "original_oracle"
    if use_diversified_oracle:
        div_code, div_label = build_diversified_oracle_code(oracle, metadata, request_id)
        if div_code:
            assistant_code = div_code
            variant_label = div_label or "diversified"

    messages = inject_images_into_prompt(row["prompt"], [chart_image])
    messages.append(
        {
            "role": "assistant",
            "content": f"```python\n{assistant_code.strip()}\n```",
        }
    )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": refocus_image},
                {"type": "text", "text": SUCCESS_OBSERVATION},
            ],
        }
    )
    return messages, variant_label


def build_processor_inputs(processor, messages: list[dict], images: list[Image.Image]) -> dict:
    from verl.utils.tokenizer import build_multimodal_processor_inputs

    raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = build_multimodal_processor_inputs(processor, text=[raw_prompt], images=images)
    return inputs


# ---------------------------------------------------------------------------
# Vision token helpers
# ---------------------------------------------------------------------------


def image_pad_spans(input_ids: torch.Tensor, image_token_id: int) -> list[tuple[int, int]]:
    """Return [start, end) spans of contiguous image-pad tokens."""
    ids = input_ids.reshape(-1).tolist()
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(ids):
        if ids[i] != image_token_id:
            i += 1
            continue
        j = i
        while j < len(ids) and ids[j] == image_token_id:
            j += 1
        spans.append((i, j))
        i = j
    return spans


@torch.inference_mode()
def encode_image_embeds(model, processor, image: Image.Image, device: torch.device) -> torch.Tensor:
    """Projected ViT tokens for a single image: shape [n_tokens, hidden]."""
    from verl.utils.transformers_compat import unpack_visual_output

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "x"},
            ],
        }
    ]
    inputs = build_processor_inputs(processor, messages, [image])
    pixel_values = inputs["pixel_values"].to(device=device, dtype=model.dtype)
    image_grid_thw = inputs["image_grid_thw"].to(device)
    inner = model.model
    embeds, _ = unpack_visual_output(inner.visual(pixel_values, grid_thw=image_grid_thw))
    return embeds


def pairwise_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def select_replacement_indices(
    cosines: torch.Tensor,
    reuse_ratio: float,
    mode: ReplacementMode,
    seed: int,
) -> np.ndarray:
    n_tokens = cosines.numel()
    k = max(1, int(round(n_tokens * reuse_ratio)))
    order = torch.argsort(cosines, descending=True).cpu().numpy()
    if mode == "high_sim":
        return order[:k]
    if mode == "low_sim":
        return order[-k:]
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_tokens, size=k, replace=False))


def apply_token_replacement(
    inputs_embeds: torch.Tensor,
    second_image_span: tuple[int, int],
    target_embeds: torch.Tensor,
    donor_embeds: torch.Tensor,
    replace_token_indices: np.ndarray,
) -> torch.Tensor:
    """Replace selected positions in the 2nd image block with donor embeddings."""
    start, end = second_image_span
    n_block = end - start
    if target_embeds.shape[0] != n_block or donor_embeds.shape[0] != n_block:
        raise ValueError(
            f"token block size {n_block} != embed rows "
            f"(target={target_embeds.shape[0]}, donor={donor_embeds.shape[0]})"
        )
    out = inputs_embeds.clone()
    for idx in replace_token_indices:
        pos = start + int(idx)
        out[0, pos] = donor_embeds[int(idx)].to(dtype=out.dtype, device=out.device)
    return out


# ---------------------------------------------------------------------------
# Model forward / generation / metrics
# ---------------------------------------------------------------------------


def make_position_ids(processor, input_ids: torch.Tensor, attention_mask: torch.Tensor, mm_inputs: dict) -> torch.Tensor:
    from verl.models.transformers.qwen2_vl import get_rope_index

    pos = get_rope_index(
        processor,
        input_ids=input_ids[0],
        image_grid_thw=mm_inputs.get("image_grid_thw"),
        video_grid_thw=mm_inputs.get("video_grid_thw"),
        attention_mask=attention_mask[0],
    )
    return pos.unsqueeze(1)  # (3, 1, seq)


@torch.inference_mode()
def build_inputs_embeds(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mm_inputs: dict,
    device: torch.device,
) -> torch.Tensor:
    from verl.models.transformers.qwen2_vl import _get_input_embeds

    inner = model.model
    pixel_values = mm_inputs.get("pixel_values")
    image_grid_thw = mm_inputs.get("image_grid_thw")
    if pixel_values is not None:
        pixel_values = pixel_values.to(device=device, dtype=inner.visual.dtype)
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)
    inputs_embeds, _ = _get_input_embeds(
        inner,
        input_ids.to(device),
        attention_mask.to(device),
        pixel_values,
        image_grid_thw=image_grid_thw,
    )
    return inputs_embeds


@torch.inference_mode()
def forward_logits(
    model,
    processor,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mm_inputs: dict,
    inputs_embeds: torch.Tensor | None = None,
    device: torch.device,
) -> torch.Tensor:
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = make_position_ids(processor, input_ids, attention_mask, mm_inputs).to(device)

    if inputs_embeds is None:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=mm_inputs["pixel_values"].to(device=device, dtype=model.dtype),
            image_grid_thw=mm_inputs["image_grid_thw"].to(device),
            use_cache=False,
            return_dict=True,
        )
    else:
        outputs = model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
    return outputs.logits


@torch.inference_mode()
def greedy_generate(
    model,
    processor,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mm_inputs: dict,
    max_new_tokens: int,
    device: torch.device,
) -> list[int]:
    gen = model.generate(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        pixel_values=mm_inputs["pixel_values"].to(device=device, dtype=model.dtype),
        image_grid_thw=mm_inputs["image_grid_thw"].to(device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        use_cache=True,
    )
    prompt_len = input_ids.shape[1]
    return gen[0, prompt_len:].tolist()


def teacher_force_metrics(
    logits_orig: torch.Tensor,
    logits_repl: torch.Tensor,
    *,
    prompt_len: int,
    response_ids: list[int],
) -> dict[str, float]:
    """Compare next-token distributions on a fixed reference answer."""
    from verl.utils.torch_functional import logprobs_from_logits

    if not response_ids:
        raise ValueError("empty response_ids")

    resp = torch.tensor(response_ids, dtype=torch.long, device=logits_orig.device)
    # logits[t] predicts token t+1
    pos = torch.arange(prompt_len - 1, prompt_len - 1 + len(response_ids), device=logits_orig.device)
    step_logits_o = logits_orig[0, pos]
    step_logits_r = logits_repl[0, pos]

    logp_o = logprobs_from_logits(step_logits_o, resp, inplace_backward=False)
    logp_r = logprobs_from_logits(step_logits_r, resp, inplace_backward=False)

    nll_o = -logp_o.sum().item()
    nll_r = -logp_r.sum().item()
    delta_nll = nll_r - nll_o
    delta_nll_per_token = delta_nll / len(response_ids)

    top1_o = step_logits_o.argmax(dim=-1)
    top1_r = step_logits_r.argmax(dim=-1)
    top1_match = (top1_o == top1_r).float().mean().item()

    return {
        "nll_original": nll_o,
        "nll_replaced": nll_r,
        "delta_nll": delta_nll,
        "delta_nll_per_token": delta_nll_per_token,
        "top1_match_rate": top1_match,
        "response_len": len(response_ids),
    }


@dataclass
class ExperimentResult:
    group_uid: str
    dataset_row: int
    target_request_id: str
    donor_request_id: str
    reuse_ratio: float
    mode: str
    n_image_tokens: int
    n_replaced: int
    mean_pairwise_cos: float
    oracle_variant: str
    prompt_tokens: int
    image_pad_tokens: int
    y_ref_text: str
    metrics: dict[str, float]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_sanity_stability(
    model, processor, inputs, mm_inputs, device, *, tol: float = 1e-4
) -> dict[str, float]:
    """Step 0: same inputs twice → near-identical logits.

    bf16/fp16 GPU kernels are not guaranteed bitwise-deterministic, so we accept
    a small tolerance instead of requiring an exact zero diff.
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    logits_a = forward_logits(model, processor, input_ids=input_ids, attention_mask=attention_mask, mm_inputs=mm_inputs, device=device)
    logits_b = forward_logits(model, processor, input_ids=input_ids, attention_mask=attention_mask, mm_inputs=mm_inputs, device=device)
    max_diff = (logits_a - logits_b).abs().max().item()
    return {"max_logit_diff": max_diff, "tol": tol, "stable": float(max_diff < tol)}


@dataclass
class SharedContext:
    """Mode-independent state computed once, reused by every replacement mode."""

    oracle_variant: str
    prompt_inputs: dict
    full_ids: torch.Tensor
    full_mask: torch.Tensor
    prompt_len: int
    image_pad_tokens: int
    response_ids: list[int]
    y_ref_text: str
    logits_orig: torch.Tensor
    embeds_a: torch.Tensor
    embeds_b: torch.Tensor
    cosines: torch.Tensor
    base_embeds: torch.Tensor
    second_span: tuple[int, int]


def prepare_shared_context(
    model,
    processor,
    *,
    row: dict,
    chart_image: Image.Image,
    target_refocus: Image.Image,
    donor_refocus: Image.Image,
    target_request_id: str,
    max_new_tokens: int,
    device: torch.device,
) -> SharedContext:
    """Compute everything that does NOT depend on the replacement mode exactly once.

    The single greedy ``y_ref`` produced here is shared by high_sim / random /
    low_sim so the three controls compare against an identical reference answer.
    """
    messages, oracle_variant = build_turn1_messages(
        row,
        chart_image=chart_image,
        refocus_image=target_refocus,
        request_id=target_request_id,
        use_diversified_oracle=True,
    )
    images = [chart_image, target_refocus]
    prompt_inputs = build_processor_inputs(processor, messages, images)
    input_ids = prompt_inputs["input_ids"]
    attention_mask = prompt_inputs["attention_mask"]
    prompt_len = input_ids.shape[1]
    image_pad_tokens = int((input_ids == processor.image_token_id).sum().item())

    # Step 5: greedy reference answer on original target context (ONCE, shared)
    response_ids = greedy_generate(
        model,
        processor,
        input_ids=input_ids,
        attention_mask=attention_mask,
        mm_inputs=prompt_inputs,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    y_ref_text = processor.tokenizer.decode(response_ids, skip_special_tokens=True)

    resp_tensor = torch.tensor([response_ids], dtype=input_ids.dtype)
    full_ids = torch.cat([input_ids, resp_tensor], dim=1)
    full_mask = torch.ones_like(full_ids)

    logits_orig = forward_logits(
        model,
        processor,
        input_ids=full_ids,
        attention_mask=full_mask,
        mm_inputs=prompt_inputs,
        device=device,
    )

    embeds_a = encode_image_embeds(model, processor, target_refocus, device)
    embeds_b = encode_image_embeds(model, processor, donor_refocus, device)
    cosines = pairwise_cosine(embeds_a, embeds_b)

    spans = image_pad_spans(input_ids[0], processor.image_token_id)
    if len(spans) < 2:
        raise ValueError(f"expected >=2 image spans, got {spans}")
    second_span = spans[1]

    base_embeds = build_inputs_embeds(model, full_ids, full_mask, prompt_inputs, device)

    return SharedContext(
        oracle_variant=oracle_variant,
        prompt_inputs=prompt_inputs,
        full_ids=full_ids,
        full_mask=full_mask,
        prompt_len=prompt_len,
        image_pad_tokens=image_pad_tokens,
        response_ids=response_ids,
        y_ref_text=y_ref_text,
        logits_orig=logits_orig,
        embeds_a=embeds_a,
        embeds_b=embeds_b,
        cosines=cosines,
        base_embeds=base_embeds,
        second_span=second_span,
    )


def run_single_replacement(
    model,
    processor,
    *,
    ctx: SharedContext,
    target_request_id: str,
    donor_request_id: str,
    reuse_ratio: float,
    mode: ReplacementMode,
    seed: int,
    device: torch.device,
    group_uid: str,
    dataset_row: int,
) -> ExperimentResult:
    replace_idx = select_replacement_indices(ctx.cosines, reuse_ratio, mode, seed)

    patched_embeds = apply_token_replacement(
        ctx.base_embeds,
        ctx.second_span,
        ctx.embeds_a,
        ctx.embeds_b,
        replace_idx,
    )
    logits_repl = forward_logits(
        model,
        processor,
        input_ids=ctx.full_ids,
        attention_mask=ctx.full_mask,
        mm_inputs=ctx.prompt_inputs,
        inputs_embeds=patched_embeds,
        device=device,
    )

    metrics = teacher_force_metrics(
        ctx.logits_orig,
        logits_repl,
        prompt_len=ctx.prompt_len,
        response_ids=ctx.response_ids,
    )

    return ExperimentResult(
        group_uid=group_uid,
        dataset_row=dataset_row,
        target_request_id=target_request_id,
        donor_request_id=donor_request_id,
        reuse_ratio=reuse_ratio,
        mode=mode,
        n_image_tokens=ctx.second_span[1] - ctx.second_span[0],
        n_replaced=len(replace_idx),
        mean_pairwise_cos=float(ctx.cosines.mean().item()),
        oracle_variant=ctx.oracle_variant,
        prompt_tokens=ctx.prompt_len,
        image_pad_tokens=ctx.image_pad_tokens,
        y_ref_text=ctx.y_ref_text,
        metrics=metrics,
    )


def dry_run_check(args) -> None:
    parquet = Path(args.parquet)
    dump_dir = Path(args.image_dump_dir)
    if not parquet.is_file():
        raise FileNotFoundError(parquet)
    if not dump_dir.is_dir():
        raise FileNotFoundError(dump_dir)

    if args.batch:
        specs = discover_groups(dump_dir, min_branches=args.min_branches)
        capped = specs[: args.max_groups] if args.max_groups > 0 else specs
        total_pairs = 0
        for spec in capped:
            pairs = build_pairs(spec.refocus_request_ids, pairs_per_group=args.pairs_per_group, seed=args.seed)
            total_pairs += len(pairs)
        reuse_ratios = parse_reuse_ratios(args.reuse_ratios) if args.reuse_ratios else [args.reuse_ratio]
        n_forward = total_pairs * len(reuse_ratios) * 3  # 3 modes
        print("=== Phase 2 BATCH dry-run OK ===")
        print(f"image_dump_dir     : {dump_dir}")
        print(f"usable groups      : {len(specs)} (>= {args.min_branches} refocus branches)")
        print(f"groups this run    : {len(capped)} (--max-groups={args.max_groups})")
        print(f"pairs_per_group    : {args.pairs_per_group} → total pairs={total_pairs}")
        print(f"reuse_ratios       : {reuse_ratios}")
        print(f"replacement forwards: {n_forward}  (pairs × ratios × 3 modes)")
        print("\nfirst few groups (uid, dataset_row, n_branches):")
        for spec in capped[:8]:
            print(f"  {spec.group_uid[:8]}  row={spec.dataset_row}  branches={len(spec.refocus_request_ids)}")
        print("\nNext: run with --batch (no --dry-run) on a GPU machine.")
        return

    row = load_parquet_row(parquet, args.dataset_row)
    chart_t, refocus_t = image_dump_paths(dump_dir, args.group_uid, args.target_request_id)
    chart_d, refocus_d = image_dump_paths(dump_dir, args.group_uid, args.donor_request_id)
    for p in (chart_t, refocus_t, chart_d, refocus_d):
        if not p.is_file():
            raise FileNotFoundError(p)

    chart = Image.open(chart_t).convert("RGB")
    refocus_t_img = Image.open(refocus_t).convert("RGB")
    messages, variant = build_turn1_messages(
        row,
        chart_image=chart,
        refocus_image=refocus_t_img,
        request_id=args.target_request_id,
    )

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    inputs = build_processor_inputs(processor, messages, [chart, refocus_t_img])
    n_pads = int((inputs["input_ids"] == processor.image_token_id).sum().item())
    spans = image_pad_spans(inputs["input_ids"][0], processor.image_token_id)

    print("=== Phase 2 dry-run OK ===")
    print(f"group_uid          : {args.group_uid}")
    print(f"dataset_row        : {args.dataset_row}")
    print(f"target branch      : {args.target_request_id[:8]}…")
    print(f"donor branch       : {args.donor_request_id[:8]}…")
    print(f"chart / refocus    : {chart_t.name} / {refocus_t.name}")
    print(f"donor refocus      : {refocus_d.name}")
    print(f"oracle variant     : {variant}")
    print(f"prompt_tokens      : {inputs['input_ids'].shape[1]}")
    print(f"image_pad_tokens   : {n_pads} (spans={spans})")
    print(f"reuse_ratio        : {args.reuse_ratio}")
    print(f"replace count@ratio: {max(1, int(round((spans[1][1]-spans[1][0]) * args.reuse_ratio)))}")
    print("\nNext: run without --dry-run on a GPU machine.")


def parse_reuse_ratios(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    for v in vals:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"reuse ratio must be in (0,1], got {v}")
    return vals


# ---------------------------------------------------------------------------
# Shared run helpers (used by both single-case and batch modes)
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "group_uid",
    "dataset_row",
    "target_request_id",
    "donor_request_id",
    "reuse_ratio",
    "mode",
    "delta_nll_per_token",
    "top1_match_rate",
    "delta_nll",
    "n_replaced",
    "n_image_tokens",
    "mean_pairwise_cos",
    "prompt_tokens",
    "response_len",
]


def result_to_row(r: ExperimentResult) -> dict:
    return {
        "group_uid": r.group_uid,
        "dataset_row": r.dataset_row,
        "target_request_id": r.target_request_id,
        "donor_request_id": r.donor_request_id,
        "reuse_ratio": r.reuse_ratio,
        "mode": r.mode,
        "delta_nll_per_token": r.metrics["delta_nll_per_token"],
        "top1_match_rate": r.metrics["top1_match_rate"],
        "delta_nll": r.metrics["delta_nll"],
        "n_replaced": r.n_replaced,
        "n_image_tokens": r.n_image_tokens,
        "mean_pairwise_cos": r.mean_pairwise_cos,
        "prompt_tokens": r.prompt_tokens,
        "response_len": r.metrics["response_len"],
    }


def write_results_csv(results: list[ExperimentResult], path: Path) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(result_to_row(r))


def aggregate_results(results: list[ExperimentResult]) -> list[dict]:
    """Aggregate per-pair metrics into mean/median/p90 by (reuse_ratio, mode)."""
    buckets: dict[tuple[float, str], dict[str, list[float]]] = {}
    for r in results:
        key = (r.reuse_ratio, r.mode)
        b = buckets.setdefault(key, {"delta_nll_per_token": [], "top1_match_rate": []})
        b["delta_nll_per_token"].append(r.metrics["delta_nll_per_token"])
        b["top1_match_rate"].append(r.metrics["top1_match_rate"])

    rows: list[dict] = []
    for (ratio, mode), vals in sorted(buckets.items()):
        dnll = np.asarray(vals["delta_nll_per_token"], dtype=float)
        top1 = np.asarray(vals["top1_match_rate"], dtype=float)
        rows.append(
            {
                "reuse_ratio": ratio,
                "mode": mode,
                "n_pairs": int(dnll.size),
                "dnll_mean": float(dnll.mean()),
                "dnll_median": float(np.median(dnll)),
                "dnll_p90": float(np.percentile(dnll, 90)),
                "top1_mean": float(top1.mean()),
                "top1_median": float(np.median(top1)),
                "top1_p10": float(np.percentile(top1, 10)),
            }
        )
    return rows


def write_aggregate_csv(agg_rows: list[dict], path: Path) -> None:
    import csv

    if not agg_rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)


def load_model_and_processor(args):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, processor, device


def run_one_pair(
    model,
    processor,
    *,
    row: dict,
    dump_dir: Path,
    group_uid: str,
    dataset_row: int,
    target_request_id: str,
    donor_request_id: str,
    reuse_ratios: list[float],
    modes: list[ReplacementMode],
    seed: int,
    max_new_tokens: int,
    sanity_tol: float,
    device: torch.device,
    run_sanity: bool,
) -> tuple[list[ExperimentResult], dict | None]:
    """Run all (ratio, mode) combinations for one (target, donor) pair.

    Shares one greedy y_ref + baseline + per-image embeds across every
    ratio/mode so high/random/low compare against an identical reference.
    """
    chart_path, refocus_t_path = image_dump_paths(dump_dir, group_uid, target_request_id)
    _, refocus_d_path = image_dump_paths(dump_dir, group_uid, donor_request_id)
    chart = Image.open(chart_path).convert("RGB")
    refocus_target = Image.open(refocus_t_path).convert("RGB")
    refocus_donor = Image.open(refocus_d_path).convert("RGB")

    ctx = prepare_shared_context(
        model,
        processor,
        row=row,
        chart_image=chart,
        target_refocus=refocus_target,
        donor_refocus=refocus_donor,
        target_request_id=target_request_id,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    sanity = None
    if run_sanity:
        sanity = run_sanity_stability(
            model, processor, ctx.prompt_inputs, ctx.prompt_inputs, device, tol=sanity_tol
        )

    results: list[ExperimentResult] = []
    for ratio in reuse_ratios:
        for mode in modes:
            res = run_single_replacement(
                model,
                processor,
                ctx=ctx,
                target_request_id=target_request_id,
                donor_request_id=donor_request_id,
                reuse_ratio=ratio,
                mode=mode,
                seed=seed,
                device=device,
                group_uid=group_uid,
                dataset_row=dataset_row,
            )
            results.append(res)
    return results, sanity


def run_batch(args) -> None:
    parquet = Path(args.parquet)
    dump_dir = Path(args.image_dump_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reuse_ratios = parse_reuse_ratios(args.reuse_ratios) if args.reuse_ratios else [args.reuse_ratio]
    modes: list[ReplacementMode] = ["high_sim", "random", "low_sim"]

    specs = discover_groups(dump_dir, min_branches=args.min_branches)
    if args.max_groups > 0:
        specs = specs[: args.max_groups]
    if not specs:
        raise SystemExit(f"no usable groups found under {dump_dir}")
    print(f"[batch] {len(specs)} groups; ratios={reuse_ratios}; pairs_per_group={args.pairs_per_group}")

    model, processor, device = load_model_and_processor(args)

    all_results: list[ExperimentResult] = []
    sanity_first: dict | None = None
    skipped: list[str] = []

    for gi, spec in enumerate(specs):
        try:
            row = load_parquet_row(parquet, spec.dataset_row)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{spec.group_uid} (row {spec.dataset_row}): {exc}")
            continue
        pairs = build_pairs(spec.refocus_request_ids, pairs_per_group=args.pairs_per_group, seed=args.seed)
        print(f"[batch] ({gi + 1}/{len(specs)}) group={spec.group_uid[:8]} row={spec.dataset_row} pairs={len(pairs)}")
        for pi, (target_rid, donor_rid) in enumerate(pairs):
            try:
                results, sanity = run_one_pair(
                    model,
                    processor,
                    row=row,
                    dump_dir=dump_dir,
                    group_uid=spec.group_uid,
                    dataset_row=spec.dataset_row,
                    target_request_id=target_rid,
                    donor_request_id=donor_rid,
                    reuse_ratios=reuse_ratios,
                    modes=modes,
                    seed=args.seed,
                    max_new_tokens=args.max_new_tokens,
                    sanity_tol=args.sanity_tol,
                    device=device,
                    run_sanity=(sanity_first is None),
                )
            except Exception as exc:  # noqa: BLE001
                skipped.append(f"{spec.group_uid[:8]} {target_rid[:8]}<-{donor_rid[:8]}: {exc}")
                continue
            if sanity is not None and sanity_first is None:
                sanity_first = sanity
                print(
                    f"  [sanity] max_logit_diff={sanity['max_logit_diff']:.6g} "
                    f"tol={sanity['tol']:.1g} stable={bool(sanity['stable'])}"
                )
            all_results.extend(results)

    if not all_results:
        raise SystemExit("no results produced; see skipped list above")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = out_dir / f"phase2_batch_{stamp}.json"
    out_csv = out_dir / f"phase2_batch_{stamp}.csv"
    agg_csv = out_dir / f"phase2_batch_{stamp}_agg.csv"

    agg_rows = aggregate_results(all_results)
    write_results_csv(all_results, out_csv)
    write_aggregate_csv(agg_rows, agg_csv)

    payload = {
        "created_at": stamp,
        "mode": "batch",
        "config": vars(args),
        "sanity": sanity_first,
        "n_groups": len(specs),
        "n_results": len(all_results),
        "skipped": skipped,
        "aggregate": agg_rows,
        "results": [asdict(r) for r in all_results],
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n[batch] {len(all_results)} pair-results; skipped={len(skipped)}")
    print("\n=== aggregate (ΔNLL/token lower=better; top1 higher=better) ===")
    print(f"{'ratio':>6} {'mode':>9} {'n':>4} {'dnll_mean':>10} {'dnll_p90':>9} {'top1_mean':>9} {'top1_p10':>9}")
    for r in agg_rows:
        print(
            f"{r['reuse_ratio']:>6.2f} {r['mode']:>9} {r['n_pairs']:>4} "
            f"{r['dnll_mean']:>10.5f} {r['dnll_p90']:>9.5f} {r['top1_mean']:>9.3f} {r['top1_p10']:>9.3f}"
        )
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {agg_csv}")


def run_single(args) -> None:
    reuse_ratios = parse_reuse_ratios(args.reuse_ratios) if args.reuse_ratios else [args.reuse_ratio]
    modes: list[ReplacementMode] = ["high_sim", "random", "low_sim"]

    parquet = Path(args.parquet)
    dump_dir = Path(args.image_dump_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    row = load_parquet_row(parquet, args.dataset_row)
    model, processor, device = load_model_and_processor(args)

    results, sanity = run_one_pair(
        model,
        processor,
        row=row,
        dump_dir=dump_dir,
        group_uid=args.group_uid,
        dataset_row=args.dataset_row,
        target_request_id=args.target_request_id,
        donor_request_id=args.donor_request_id,
        reuse_ratios=reuse_ratios,
        modes=modes,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        sanity_tol=args.sanity_tol,
        device=device,
        run_sanity=True,
    )
    if results:
        print(f"[ref] y_ref={results[0].y_ref_text!r}")
        print(f"[ref] mean_pairwise_cos(target,donor)={results[0].mean_pairwise_cos:.4f}")
    if sanity is not None:
        print(
            f"[sanity] max_logit_diff={sanity['max_logit_diff']:.6g} "
            f"tol={sanity['tol']:.1g} stable={bool(sanity['stable'])}"
        )
    for res in results:
        m = res.metrics
        print(
            f"[run] reuse_ratio={res.reuse_ratio:.2f} mode={res.mode}  "
            f"ΔNLL/token={m['delta_nll_per_token']:.6f}  "
            f"top1_match={m['top1_match_rate']:.3f}  "
            f"replaced={res.n_replaced}/{res.n_image_tokens}"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    group_short = args.group_uid.split("-")[0]
    out_json = out_dir / f"phase2_{group_short}_{stamp}.json"
    out_csv = out_dir / f"phase2_{group_short}_{stamp}.csv"

    payload = {
        "created_at": stamp,
        "mode": "single",
        "config": vars(args),
        "sanity": sanity,
        "results": [asdict(r) for r in results],
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_results_csv(results, out_csv)

    print(f"\nWrote {out_json}")
    print(f"Wrote {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 token replacement ablation")
    parser.add_argument("--parquet", default="/data/refocus_chart_multiturn/train.parquet")
    parser.add_argument(
        "--image-dump-dir",
        default="profile_logs_refocus_chart/image_dump_refocus_chart_multiturn_bs64_n4_diversified",
    )
    parser.add_argument("--group-uid", default=DEFAULT_GROUP_UID)
    parser.add_argument("--dataset-row", type=int, default=DEFAULT_DATASET_ROW)
    parser.add_argument("--target-request-id", default=DEFAULT_TARGET_REQUEST_ID)
    parser.add_argument("--donor-request-id", default=DEFAULT_DONOR_REQUEST_ID)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--reuse-ratio", type=float, default=DEFAULT_REUSE_RATIO)
    parser.add_argument("--reuse-ratios", default="", help="comma-separated sweep, e.g. 0.1,0.2,0.4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--sanity-tol", type=float, default=1e-4, help="max |Δlogit| allowed in stability check")
    parser.add_argument(
        "--out-dir",
        default="profile_logs_refocus_chart/similarity_diversified/phase2",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate paths/context only")
    # batch mode: auto-enumerate groups/pairs from the image-dump manifest
    parser.add_argument("--batch", action="store_true", help="run over many groups/pairs from manifest")
    parser.add_argument("--max-groups", type=int, default=20, help="batch: cap number of groups (<=0 = all)")
    parser.add_argument("--pairs-per-group", type=int, default=6, help="batch: directed A<-B pairs per group (<=0 = all)")
    parser.add_argument("--min-branches", type=int, default=2, help="batch: require >= this many refocus branches")
    args = parser.parse_args()

    if args.dry_run:
        dry_run_check(args)
        return

    if args.batch:
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
