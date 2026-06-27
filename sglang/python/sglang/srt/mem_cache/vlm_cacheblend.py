"""VLM-CacheBlend: selective visual-KV reuse for Qwen2.5-VL GRPO turn1 prefill.

This module ports CacheBlend (non-prefix KV reuse + selective recompute, originally
for RAG / text chunks) to the LLM prefill stage (§6) of a VLM rollout.

Within one GRPO group (same ``agent_uid``) at turn1, the *refocus image* token span
is highly similar across branches but is **not** a common prefix (the per-branch
``turn0 response`` diverges before it), so SGLang's RadixCache cannot reuse it. Here a
*donor* branch's per-layer image-token K/V is stored and reused by *recipient*
branches; only a small high-deviation subset of visual tokens is recomputed to restore
cross-attention.

Design doc:
    verl_vision/examples/profile/shared/docs/VLM_CACHEBLEND_DESIGN.md

Everything is gated behind ``SGLANG_VLM_CACHEBLEND`` (default off). When disabled, the
SGLang code path is byte-for-byte the original; this file is import-safe and never runs
unless explicitly enabled by the integration hooks in ``models/qwen2.py``.

The functions here are pure tensor ops (no SGLang-internal coupling) so they can be
unit tested directly. Integration glue lives in the model forward hooks.
"""

from __future__ import annotations

import os
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


# --------------------------------------------------------------------------- #
# Config / macro toggles
# --------------------------------------------------------------------------- #
def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class VLMCacheBlendConfig:
    """Snapshot of ``SGLANG_VLM_CACHEBLEND*`` env vars (read once at import)."""

    enabled: bool = False
    # "same"     : require donor/recipient image span at identical absolute positions
    #              (no RoPE rotation needed). v1, safest.
    # "rerotate" : re-rotate donor K via mRoPE when positions differ. v2/experimental.
    pos_mode: str = "same"
    # token selection for recompute: "topr" | "kvdev" | "sim"
    select_mode: str = "topr"
    # fraction of image tokens to recompute (HKVD budget). 1.0 == full prefill.
    recompute_ratio: float = 0.15
    # for select_mode="sim": tokens with similarity < this are forced to recompute.
    sim_threshold: float = 0.90
    # only act on these turns (turn1 is the GRPO refocus turn).
    target_turn: int = 1
    # Which logical image slot to cache. -1 means the last image span, which is the
    # refocus image in the turn1 prompts used by this experiment.
    target_image_slot: int = -1
    # max number of groups kept in the donor store (LRU).
    max_groups: int = 64
    # emit per-request stats to stderr/log for profiling.
    verbose: bool = False
    # First-stage recipient fast path: during prefill, after the backend writes
    # recipient K/V but before the attention kernel reads the cache, replace reusable
    # image-token K/V slots with donor K/V. This makes the current prefill attention
    # consume the blended cache. Skipping recipient QKV/MLP compute is a later backend
    # optimization.
    fast_path: bool = True
    # Skip the decoder MLP for reusable image tokens. Their per-layer K/V is supplied by
    # the donor before attention, and prompt image-token hidden states are not used for
    # logits. Text tokens and recompute image tokens still run the full layer.
    skip_reuse_mlp: bool = True
    # Skip QKV/O projections for reusable image tokens. The attention kernel still runs
    # with the original batch metadata, but those tokens get donor K/V and zero local
    # layer deltas.
    skip_reuse_qkv_proj: bool = True
    # Skip attention queries for reusable image tokens when the backend can preserve
    # the original causal alignment by running only active contiguous query ranges.
    skip_reuse_attention: bool = True
    # Legacy diagnostic path that overwrites KV after attention has already consumed the
    # recipient cache. This is unsafe for rollout quality and must stay off unless one is
    # explicitly debugging the old plumbing.
    unsafe_post_attention_overlay: bool = False
    # Store donor image-span K/V on CPU instead of GPU. Off by default (GPU keeps the
    # reuse path copy-free); turn on to cap GPU memory at the cost of a per-layer H2D copy
    # on reuse. With ``max_groups`` donors x num_layers x image-span K/V this is the main
    # GPU-memory lever for the LLM side.
    donor_to_cpu: bool = False

    @staticmethod
    def from_env() -> "VLMCacheBlendConfig":
        return VLMCacheBlendConfig(
            enabled=_env_flag("SGLANG_VLM_CACHEBLEND", "0"),
            pos_mode=os.environ.get("SGLANG_VLM_CACHEBLEND_POS_MODE", "same").strip().lower(),
            select_mode=os.environ.get("SGLANG_VLM_CACHEBLEND_SELECT", "topr").strip().lower(),
            recompute_ratio=_env_float("SGLANG_VLM_CACHEBLEND_RECOMPUTE_RATIO", 0.15),
            sim_threshold=_env_float("SGLANG_VLM_CACHEBLEND_SIM_THRESHOLD", 0.90),
            target_turn=_env_int("SGLANG_VLM_CACHEBLEND_TARGET_TURN", 1),
            target_image_slot=_env_int("SGLANG_VLM_CACHEBLEND_TARGET_IMAGE_SLOT", -1),
            max_groups=_env_int("SGLANG_VLM_CACHEBLEND_MAX_GROUPS", 64),
            verbose=_env_flag("SGLANG_VLM_CACHEBLEND_VERBOSE", "0"),
            fast_path=_env_flag("SGLANG_VLM_CACHEBLEND_FAST_PATH", "1"),
            skip_reuse_mlp=_env_flag("SGLANG_VLM_CACHEBLEND_SKIP_REUSE_MLP", "1"),
            skip_reuse_qkv_proj=_env_flag(
                "SGLANG_VLM_CACHEBLEND_SKIP_REUSE_QKV_PROJ", "1"
            ),
            skip_reuse_attention=_env_flag(
                "SGLANG_VLM_CACHEBLEND_SKIP_REUSE_ATTENTION", "1"
            ),
            unsafe_post_attention_overlay=_env_flag(
                "SGLANG_VLM_CACHEBLEND_UNSAFE_POST_ATTENTION_OVERLAY", "0"
            ),
            donor_to_cpu=_env_flag("SGLANG_VLM_CACHEBLEND_DONOR_TO_CPU", "0"),
        )


_CONFIG = VLMCacheBlendConfig.from_env()
_DUMP_ONCE_ENABLED = _env_flag("SGLANG_VLM_CACHEBLEND_DUMP_ONCE", "0")
_DUMP_ONCE_LOCK = threading.Lock()
_DUMP_ONCE_DONE = False


def get_config() -> VLMCacheBlendConfig:
    return _CONFIG


def cacheblend_enabled() -> bool:
    return _CONFIG.enabled


def reload_config_from_env() -> VLMCacheBlendConfig:
    """Re-read env (useful for tests)."""
    global _CONFIG, _DUMP_ONCE_ENABLED, _DUMP_ONCE_DONE
    _CONFIG = VLMCacheBlendConfig.from_env()
    _DUMP_ONCE_ENABLED = _env_flag("SGLANG_VLM_CACHEBLEND_DUMP_ONCE", "0")
    _DUMP_ONCE_DONE = False
    return _CONFIG


# --------------------------------------------------------------------------- #
# Donor KV store (group-keyed, per-layer image-token K/V)
# --------------------------------------------------------------------------- #
@dataclass
class DonorLayerKV:
    """One layer's stored image-span K/V for a donor branch.

    Shapes:
        k, v: ``[n_img_tok, n_kv_head, head_dim]`` (CPU or GPU, contiguous)
    """

    k: torch.Tensor
    v: torch.Tensor


@dataclass
class DonorEntry:
    """All layers of a donor branch's refocus image-token span."""

    group_key: Tuple
    n_image_tokens: int
    grid_sig: Tuple  # (t, h, w) per image slot; reuse is refused if recipient differs
    # mrope positions of the image span, shape [3, n_img_tok] (t/h/w sections).
    positions: Optional[torch.Tensor] = None
    layers: Dict[int, DonorLayerKV] = field(default_factory=dict)
    complete: bool = False

    def record_layer(self, layer_id: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.layers[layer_id] = DonorLayerKV(k=k, v=v)

    def has_layer(self, layer_id: int) -> bool:
        return layer_id in self.layers


@dataclass
class RecipientKVBlendPlan:
    request_id: str
    group_key: Tuple
    img_locs: torch.Tensor
    positions: Optional[torch.Tensor]
    recompute_mask: torch.Tensor
    grid_sig: Tuple
    n_image_tokens: int
    reused_tokens: int
    recomputed_tokens: int
    pos_mode: str
    select_mode: str
    # For select_mode="kvdev": the recompute set is not known until a bootstrap layer
    # has measured per-token KV deviation against the donor. Until then the mask marks
    # every token "recompute" (so no reuse/skip happens) and ``pending_deviation`` is
    # True; ``bootstrap_layer_id`` is the layer that finalizes the mask.
    bootstrap_layer_id: int = -1
    pending_deviation: bool = False


@dataclass(frozen=True)
class RecipientActiveQueryRanges:
    """Contiguous non-reuse query ranges for a current extend batch.

    Ranges are expressed in flattened current-batch token offsets. ``range_end_lens``
    stores the per-range absolute sequence length at the end of the query range; FA3
    can use this as a temporary cache length so its bottom-right causal mask remains
    aligned with the original token positions.
    """

    query_indices: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seqlen_q: int
    range_end_lens: torch.Tensor
    range_batch_indices: torch.Tensor
    range_req_indices: torch.Tensor


class DonorKVStore:
    """Thread-safe LRU store of donor branches keyed by GRPO group key.

    Mirrors the role of ``_GROUP_CACHE`` in ``grpo_similarity_cache.py`` but stores
    LLM-side K/V instead of ViT embeddings.
    """

    def __init__(self, max_groups: int = 64):
        self._lock = threading.Lock()
        self._store: "OrderedDict[Tuple, DonorEntry]" = OrderedDict()
        self._max_groups = max_groups

    def get_or_create_donor(
        self,
        group_key: Tuple,
        n_image_tokens: int,
        grid_sig: Tuple,
        positions: Optional[torch.Tensor],
    ) -> DonorEntry:
        with self._lock:
            entry = self._store.get(group_key)
            if entry is None:
                entry = DonorEntry(
                    group_key=group_key,
                    n_image_tokens=n_image_tokens,
                    grid_sig=grid_sig,
                    positions=positions,
                )
                self._store[group_key] = entry
                self._evict_if_needed()
            self._store.move_to_end(group_key)
            return entry

    def lookup(self, group_key: Tuple) -> Optional[DonorEntry]:
        with self._lock:
            entry = self._store.get(group_key)
            if entry is not None:
                self._store.move_to_end(group_key)
            return entry

    def mark_complete(self, group_key: Tuple) -> None:
        with self._lock:
            entry = self._store.get(group_key)
            if entry is not None:
                entry.complete = True

    def drop(self, group_key: Tuple) -> None:
        with self._lock:
            self._store.pop(group_key, None)

    def _evict_if_needed(self) -> None:
        while len(self._store) > self._max_groups:
            self._store.popitem(last=False)


_DONOR_STORE = DonorKVStore(max_groups=_CONFIG.max_groups)


def get_donor_store() -> DonorKVStore:
    return _DONOR_STORE


# --------------------------------------------------------------------------- #
# Position alignment
# --------------------------------------------------------------------------- #
def positions_match(donor_pos: Optional[torch.Tensor], recipient_pos: Optional[torch.Tensor]) -> bool:
    """True if donor/recipient image-span absolute positions are identical.

    In ``same`` pos_mode this must hold (Delta == 0) so donor K can be reused without
    rotation. ``positions`` are mrope positions, shape ``[3, n_img_tok]``.
    """
    if donor_pos is None or recipient_pos is None:
        return False
    if donor_pos.shape != recipient_pos.shape:
        return False
    return bool(torch.equal(donor_pos.to(torch.long), recipient_pos.to(torch.long)))


def _rotate_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply a neox-style rotation. x:[n,h,d], cos/sin:[n,d//2]."""
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


def rerotate_keys_mrope(
    donor_k: torch.Tensor,
    donor_pos: torch.Tensor,
    recipient_pos: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    mrope_section: List[int],
) -> torch.Tensor:
    """Re-rotate donor K from donor positions to recipient positions (mRoPE).

    RoPE attention scores depend only on relative position, so a key already rotated to
    ``donor_pos`` can be re-aligned to ``recipient_pos`` by rotating by the delta. For
    mRoPE the head_dim is split into t/h/w sections, each driven by its own positional
    axis; the delta rotation is applied section-wise.

    Args:
        donor_k: ``[n_img_tok, n_kv_head, head_dim]`` (already rotated at donor_pos).
        donor_pos / recipient_pos: ``[3, n_img_tok]`` long.
        cos_sin_cache: ``[max_pos, rotary_dim]`` (first half cos, second half sin).
        mrope_section: e.g. ``[16, 24, 24]`` summing to rotary_dim//2.

    Returns:
        K re-rotated to recipient positions, same shape as ``donor_k``.

    Note: experimental (pos_mode="rerotate"). v1 uses pos_mode="same" (delta==0) and
    never calls this.
    """
    delta = (recipient_pos.to(torch.long) - donor_pos.to(torch.long))  # [3, n]
    half = cos_sin_cache.shape[-1] // 2
    cos_cache = cos_sin_cache[..., :half]
    sin_cache = cos_sin_cache[..., half:]

    # Build per-token cos/sin by selecting each section from its own axis delta.
    # We rotate by absolute index = position; for a *delta* rotation we use the fact
    # that rotating by p_new - p_old equals composing the inverse donor rotation with
    # the recipient rotation. With a precomputed absolute cache we approximate the
    # delta rotation by indexing the cache at the (clamped) absolute delta when it is
    # non-negative, else applying the inverse. To stay exact and simple we instead
    # compute cos/sin for recipient and donor and compose.
    def gather(axis_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # axis_pos: [3, n]; returns cos,sin [n, half] assembled section-wise.
        cos_parts, sin_parts = [], []
        offset = 0
        for sec_idx, sec_len in enumerate(mrope_section):
            idx = axis_pos[sec_idx].clamp_min(0)
            c = cos_cache.index_select(0, idx)[:, offset : offset + sec_len]
            s = sin_cache.index_select(0, idx)[:, offset : offset + sec_len]
            cos_parts.append(c)
            sin_parts.append(s)
            offset += sec_len
        return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)

    cos_d, sin_d = gather(donor_pos.to(torch.long))
    cos_r, sin_r = gather(recipient_pos.to(torch.long))
    # inverse donor rotation: angle -> -angle  (cos same, sin negated)
    k_unrot = _rotate_neox(donor_k, cos_d, -sin_d)
    k_aligned = _rotate_neox(k_unrot, cos_r, sin_r)
    return k_aligned


def align_donor_keys(
    donor_k: torch.Tensor,
    donor_pos: Optional[torch.Tensor],
    recipient_pos: Optional[torch.Tensor],
    cfg: VLMCacheBlendConfig,
    cos_sin_cache: Optional[torch.Tensor] = None,
    mrope_section: Optional[List[int]] = None,
) -> Optional[torch.Tensor]:
    """Return donor K aligned to recipient positions, or None if alignment refused."""
    if cfg.pos_mode == "same":
        return donor_k if positions_match(donor_pos, recipient_pos) else None
    if cfg.pos_mode == "rerotate":
        if cos_sin_cache is None or mrope_section is None or donor_pos is None or recipient_pos is None:
            return None
        return rerotate_keys_mrope(donor_k, donor_pos, recipient_pos, cos_sin_cache, mrope_section)
    return None


# --------------------------------------------------------------------------- #
# Selective recompute selection
# --------------------------------------------------------------------------- #
def _topr_count(n: int, ratio: float) -> int:
    ratio = min(max(ratio, 0.0), 1.0)
    return int(min(n, max(0, round(n * ratio))))


def select_recompute_tokens(
    n_image_tokens: int,
    cfg: VLMCacheBlendConfig,
    *,
    deviation: Optional[torch.Tensor] = None,
    similarity: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return a boolean mask ``[n_image_tokens]`` marking tokens to recompute.

    Modes:
      * ``topr``  : top-r% by ``deviation`` if given else first r% (deterministic).
      * ``kvdev`` : top-r% by ``deviation`` (``deviation`` must be provided;
                    CacheBlend's high-KV-deviation set).
      * ``sim``   : recompute tokens whose ``similarity`` < ``sim_threshold`` (the
                    visual-similarity prior); falls back to top-r% within that set if a
                    ratio cap is desired.
    True  => recompute (donor not trusted).
    False => reuse donor KV.
    """
    device = device or (deviation.device if deviation is not None else
                        similarity.device if similarity is not None else torch.device("cpu"))
    mask = torch.zeros(n_image_tokens, dtype=torch.bool, device=device)
    if n_image_tokens == 0 or cfg.recompute_ratio >= 1.0:
        return torch.ones(n_image_tokens, dtype=torch.bool, device=device)

    mode = cfg.select_mode
    if mode == "sim" and similarity is not None:
        mask = similarity < cfg.sim_threshold
        # optional cap to recompute_ratio (keep the lowest-similarity ones)
        cap = _topr_count(n_image_tokens, cfg.recompute_ratio)
        if cap > 0 and int(mask.sum()) > cap:
            worst = torch.topk(-similarity, cap).indices
            mask = torch.zeros_like(mask)
            mask[worst] = True
        return mask

    score = deviation
    if score is None:
        # deterministic fallback: recompute the first r% tokens.
        k = _topr_count(n_image_tokens, cfg.recompute_ratio)
        mask[:k] = True
        return mask

    k = _topr_count(n_image_tokens, cfg.recompute_ratio)
    if k <= 0:
        return mask
    top = torch.topk(score.reshape(-1), k).indices
    mask[top] = True
    return mask


def kv_deviation(
    recipient_k: torch.Tensor,
    donor_k_aligned: torch.Tensor,
) -> torch.Tensor:
    """Per-token KV deviation = L2 over (head, dim), shape ``[n_img_tok]``.

    Used by select_mode="kvdev". ``recipient_k`` is the freshly computed (and rotated)
    key for the image span at a bootstrap layer; ``donor_k_aligned`` is the donor key
    re-aligned to recipient positions.
    """
    diff = (recipient_k - donor_k_aligned).reshape(recipient_k.shape[0], -1)
    return torch.linalg.vector_norm(diff, dim=-1)


def finalize_recipient_plan_deviation(
    plan: "RecipientKVBlendPlan",
    recipient_k: torch.Tensor,
    donor_k_aligned: torch.Tensor,
    cfg: VLMCacheBlendConfig,
) -> torch.Tensor:
    """Resolve a deferred ``kvdev`` plan from a bootstrap layer's recipient K.

    ``recipient_k`` is the freshly computed key for the image span at the bootstrap
    layer; ``donor_k_aligned`` is the donor key re-aligned to the recipient positions.
    Picks the high-KV-deviation ("high-risk") tokens to recompute and reuses the rest.
    Mutates ``plan`` in place and returns the per-token deviation (for logging/tests).
    """
    deviation = kv_deviation(recipient_k.float(), donor_k_aligned.float())
    mask = select_recompute_tokens(
        plan.n_image_tokens, cfg, deviation=deviation, device=deviation.device
    )
    plan.recompute_mask = mask.to(device=plan.recompute_mask.device, dtype=torch.bool)
    plan.recomputed_tokens = int(plan.recompute_mask.sum().item())
    plan.reused_tokens = int(plan.n_image_tokens - plan.recomputed_tokens)
    plan.pending_deviation = False
    return deviation


# --------------------------------------------------------------------------- #
# KV blend
# --------------------------------------------------------------------------- #
def blend_kv(
    donor_k: torch.Tensor,
    donor_v: torch.Tensor,
    recomputed_k: torch.Tensor,
    recomputed_v: torch.Tensor,
    recompute_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse donor and recomputed K/V per token.

    recompute_mask True  -> take recomputed value.
    recompute_mask False -> take donor value.

    All tensors share shape ``[n_img_tok, n_kv_head, head_dim]`` except mask
    ``[n_img_tok]``. The donor tensors are assumed already position-aligned.
    """
    m = recompute_mask.reshape(-1, 1, 1)
    k = torch.where(m, recomputed_k, donor_k)
    v = torch.where(m, recomputed_v, donor_v)
    return k, v


def build_recipient_kv_blend_plan(
    ctx: RequestContext,
    img_locs: torch.Tensor,
    img_positions: Optional[torch.Tensor],
) -> Tuple[Optional[RecipientKVBlendPlan], "CacheBlendStats"]:
    cfg = get_config()
    stats = CacheBlendStats(
        role="recipient",
        request_id=ctx.request_id,
        n_image_tokens=int(img_locs.numel()),
        pos_mode=cfg.pos_mode,
        select_mode=cfg.select_mode,
    )
    if not cfg.fast_path:
        stats.fallback_reason = "recipient_fast_path_disabled"
        return None, stats.finalize()
    donor = get_donor_store().lookup(ctx.group_key)
    if donor is None or not donor.complete:
        stats.fallback_reason = "donor_not_ready"
        return None, stats.finalize()
    if tuple(donor.grid_sig) != tuple(ctx.grid_sig):
        stats.fallback_reason = "grid_mismatch"
        return None, stats.finalize()
    if int(donor.n_image_tokens) != int(img_locs.numel()):
        stats.fallback_reason = "image_token_count_mismatch"
        return None, stats.finalize()
    if cfg.pos_mode == "same" and not positions_match(donor.positions, img_positions):
        stats.fallback_reason = "position_mismatch"
        return None, stats.finalize()
    if cfg.pos_mode == "rerotate" and (donor.positions is None or img_positions is None):
        stats.fallback_reason = "missing_positions_for_rerotate"
        return None, stats.finalize()

    n_img = int(img_locs.numel())
    bootstrap_layer_id = -1
    pending_deviation = False
    if cfg.select_mode == "kvdev" and donor.layers:
        # Defer the recompute selection: mark every token "recompute" so no reuse/skip
        # happens until the bootstrap (lowest donor) layer measures the recipient's true
        # per-token KV deviation, then reuse the low-deviation tokens from the next layer.
        bootstrap_layer_id = min(int(layer) for layer in donor.layers.keys())
        pending_deviation = True
        recompute_mask = torch.ones(n_img, dtype=torch.bool, device=img_locs.device)
    else:
        recompute_mask = select_recompute_tokens(n_img, cfg, device=img_locs.device)
    stats.eligible = True
    stats.recomputed_tokens = int(recompute_mask.sum().item())
    stats.reused_tokens = n_img - stats.recomputed_tokens
    stats.fallback_reason = (
        "recipient_kv_blend_deferred_kvdev"
        if pending_deviation
        else "recipient_kv_blend_planned"
    )
    plan = RecipientKVBlendPlan(
        request_id=ctx.request_id,
        group_key=ctx.group_key,
        img_locs=img_locs.detach().clone(),
        positions=(img_positions.detach().clone() if img_positions is not None else None),
        recompute_mask=recompute_mask.detach().clone(),
        grid_sig=ctx.grid_sig,
        n_image_tokens=n_img,
        reused_tokens=stats.reused_tokens,
        recomputed_tokens=stats.recomputed_tokens,
        pos_mode=cfg.pos_mode,
        select_mode=cfg.select_mode,
        bootstrap_layer_id=bootstrap_layer_id,
        pending_deviation=pending_deviation,
    )
    return plan, stats.finalize()


def _rerotate_plan_keys_if_needed(
    donor_k: torch.Tensor,
    donor_positions: Optional[torch.Tensor],
    recipient_positions: Optional[torch.Tensor],
    *,
    pos_mode: str,
    rotary_emb: Any,
) -> torch.Tensor:
    if pos_mode != "rerotate":
        return donor_k
    if donor_positions is None or recipient_positions is None or rotary_emb is None:
        return donor_k
    cos_sin_cache = getattr(rotary_emb, "cos_sin_cache", None)
    mrope_section = getattr(rotary_emb, "mrope_section", None)
    if cos_sin_cache is None or not mrope_section:
        return donor_k
    return rerotate_keys_mrope(
        donor_k,
        donor_positions.to(device=donor_k.device),
        recipient_positions.to(device=donor_k.device),
        cos_sin_cache.to(device=donor_k.device, dtype=donor_k.dtype),
        mrope_section,
    )


def apply_recipient_kv_blend_for_layer(
    *,
    forward_batch: Any,
    layer_id: int,
    cache_locs: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    rotary_emb: Any = None,
) -> int:
    """Blend donor K/V into recipient image-token slots for one layer.

    This is intended to run after the backend has materialized recipient K/V but
    before the prefill attention kernel consumes those K/V. It updates both the
    token-to-KV pool and, when supplied, the direct ``k``/``v`` tensors used by
    ragged/triton extend kernels.
    """
    plans = get_recipient_blend_plans()
    if not plans:
        return 0
    pool = getattr(forward_batch, "token_to_kv_pool", None)
    if pool is None:
        return 0
    try:
        k_buf = pool.get_key_buffer(layer_id)
        v_buf = pool.get_value_buffer(layer_id)
    except Exception:
        return 0

    total_reused = 0
    for plan in plans:
        donor = get_donor_store().lookup(plan.group_key)
        if donor is None or not donor.complete or not donor.has_layer(layer_id):
            continue
        donor_layer = donor.layers[layer_id]
        img_locs = plan.img_locs.to(device=k_buf.device, dtype=torch.long)
        if plan.pending_deviation and layer_id == plan.bootstrap_layer_id:
            # Bootstrap layer: the recipient's own K for the image span is already in the
            # pool (set_kv_buffer ran before this hook) and not yet overwritten by the
            # donor. Measure per-token KV deviation NOW (before any overwrite), finalize
            # the high-deviation ("high-risk") recompute set, then fall through so this
            # SAME layer also writes donor K/V for the reused tokens. Falling through is
            # required for correctness: the attention active-query ranges and the MLP skip
            # later in this layer read the just-finalized mask and will skip the reused
            # tokens, so those tokens must already hold donor K/V here (not the recipient's
            # own K/V). ``pending_deviation`` is cleared so this runs only once.
            if img_locs.numel() > 0:
                donor_k_boot = _rerotate_plan_keys_if_needed(
                    donor_layer.k.to(device=k_buf.device, dtype=k_buf.dtype),
                    donor.positions,
                    plan.positions,
                    pos_mode=plan.pos_mode,
                    rotary_emb=rotary_emb,
                )
                finalize_recipient_plan_deviation(
                    plan, k_buf[img_locs], donor_k_boot, get_config()
                )
            else:
                plan.pending_deviation = False
        recompute_mask = plan.recompute_mask.to(device=k_buf.device, dtype=torch.bool)
        reuse_mask = ~recompute_mask
        if img_locs.numel() == 0 or not bool(reuse_mask.any()):
            continue
        donor_k = donor_layer.k.to(device=k_buf.device, dtype=k_buf.dtype)
        donor_v = donor_layer.v.to(device=v_buf.device, dtype=v_buf.dtype)
        donor_k = _rerotate_plan_keys_if_needed(
            donor_k,
            donor.positions,
            plan.positions,
            pos_mode=plan.pos_mode,
            rotary_emb=rotary_emb,
        ).to(device=k_buf.device, dtype=k_buf.dtype)
        dst = img_locs[reuse_mask]
        donor_k_reuse = donor_k[reuse_mask]
        donor_v_reuse = donor_v[reuse_mask]
        k_buf[dst] = donor_k_reuse
        v_buf[dst] = donor_v_reuse
        if cache_locs is not None and k is not None and v is not None:
            locs = cache_locs.to(device=k.device, dtype=torch.long).reshape(-1)
            matches = dst.to(device=k.device).reshape(-1, 1).eq(locs.reshape(1, -1))
            donor_idx, kv_idx = matches.nonzero(as_tuple=True)
            if kv_idx.numel() > 0:
                k[kv_idx] = donor_k_reuse.to(device=k.device, dtype=k.dtype)[donor_idx]
                v[kv_idx] = donor_v_reuse.to(device=v.device, dtype=v.dtype)[donor_idx]
        total_reused += int(dst.numel())
        mark_recipient_blend_used(plan.request_id)
    return total_reused


# --------------------------------------------------------------------------- #
# Stats / metrics
# --------------------------------------------------------------------------- #
@dataclass
class CacheBlendStats:
    role: str = "none"  # "donor" | "recipient" | "none"
    used: bool = False
    eligible: bool = False
    fallback_reason: str = ""
    request_id: str = ""
    n_image_tokens: int = 0
    reused_tokens: int = 0
    recomputed_tokens: int = 0
    pos_mode: str = ""
    select_mode: str = ""
    recompute_ratio_effective: float = 0.0
    extend_wall_ms: float = -1.0
    attention_skipped_tokens: int = 0
    attention_active_ranges: int = 0

    def finalize(self) -> "CacheBlendStats":
        if self.n_image_tokens > 0:
            self.recompute_ratio_effective = self.recomputed_tokens / self.n_image_tokens
        return self

    def to_dict(self) -> Dict[str, str]:
        return {
            "cacheblend_role": self.role,
            "cacheblend_used": "1" if self.used else "0",
            "cacheblend_eligible": "1" if self.eligible else "0",
            "cacheblend_fallback_reason": self.fallback_reason,
            "cacheblend_request_id": self.request_id,
            "cacheblend_n_image_tokens": str(self.n_image_tokens),
            "cacheblend_reused_tokens": str(self.reused_tokens),
            "cacheblend_recomputed_tokens": str(self.recomputed_tokens),
            "cacheblend_pos_mode": self.pos_mode,
            "cacheblend_select_mode": self.select_mode,
            "cacheblend_recompute_ratio": f"{self.recompute_ratio_effective:.4f}",
            "cacheblend_extend_wall_ms": (
                "" if self.extend_wall_ms < 0 else f"{self.extend_wall_ms:.3f}"
            ),
            "cacheblend_attention_skipped_tokens": str(self.attention_skipped_tokens),
            "cacheblend_attention_active_ranges": str(self.attention_active_ranges),
        }


def log_stats(stats: CacheBlendStats) -> None:
    set_last_stats(stats)
    if _CONFIG.verbose:
        import sys

        parts = " ".join(f"{k}={v}" for k, v in stats.to_dict().items())
        print(f"[VLM-CacheBlend] {parts}", file=sys.stderr, flush=True)


_LAST_STATS = threading.local()


def set_last_stats(stats: Optional[CacheBlendStats]) -> None:
    _LAST_STATS.value = stats


def pop_last_stats() -> Optional[CacheBlendStats]:
    stats = getattr(_LAST_STATS, "value", None)
    _LAST_STATS.value = None
    return stats


# --------------------------------------------------------------------------- #
# Per-request context (set by the scheduler / agent loop, read by the LLM forward)
# --------------------------------------------------------------------------- #
#
# Role/group resolution needs GRPO agent metadata (``agent_uid``, ``agent_turn``)
# which already exists in ``grpo_similarity_cache`` (``_REQUEST_META`` keyed by rid).
# The LLM forward (Qwen2Model.forward) does not see the rid directly, so the
# integration sets a thread-local context just before model.forward, the same place
# the ViT cache resolves donor/recipient. Default is None => the forward hook is a
# pure no-op and the baseline path is unchanged.
@dataclass
class RequestContext:
    group_key: Tuple
    role: str  # "donor" | "recipient"
    image_token_id: int
    image_token_values: Tuple[int, ...] = ()
    request_id: str = ""
    global_step: int = -1
    agent_uid: str = ""
    agent_turn: int = -1
    target_image_slot: int = -1
    grid_sig: Tuple = ()
    request_index: int = 0


@dataclass
class BatchRequestContext:
    contexts: Tuple[RequestContext, ...]


_REQUEST_CTX = threading.local()
_SOURCE_INPUT_IDS = threading.local()
_RECIPIENT_BLEND_PLANS = threading.local()
_RECIPIENT_BLEND_USED = threading.local()
_RECIPIENT_ATTENTION_SKIP = threading.local()
_RECIPIENT_ACTIVE_QUERY_RANGES = threading.local()


def set_request_context(ctx: Optional[RequestContext]) -> None:
    _REQUEST_CTX.value = ctx


def get_request_context() -> Optional[RequestContext]:
    return getattr(_REQUEST_CTX, "value", None)


def set_source_input_ids(input_ids: Optional[torch.Tensor]) -> None:
    _SOURCE_INPUT_IDS.value = input_ids


def get_source_input_ids() -> Optional[torch.Tensor]:
    return getattr(_SOURCE_INPUT_IDS, "value", None)


def set_recipient_blend_plans(plans: Optional[Tuple[RecipientKVBlendPlan, ...]]) -> None:
    _RECIPIENT_BLEND_PLANS.value = tuple(plans or ())
    _RECIPIENT_BLEND_USED.value = set()
    _RECIPIENT_ATTENTION_SKIP.value = {}
    _RECIPIENT_ACTIVE_QUERY_RANGES.value = None


def get_recipient_blend_plans() -> Tuple[RecipientKVBlendPlan, ...]:
    return tuple(getattr(_RECIPIENT_BLEND_PLANS, "value", ()) or ())


def clear_recipient_blend_plans() -> None:
    _RECIPIENT_BLEND_PLANS.value = ()
    _RECIPIENT_BLEND_USED.value = set()
    _RECIPIENT_ATTENTION_SKIP.value = {}
    _RECIPIENT_ACTIVE_QUERY_RANGES.value = None


def recipient_reuse_token_indices(
    cache_locs: Optional[torch.Tensor],
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return current-batch token offsets whose image K/V is donor-reused."""
    if cache_locs is None:
        return torch.empty(0, dtype=torch.long, device=device)
    plans = get_recipient_blend_plans()
    if not plans:
        return torch.empty(0, dtype=torch.long, device=device or cache_locs.device)
    locs = cache_locs.reshape(-1).to(device=device or cache_locs.device, dtype=torch.long)
    pieces = []
    for plan in plans:
        reuse_mask = ~plan.recompute_mask.to(device=locs.device, dtype=torch.bool)
        if not bool(reuse_mask.any()):
            continue
        dst = plan.img_locs.to(device=locs.device, dtype=torch.long)[reuse_mask]
        matches = dst.reshape(-1, 1).eq(locs.reshape(1, -1))
        _, kv_idx = matches.nonzero(as_tuple=True)
        if kv_idx.numel() > 0:
            pieces.append(kv_idx)
    if not pieces:
        return torch.empty(0, dtype=torch.long, device=locs.device)
    return torch.unique(torch.cat(pieces), sorted=True)


def _active_query_ranges_cache_key(forward_batch: Any) -> Optional[Tuple]:
    out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
    seq_lens = getattr(forward_batch, "seq_lens", None)
    req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
    extend_seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if (
        out_cache_loc is None
        or seq_lens is None
        or req_pool_indices is None
        or extend_seq_lens_cpu is None
    ):
        return None
    return (
        int(out_cache_loc.data_ptr()),
        tuple(out_cache_loc.shape),
        int(seq_lens.data_ptr()),
        tuple(seq_lens.shape),
        int(req_pool_indices.data_ptr()),
        tuple(req_pool_indices.shape),
        tuple(int(x) for x in extend_seq_lens_cpu),
        len(get_recipient_blend_plans()),
    )


def recipient_active_query_ranges(forward_batch: Any) -> Optional[RecipientActiveQueryRanges]:
    """Build active-query ranges after dropping reusable image-token queries.

    The returned ranges are valid for kernels whose causal mask can be aligned by
    treating each contiguous active range as the final query block of a temporary
    sequence ending at that range's original endpoint.
    """
    cache_key = _active_query_ranges_cache_key(forward_batch)
    cached = getattr(_RECIPIENT_ACTIVE_QUERY_RANGES, "value", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    if not cacheblend_enabled():
        return None
    cfg = get_config()
    if not cfg.fast_path:
        return None
    out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
    extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
    extend_seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
    seq_lens = getattr(forward_batch, "seq_lens", None)
    req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
    if (
        out_cache_loc is None
        or extend_seq_lens is None
        or extend_seq_lens_cpu is None
        or seq_lens is None
        or req_pool_indices is None
    ):
        if cache_key is not None:
            _RECIPIENT_ACTIVE_QUERY_RANGES.value = (cache_key, None)
        return None
    reuse_idx = recipient_reuse_token_indices(out_cache_loc, device=out_cache_loc.device)
    total_tokens = int(out_cache_loc.numel())
    if reuse_idx.numel() == 0 or reuse_idx.numel() >= total_tokens:
        if cache_key is not None:
            _RECIPIENT_ACTIVE_QUERY_RANGES.value = (cache_key, None)
        return None

    device = out_cache_loc.device
    active = torch.ones(total_tokens, dtype=torch.bool, device=device)
    active[reuse_idx] = False
    query_parts: List[torch.Tensor] = []
    cu_values: List[int] = [0]
    end_lens: List[int] = []
    batch_indices: List[int] = []
    req_indices: List[int] = []
    cursor = 0
    seq_lens_cpu = seq_lens.detach().cpu().tolist()
    req_indices_cpu = req_pool_indices.detach().cpu().tolist()
    for batch_idx, extend_len in enumerate(list(extend_seq_lens_cpu)):
        extend_len = int(extend_len)
        if extend_len <= 0:
            continue
        local = active[cursor : cursor + extend_len].detach().cpu().tolist()
        start = None
        for local_idx, is_active in enumerate(local + [False]):
            if is_active and start is None:
                start = local_idx
            elif not is_active and start is not None:
                end = local_idx
                idx = torch.arange(
                    cursor + start, cursor + end, dtype=torch.long, device=device
                )
                query_parts.append(idx)
                cu_values.append(cu_values[-1] + int(end - start))
                prefix_len = int(seq_lens_cpu[batch_idx]) - extend_len
                end_lens.append(prefix_len + end)
                batch_indices.append(int(batch_idx))
                req_indices.append(int(req_indices_cpu[batch_idx]))
                start = None
        cursor += extend_len

    if not query_parts:
        if cache_key is not None:
            _RECIPIENT_ACTIVE_QUERY_RANGES.value = (cache_key, None)
        return None
    query_indices = torch.cat(query_parts)
    cu = torch.tensor(cu_values, dtype=torch.int32, device=device)
    range_end_lens = torch.tensor(end_lens, dtype=torch.int32, device=device)
    range_batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=device)
    range_req_indices = torch.tensor(req_indices, dtype=torch.long, device=device)
    max_q = max((cu_values[i + 1] - cu_values[i]) for i in range(len(cu_values) - 1))
    ranges = RecipientActiveQueryRanges(
        query_indices=query_indices,
        cu_seqlens_q=cu,
        max_seqlen_q=int(max_q),
        range_end_lens=range_end_lens,
        range_batch_indices=range_batch_indices,
        range_req_indices=range_req_indices,
    )
    if cache_key is not None:
        _RECIPIENT_ACTIVE_QUERY_RANGES.value = (cache_key, ranges)
    return ranges


def mark_recipient_blend_used(request_id: str) -> None:
    used = getattr(_RECIPIENT_BLEND_USED, "value", None)
    if used is None:
        used = set()
    used.add(str(request_id))
    _RECIPIENT_BLEND_USED.value = used


def mark_recipient_attention_skip_used(active_ranges: RecipientActiveQueryRanges) -> None:
    """Record that a backend successfully skipped reusable attention queries."""
    skipped = getattr(_RECIPIENT_ATTENTION_SKIP, "value", None)
    if skipped is None:
        skipped = {}
    n_ranges = int(active_ranges.range_end_lens.numel())
    for plan in get_recipient_blend_plans():
        if plan.reused_tokens <= 0:
            continue
        entry = skipped.setdefault(
            str(plan.request_id),
            {"tokens": 0, "ranges": 0},
        )
        entry["tokens"] = max(int(entry.get("tokens", 0)), int(plan.reused_tokens))
        entry["ranges"] = max(int(entry.get("ranges", 0)), n_ranges)
    _RECIPIENT_ATTENTION_SKIP.value = skipped


def recipient_attention_skip_stats(request_id: str) -> Tuple[int, int]:
    skipped = getattr(_RECIPIENT_ATTENTION_SKIP, "value", {}) or {}
    entry = skipped.get(str(request_id), {})
    return int(entry.get("tokens", 0)), int(entry.get("ranges", 0))


def recipient_blend_was_used(request_id: str) -> bool:
    return str(request_id) in (getattr(_RECIPIENT_BLEND_USED, "value", set()) or set())


def build_group_key(
    agent_uid: str,
    agent_turn: int,
    grid_sig: Tuple,
    *,
    global_step: int = -1,
    image_slot: int = -1,
) -> Tuple:
    """Group key for donor/recipient matching: branches of the same GRPO group at the
    same turn and identical image grid share a donor."""
    return (int(global_step), str(agent_uid), int(agent_turn), int(image_slot), tuple(grid_sig))


def _tensor_rows_to_grid_sigs(value: Any) -> List[Tuple[int, int, int]]:
    if value is None:
        return []
    try:
        if isinstance(value, torch.Tensor):
            rows = value.detach().cpu().reshape(-1, 3).tolist()
        else:
            rows = torch.as_tensor(value).reshape(-1, 3).tolist()
    except Exception:
        return []
    out = []
    for row in rows:
        if len(row) >= 3:
            out.append((int(row[0]), int(row[1]), int(row[2])))
    return out


def _image_grid_sigs_from_req(req: Any) -> List[Tuple[int, int, int]]:
    """Best-effort extraction of logical image grids from an SGLang request."""
    mm = getattr(req, "multimodal_inputs", None)
    if mm is None:
        return []
    items = getattr(mm, "mm_items", None) or []
    grids: List[Tuple[int, int, int]] = []
    for item in items:
        grids.extend(_tensor_rows_to_grid_sigs(getattr(item, "image_grid_thw", None)))
        model_specific = getattr(item, "model_specific_data", None) or {}
        if isinstance(model_specific, dict):
            grids.extend(_tensor_rows_to_grid_sigs(model_specific.get("image_grid_thw")))
    return grids


def _image_pad_values_from_req(req: Any) -> Tuple[int, ...]:
    """Image placeholder values used in SGLang's padded/radix input ids."""
    mm = getattr(req, "multimodal_inputs", None)
    if mm is None:
        return ()
    values = []
    for item in getattr(mm, "mm_items", None) or []:
        try:
            is_image = bool(item.is_image())
        except Exception:
            modality = getattr(getattr(item, "modality", None), "name", "")
            is_image = modality in ("IMAGE", "MULTI_IMAGES")
        if not is_image:
            continue
        pad_value = getattr(item, "pad_value", None)
        if pad_value is None:
            continue
        try:
            values.append(int(pad_value))
        except Exception:
            continue
    return tuple(values)


def _token_span_count(seq: List[int], values: Tuple[int, ...]) -> int:
    value_set = set(int(v) for v in values if v is not None)
    if not seq or not value_set:
        return 0
    n_spans = 0
    in_span = False
    for tok in seq:
        is_match = int(tok) in value_set
        if is_match and not in_span:
            n_spans += 1
        in_span = is_match
    return n_spans


def _image_span_count_from_req(req: Any, image_token_id: Optional[int]) -> int:
    if req is None:
        return 0
    fill_ids = list(getattr(req, "fill_ids", None) or [])
    if not fill_ids:
        fill_ids = list(getattr(req, "origin_input_ids", None) or []) + list(
            getattr(req, "output_ids", None) or []
        )
    pad_values = _image_pad_values_from_req(req)
    n_spans = _token_span_count(fill_ids, pad_values)
    if n_spans:
        return n_spans
    if image_token_id is None:
        return 0
    return _token_span_count(fill_ids, (int(image_token_id),))


def _safe_int(value: Any) -> Optional[int]:
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return int(value.detach().cpu().item())
        return int(value)
    except Exception:
        return None


def _short_positions(seq: List[int], values: Tuple[int, ...], limit: int = 32) -> Dict[str, Any]:
    value_set = set(int(v) for v in values if v is not None)
    positions = [i for i, tok in enumerate(seq) if int(tok) in value_set]
    counts = {str(v): 0 for v in value_set}
    for tok in seq:
        tok = int(tok)
        if tok in value_set:
            counts[str(tok)] = counts.get(str(tok), 0) + 1
    return {
        "values": [int(v) for v in values if v is not None],
        "count_total": len(positions),
        "counts_by_value": counts,
        "positions_first": positions[:limit],
        "positions_last": positions[-limit:] if len(positions) > limit else positions[:],
    }


def _tensor_to_short_list(value: Any, limit: int = 32) -> List[int]:
    if value is None:
        return []
    try:
        if isinstance(value, torch.Tensor):
            flat = value.detach().cpu().reshape(-1).tolist()
        else:
            flat = list(value)
    except Exception:
        return []
    out = []
    for item in flat[:limit]:
        safe = _safe_int(item)
        if safe is not None:
            out.append(safe)
    return out


def _shape_of(value: Any) -> Optional[List[int]]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(x) for x in shape]
    except Exception:
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": _shape_of(value),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    try:
        return str(value)
    except Exception:
        return "<unserializable>"


def _dump_path() -> str:
    explicit = os.environ.get("SGLANG_VLM_CACHEBLEND_DUMP_PATH", "").strip()
    if explicit:
        return explicit
    log_dir = os.environ.get("SGLANG_INFERENCE_LOG_DIR", ".")
    suffix = os.environ.get("SGLANG_INFERENCE_LOG_SUFFIX", "").strip()
    filename = (
        f"vlm_cacheblend_req_dump_{suffix}.jsonl"
        if suffix
        else "vlm_cacheblend_req_dump.jsonl"
    )
    return os.path.join(log_dir, filename)


def _mm_item_dump(item: Any) -> Dict[str, Any]:
    try:
        is_image = bool(item.is_image())
    except Exception:
        is_image = None
    modality = getattr(getattr(item, "modality", None), "name", None)
    model_specific = getattr(item, "model_specific_data", None) or {}
    return {
        "modality": modality,
        "is_image": is_image,
        "hash": _safe_int(getattr(item, "hash", None)),
        "pad_value": _safe_int(getattr(item, "pad_value", None)),
        "offsets": getattr(item, "offsets", None),
        "image_grid_thw": _tensor_rows_to_grid_sigs(getattr(item, "image_grid_thw", None)),
        "model_specific_keys": sorted(model_specific.keys()) if isinstance(model_specific, dict) else [],
        "model_specific_image_grid_thw": _tensor_rows_to_grid_sigs(
            model_specific.get("image_grid_thw") if isinstance(model_specific, dict) else None
        ),
    }


def maybe_dump_request_debug(
    *,
    forward_batch: Any,
    ctx: RequestContext,
    input_ids: Optional[torch.Tensor],
    out_cache_loc: Optional[torch.Tensor],
    request_locs_reason: str = "",
    img_locs: Optional[torch.Tensor] = None,
) -> None:
    """One-shot structural dump for diagnosing image-span lookup.

    Enabled by ``SGLANG_VLM_CACHEBLEND_DUMP_ONCE=1``. The dump intentionally avoids
    prompt text and records token ids / metadata only.
    """
    global _DUMP_ONCE_DONE
    if not _DUMP_ONCE_ENABLED or ctx.role != "donor":
        return
    with _DUMP_ONCE_LOCK:
        if _DUMP_ONCE_DONE:
            return
        _DUMP_ONCE_DONE = True

    reqs = list(getattr(forward_batch, "reqs", None) or [])
    req_index = int(getattr(ctx, "request_index", 0))
    req = reqs[req_index] if 0 <= req_index < len(reqs) else None
    mm = getattr(req, "multimodal_inputs", None) if req is not None else None
    mm_items = list(getattr(mm, "mm_items", None) or []) if mm is not None else []
    fill_ids = list(getattr(req, "fill_ids", None) or []) if req is not None else []
    if req is not None and not fill_ids:
        fill_ids = list(getattr(req, "origin_input_ids", None) or []) + list(
            getattr(req, "output_ids", None) or []
        )
    origin_input_ids = list(getattr(req, "origin_input_ids", None) or []) if req is not None else []
    origin_unpadded = (
        list(getattr(req, "origin_input_ids_unpadded", None) or []) if req is not None else []
    )
    output_ids = list(getattr(req, "output_ids", None) or []) if req is not None else []
    pad_values = tuple(v for v in _image_pad_values_from_req(req) if v is not None) if req is not None else ()
    all_ctx_values = tuple(ctx.image_token_values or ())
    image_token_tuple = (ctx.image_token_id,) if ctx.image_token_id is not None else ()

    req_to_token_pool = getattr(forward_batch, "req_to_token_pool", None)
    req_to_token = getattr(req_to_token_pool, "req_to_token", None)
    req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
    req_pool_idx = None
    mapped_locs = []
    if req_to_token is not None and req_pool_indices is not None and fill_ids:
        try:
            flat_req_pool_indices = req_pool_indices.reshape(-1)
            if req_index < int(flat_req_pool_indices.numel()):
                req_pool_idx = _safe_int(flat_req_pool_indices[req_index])
            ids = torch.as_tensor(fill_ids, dtype=torch.long, device=req_to_token.device)
            mask, _, _ = _image_token_mask_with_span_info(
                ids,
                ctx.image_token_id,
                ctx.target_image_slot,
                ctx.image_token_values,
            )
            token_pos = torch.nonzero(mask, as_tuple=False).reshape(-1)
            if req_pool_idx is not None and token_pos.numel() > 0:
                max_len = req_to_token.shape[1]
                token_pos = token_pos[token_pos < max_len]
                mapped_locs = _tensor_to_short_list(req_to_token[req_pool_idx, token_pos], limit=64)
        except Exception as exc:
            mapped_locs = [f"error:{type(exc).__name__}:{str(exc)[:120]}"]

    input_ids_list = _tensor_to_short_list(input_ids, limit=int(input_ids.numel()) if isinstance(input_ids, torch.Tensor) and input_ids.numel() <= 8192 else 8192)
    row = {
        "timestamp": f"{time.time():.6f}",
        "pid": os.getpid(),
        "request_locs_reason": request_locs_reason,
        "img_locs_numel": int(img_locs.numel()) if isinstance(img_locs, torch.Tensor) else None,
        "img_locs_first": _tensor_to_short_list(img_locs, limit=64),
        "ctx": {
            "request_id": ctx.request_id,
            "role": ctx.role,
            "agent_uid": ctx.agent_uid,
            "agent_turn": ctx.agent_turn,
            "global_step": ctx.global_step,
            "target_image_slot": ctx.target_image_slot,
            "grid_sig": ctx.grid_sig,
            "group_key": ctx.group_key,
            "image_token_id": ctx.image_token_id,
            "image_token_values": all_ctx_values,
        },
        "forward_batch": {
            "req_count": len(reqs),
            "mode": str(getattr(getattr(forward_batch, "forward_mode", None), "name", "")),
            "batch_size": getattr(forward_batch, "batch_size", None),
            "seq_lens_cpu": _tensor_to_short_list(getattr(forward_batch, "seq_lens_cpu", None), limit=16),
            "extend_prefix_lens_cpu": list(getattr(forward_batch, "extend_prefix_lens_cpu", None) or []),
            "extend_seq_lens_cpu": list(getattr(forward_batch, "extend_seq_lens_cpu", None) or []),
            "req_pool_indices": _tensor_to_short_list(req_pool_indices, limit=16),
            "out_cache_loc_shape": _shape_of(out_cache_loc),
            "out_cache_loc_first": _tensor_to_short_list(out_cache_loc, limit=32),
            "req_to_token_shape": _shape_of(req_to_token),
            "req_pool_idx": req_pool_idx,
            "mapped_image_locs_first": mapped_locs,
        },
        "req": {
            "rid": str(getattr(req, "rid", "") or "") if req is not None else "",
            "agent_uid": str(getattr(req, "agent_uid", "") or "") if req is not None else "",
            "agent_turn": getattr(req, "agent_turn", None) if req is not None else None,
            "training_global_step": getattr(req, "training_global_step", None) if req is not None else None,
            "prefix_indices_len": len(getattr(req, "prefix_indices", None) or []) if req is not None else 0,
            "fill_ids_len": len(fill_ids),
            "origin_input_ids_len": len(origin_input_ids),
            "origin_input_ids_unpadded_len": len(origin_unpadded),
            "output_ids_len": len(output_ids),
            "mm_present": mm is not None,
            "mm_im_token_id": _safe_int(getattr(mm, "im_token_id", None)) if mm is not None else None,
            "mm_video_token_id": _safe_int(getattr(mm, "video_token_id", None)) if mm is not None else None,
            "mm_image_pad_len": getattr(mm, "image_pad_len", None) if mm is not None else None,
            "mm_num_image_tokens": getattr(mm, "num_image_tokens", None) if mm is not None else None,
            "mm_mrope_positions_shape": _shape_of(getattr(mm, "mrope_positions", None)) if mm is not None else None,
            "mm_items": [_mm_item_dump(item) for item in mm_items],
            "image_grid_sigs": _image_grid_sigs_from_req(req) if req is not None else [],
            "image_pad_values_from_req": pad_values,
        },
        "intersections": {
            "fill_ids_vs_ctx_image_token_values": _short_positions(fill_ids, all_ctx_values),
            "fill_ids_vs_req_pad_values": _short_positions(fill_ids, pad_values),
            "fill_ids_vs_image_token_id": _short_positions(fill_ids, image_token_tuple),
            "origin_input_ids_vs_req_pad_values": _short_positions(origin_input_ids, pad_values),
            "origin_unpadded_vs_image_token_id": _short_positions(origin_unpadded, image_token_tuple),
            "current_input_ids_vs_ctx_values": _short_positions(input_ids_list, all_ctx_values),
            "current_input_ids_vs_image_token_id": _short_positions(input_ids_list, image_token_tuple),
        },
    }

    path = _dump_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")


def select_grid_sig_for_req(
    req: Any,
    target_image_slot: int,
    image_token_id: Optional[int] = None,
) -> Tuple[int, Tuple[int, int, int]]:
    """Return the current-request image span slot and matching grid signature.

    In multiturn rollout, ``mm_items``/grid metadata can include historical images,
    while the current prefill request may only contain the last N image spans. Align
    the current spans to the suffix of the grid list so ``-1`` still means the refocus
    image and an accidental global slot does not become out-of-range for fill_ids.
    """
    grids = _image_grid_sigs_from_req(req)
    if not grids:
        return target_image_slot, ()
    n_spans = _image_span_count_from_req(req, image_token_id)
    if n_spans > 0:
        span_slot = target_image_slot if target_image_slot >= 0 else n_spans - 1
        if span_slot < 0 or span_slot >= n_spans:
            span_slot = n_spans - 1
        grid_base = max(0, len(grids) - n_spans)
        grid_slot = min(len(grids) - 1, grid_base + span_slot)
        return span_slot, grids[grid_slot]
    slot = target_image_slot if target_image_slot >= 0 else len(grids) - 1
    if slot < 0 or slot >= len(grids):
        slot = len(grids) - 1
    return slot, grids[slot]


def _build_request_context_for_req(
    forward_batch: Any,
    req: Any,
    req_index: int,
    *,
    image_token_id: int,
) -> Optional[RequestContext]:
    rid = str(getattr(req, "rid", "") or "")
    meta = _lookup_request_meta(rid)
    agent_uid = str(getattr(req, "agent_uid", "") or "")
    if not agent_uid:
        agent_uid = str(meta.get("agent_uid", "") or "")
    agent_turn = getattr(req, "agent_turn", None)
    parsed_turn = _parse_turn_from_rid(rid)
    if parsed_turn is not None:
        agent_turn = parsed_turn
    elif agent_turn is None:
        agent_turn = meta.get("agent_turn")
    if not agent_uid or agent_turn is None:
        log_stats(
            CacheBlendStats(
                role="none",
                request_id=rid,
                fallback_reason="missing_agent_meta",
            ).finalize()
        )
        return None
    cfg = get_config()
    agent_turn = int(agent_turn)
    if agent_turn != int(cfg.target_turn):
        log_stats(
            CacheBlendStats(
                role="none",
                request_id=rid,
                fallback_reason=f"agent_turn_mismatch:{agent_turn}",
            ).finalize()
        )
        return None
    global_step = getattr(req, "training_global_step", None)
    if global_step is None:
        global_step = meta.get("global_step")
    if global_step is None:
        global_step = getattr(forward_batch, "training_global_step", -1)
    try:
        global_step = int(global_step)
    except Exception:
        global_step = -1
    image_slot, grid_sig = select_grid_sig_for_req(
        req, cfg.target_image_slot, image_token_id=image_token_id
    )
    if not grid_sig:
        log_stats(
            CacheBlendStats(
                role="none",
                request_id=rid,
                fallback_reason="missing_image_grid",
            ).finalize()
        )
        return None
    group_key = build_group_key(
        agent_uid,
        agent_turn,
        grid_sig,
        global_step=global_step,
        image_slot=image_slot,
    )
    donor = get_donor_store().lookup(group_key)
    role = "recipient" if donor is not None and donor.complete else "donor"
    log_stats(
        CacheBlendStats(
            role=role,
            request_id=rid,
            pos_mode=cfg.pos_mode,
            select_mode=cfg.select_mode,
            fallback_reason="context_ready",
        ).finalize()
    )
    return RequestContext(
        group_key=group_key,
        role=role,
        image_token_id=int(image_token_id),
        image_token_values=_image_pad_values_from_req(req),
        request_id=rid,
        global_step=global_step,
        agent_uid=agent_uid,
        agent_turn=agent_turn,
        target_image_slot=image_slot,
        grid_sig=grid_sig,
        request_index=int(req_index),
    )


def _parse_turn_from_rid(rid: str) -> Optional[int]:
    try:
        from sglang.srt.mem_cache.grpo_similarity_cache import parse_turn_from_rid

        return parse_turn_from_rid(rid)
    except Exception:
        return None


def _lookup_request_meta(rid: str) -> Dict[str, Any]:
    try:
        from sglang.srt.mem_cache.grpo_similarity_cache import lookup_request_meta

        return lookup_request_meta(rid) or {}
    except Exception:
        return {}


def build_request_context_from_forward_batch(
    forward_batch: Any,
    *,
    image_token_id: Optional[int],
) -> Optional[RequestContext | BatchRequestContext]:
    """Create per-forward CacheBlend context for each eligible request."""
    if not cacheblend_enabled():
        return None
    set_last_stats(None)
    if image_token_id is None:
        log_stats(
            CacheBlendStats(
                role="none",
                fallback_reason="image_token_id_missing",
            ).finalize()
        )
        return None
    if not getattr(forward_batch, "forward_mode", None).is_extend():
        return None
    reqs = list(getattr(forward_batch, "reqs", None) or [])
    if not reqs:
        log_stats(
            CacheBlendStats(
                role="none",
                fallback_reason="empty_reqs",
            ).finalize()
        )
        return None
    contexts = tuple(
        ctx
        for i, req in enumerate(reqs)
        for ctx in [
            _build_request_context_for_req(
                forward_batch,
                req,
                i,
                image_token_id=int(image_token_id),
            )
        ]
        if ctx is not None
    )
    if not contexts:
        return None
    if len(contexts) == 1:
        return contexts[0]
    return BatchRequestContext(contexts=contexts)


def image_token_mask_for_slot(
    input_ids: torch.Tensor,
    image_token_id: Optional[int],
    target_image_slot: int = -1,
    image_token_values: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """Boolean mask for one contiguous logical image-token span in one request."""
    mask = None
    values = tuple(int(v) for v in (image_token_values or ()) if v is not None)
    if values:
        value_tensor = torch.as_tensor(values, device=input_ids.device, dtype=input_ids.dtype)
        mask = torch.isin(input_ids, value_tensor)
        if not bool(mask.any()):
            mask = None
    if mask is None:
        if image_token_id is None:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        mask = input_ids == int(image_token_id)
    if mask.numel() == 0 or not bool(mask.any()):
        return torch.zeros_like(mask, dtype=torch.bool)
    idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
    breaks = torch.nonzero(idx[1:] != idx[:-1] + 1, as_tuple=False).reshape(-1) + 1
    starts = torch.cat([idx.new_tensor([0]), breaks])
    ends = torch.cat([breaks, idx.new_tensor([idx.numel()])])
    n_spans = int(starts.numel())
    slot = target_image_slot if target_image_slot >= 0 else n_spans - 1
    if slot < 0 or slot >= n_spans:
        return torch.zeros_like(mask, dtype=torch.bool)
    selected = idx[starts[slot] : ends[slot]]
    out = torch.zeros_like(mask, dtype=torch.bool)
    out[selected] = True
    return out


def _empty_locs_like_forward_batch(
    forward_batch: Any,
    reason: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
    device = getattr(getattr(forward_batch, "out_cache_loc", None), "device", None)
    device = device or torch.device("cpu")
    return torch.empty(0, dtype=torch.long, device=device), None, reason


def _image_token_mask_with_span_info(
    input_ids: torch.Tensor,
    image_token_id: Optional[int],
    target_image_slot: int,
    image_token_values: Optional[Tuple[int, ...]],
) -> Tuple[torch.Tensor, int, str]:
    values = tuple(int(v) for v in (image_token_values or ()) if v is not None)
    match_source = "pad_value" if values else "image_token_id"
    mask = None
    if values:
        value_tensor = torch.as_tensor(values, device=input_ids.device, dtype=input_ids.dtype)
        mask = torch.isin(input_ids, value_tensor)
        if not bool(mask.any()):
            mask = None
    if mask is None:
        if image_token_id is None:
            return torch.zeros_like(input_ids, dtype=torch.bool), 0, "missing_image_token_id"
        mask = input_ids == int(image_token_id)
        match_source = "image_token_id"
    if mask.numel() == 0 or not bool(mask.any()):
        return torch.zeros_like(mask, dtype=torch.bool), 0, f"no_image_mask:{match_source}"
    idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
    breaks = torch.nonzero(idx[1:] != idx[:-1] + 1, as_tuple=False).reshape(-1) + 1
    starts = torch.cat([idx.new_tensor([0]), breaks])
    ends = torch.cat([breaks, idx.new_tensor([idx.numel()])])
    n_spans = int(starts.numel())
    slot = target_image_slot if target_image_slot >= 0 else n_spans - 1
    if slot < 0 or slot >= n_spans:
        return (
            torch.zeros_like(mask, dtype=torch.bool),
            n_spans,
            f"slot_out_of_range:{slot}/{n_spans}:{match_source}",
        )
    selected = idx[starts[slot] : ends[slot]]
    out = torch.zeros_like(mask, dtype=torch.bool)
    out[selected] = True
    return out, n_spans, f"ok:{slot}/{n_spans}:{match_source}"


def image_token_locs(
    input_ids: torch.Tensor,
    out_cache_loc: torch.Tensor,
    image_token_id: Optional[int],
    target_image_slot: int = -1,
    image_token_values: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """KV-pool slot indices of image tokens among the freshly-extended tokens.

    ``input_ids`` and ``out_cache_loc`` are aligned (one entry per new token in the
    extend batch). Returns the pool slots holding image-token K/V.
    """
    img_mask = image_token_mask_for_slot(
        input_ids,
        image_token_id,
        target_image_slot,
        image_token_values=image_token_values,
    )
    return out_cache_loc[img_mask]


def image_token_locs_from_request(
    forward_batch: Any,
    ctx: RequestContext,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
    """Locate image-token KV slots using the request's full padded token sequence.

    In a turn1 extend, image tokens are often part of the matched prefix rather than
    the freshly extended ``input_ids``/``out_cache_loc`` slice. ``req_to_token_pool``
    maps full request token positions to KV-pool slots, so it can recover those prefix
    image slots after the normal extend-slice lookup misses.
    """
    reqs = list(getattr(forward_batch, "reqs", None) or [])
    req_index = int(getattr(ctx, "request_index", 0))
    if req_index < 0 or req_index >= len(reqs):
        return _empty_locs_like_forward_batch(
            forward_batch, f"full_request_req_index_oob:{req_index}/{len(reqs)}"
        )

    req = reqs[req_index]
    full_ids = list(getattr(req, "fill_ids", None) or [])
    if not full_ids:
        full_ids = list(getattr(req, "origin_input_ids", None) or []) + list(
            getattr(req, "output_ids", None) or []
        )

    req_to_token_pool = getattr(forward_batch, "req_to_token_pool", None)
    req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
    if req_to_token_pool is None or req_pool_indices is None or not full_ids:
        missing = []
        if req_to_token_pool is None:
            missing.append("req_to_token_pool")
        if req_pool_indices is None:
            missing.append("req_pool_indices")
        if not full_ids:
            missing.append("full_ids")
        return _empty_locs_like_forward_batch(
            forward_batch, "full_request_missing_" + "+".join(missing)
        )

    req_to_token = getattr(req_to_token_pool, "req_to_token", None)
    if req_to_token is None:
        return _empty_locs_like_forward_batch(forward_batch, "full_request_missing_req_to_token")

    device = req_to_token.device
    ids = torch.as_tensor(full_ids, dtype=torch.long, device=device)
    img_mask, n_spans, span_reason = _image_token_mask_with_span_info(
        ids,
        ctx.image_token_id,
        ctx.target_image_slot,
        ctx.image_token_values,
    )
    if img_mask.numel() == 0 or not bool(img_mask.any()):
        return (
            torch.empty(0, dtype=torch.long, device=device),
            None,
            f"full_request_{span_reason}:len={ids.numel()}",
        )

    token_pos = torch.nonzero(img_mask, as_tuple=False).reshape(-1)
    flat_req_pool_indices = req_pool_indices.reshape(-1)
    if req_index >= int(flat_req_pool_indices.numel()):
        return _empty_locs_like_forward_batch(
            forward_batch,
            f"full_request_req_pool_index_oob:{req_index}/{int(flat_req_pool_indices.numel())}",
        )
    req_pool_idx = flat_req_pool_indices[req_index].to(device=device, dtype=torch.long)
    max_len = req_to_token.shape[1]
    token_pos = token_pos[token_pos < max_len]
    if token_pos.numel() == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            None,
            f"full_request_token_pos_oob:spans={n_spans}:max_len={max_len}",
        )

    img_locs = req_to_token[req_pool_idx, token_pos].to(dtype=torch.long)
    valid_locs = img_locs >= 0
    img_locs = img_locs[valid_locs]
    token_pos = token_pos[valid_locs]
    if img_locs.numel() == 0:
        return img_locs, None, f"full_request_no_valid_kv_locs:spans={n_spans}"

    img_positions = None
    mm_input = getattr(req, "multimodal_inputs", None)
    full_positions = getattr(mm_input, "mrope_positions", None)
    if isinstance(full_positions, torch.Tensor):
        pos_idx = token_pos.to(device=full_positions.device)
        pos_idx = pos_idx[pos_idx < full_positions.shape[-1]]
        if pos_idx.numel() == img_locs.numel():
            img_positions = full_positions[:, pos_idx].to(device=img_locs.device)
    return img_locs, img_positions, f"full_request_ok:{span_reason}"


def capture_donor_kv(
    forward_batch,
    layer_ids: List[int],
    img_locs: torch.Tensor,
    group_key: Tuple,
    grid_sig: Tuple,
    positions: Optional[torch.Tensor],
    to_cpu: bool = False,
) -> Optional[DonorEntry]:
    """Read image-token K/V out of the KV pool for every layer and store as donor.

    Low-risk, read-only: runs only for the donor branch when the macro is on. Does not
    touch attention kernels. Returns the populated DonorEntry (or None if no image
    tokens / pool unavailable).
    """
    pool = getattr(forward_batch, "token_to_kv_pool", None)
    if pool is None or img_locs.numel() == 0:
        return None
    n_img = int(img_locs.numel())
    entry = _DONOR_STORE.get_or_create_donor(
        group_key=group_key,
        n_image_tokens=n_img,
        grid_sig=grid_sig,
        positions=(positions.detach().clone() if positions is not None else None),
    )
    for layer_id in layer_ids:
        k_buf = pool.get_key_buffer(layer_id)
        v_buf = pool.get_value_buffer(layer_id)
        k = k_buf[img_locs].detach().clone()
        v = v_buf[img_locs].detach().clone()
        if to_cpu:
            k = k.cpu()
            v = v.cpu()
        entry.record_layer(layer_id, k, v)
    entry.complete = True
    return entry
