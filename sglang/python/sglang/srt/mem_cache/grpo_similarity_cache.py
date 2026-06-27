"""GRPO group-level turn1 image-token similarity cache for Qwen2.5-VL rollouts.

Targets cross-branch reuse within the same GRPO group (agent_uid): at turn1 each
branch encodes two images (turn0 chart + refocus). Image slot 0 is byte-identical
across branches; image slot 1 (refocus) is similar-but-not-identical.

Enable with SGLANG_GRPO_SIM_CACHE=1. Reuse mode is explicit:
  - whole_slot_reuse: slot-level exact/similarity skip, reusing the donor's whole
    final image embedding for a slot;
  - token_or_window_partial_reuse: cross-branch partial ViT reuse for
    non-identical refocus slots. With window granularity, similar windows reuse
    donor pre-full-attention hidden states. With token granularity, similar
    patch tokens reuse donor hidden states as a probe path. With merged_window
    granularity, stable merged-token proxy masks are mapped back to whole ViT
    windows for pre-full-attention skip.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_GRPO_SIM_CACHE_ENABLED = os.environ.get("SGLANG_GRPO_SIM_CACHE", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_GRPO_SIM_RAW_COSINE_THRESH = float(os.environ.get("SGLANG_GRPO_SIM_RAW_COSINE_THRESH", "0.995"))
_GRPO_SIM_RAW_COSINE_RATIO = float(os.environ.get("SGLANG_GRPO_SIM_RAW_COSINE_RATIO", "0.90"))
_GRPO_SIM_PARTIAL_VIT_REUSE_ENV = os.environ.get("SGLANG_GRPO_ENABLE_PARTIAL_VIT_REUSE", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_GRPO_SIM_PARTIAL_REUSE_THRESHOLD = float(os.environ.get("SGLANG_GRPO_PARTIAL_REUSE_THRESHOLD", "0.98"))
_GRPO_SIM_PARTIAL_REUSE_GRANULARITY = os.environ.get(
    "SGLANG_GRPO_PARTIAL_REUSE_GRANULARITY", "window"
).lower()
_GRPO_SIM_REUSE_MODE_RAW = os.environ.get("SGLANG_GRPO_REUSE_MODE", "").strip().lower()
_GRPO_SIM_TARGET_TURNS_RAW = os.environ.get("SGLANG_GRPO_SIM_TARGET_TURNS", "1").strip().lower()
_GRPO_SIM_MAX_GROUPS = int(os.environ.get("SGLANG_GRPO_SIM_MAX_GROUPS", "512"))
_GRPO_SIM_MAX_REQUEST_META = int(os.environ.get("SGLANG_GRPO_SIM_MAX_REQUEST_META", "65536"))

_RID_TURN_RE = re.compile(r"_t(\d+)$")

_LOCK = threading.Lock()
_REQUEST_META: Dict[str, Dict[str, Any]] = {}
_GROUP_CACHE: Dict[Tuple, "_GroupCacheEntry"] = {}
_LAST_GLOBAL_STEP: Optional[int] = None
_STATS = {
    "vit_calls": 0,
    "vit_skipped": 0,
    "slot0_skipped": 0,
    "slot1_skipped": 0,
    "exact_skipped": 0,
    "similarity_skipped": 0,
    "no_meta": 0,
    "no_step": 0,
}


def _normalize_reuse_mode(raw: str) -> str:
    if not raw:
        return "token_or_window_partial_reuse" if _GRPO_SIM_PARTIAL_VIT_REUSE_ENV else "whole_slot_reuse"
    aliases = {
        "whole": "whole_slot_reuse",
        "slot": "whole_slot_reuse",
        "whole_slot": "whole_slot_reuse",
        "whole_slot_reuse": "whole_slot_reuse",
        "partial": "token_or_window_partial_reuse",
        "partial_window": "token_or_window_partial_reuse",
        "merged_window": "token_or_window_partial_reuse",
        "merged-window": "token_or_window_partial_reuse",
        "window": "token_or_window_partial_reuse",
        "token": "token_or_window_partial_reuse",
        "merged": "token_or_window_partial_reuse",
        "token_window": "token_or_window_partial_reuse",
        "token_or_window_partial_reuse": "token_or_window_partial_reuse",
    }
    return aliases.get(raw, "whole_slot_reuse")


_GRPO_SIM_REUSE_MODE = _normalize_reuse_mode(_GRPO_SIM_REUSE_MODE_RAW)
_GRPO_SIM_PARTIAL_VIT_REUSE = (
    _GRPO_SIM_PARTIAL_VIT_REUSE_ENV or _GRPO_SIM_REUSE_MODE == "token_or_window_partial_reuse"
)


@dataclass
class _GroupCacheEntry:
    pixel_values: torch.Tensor
    embedding: torch.Tensor
    grid_sig: Tuple[int, int, int]
    donor_rid: str
    partial_cache: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class GrpoEncodeStats:
    vit_calls: int = 0
    vit_skipped: int = 0
    slot0_skipped: int = 0
    slot1_skipped: int = 0
    agent_uid: str = ""
    agent_turn: int = -1
    global_step: int = -1
    request_id: str = ""
    cache_hits: int = 0
    cache_misses: int = 0
    slot0_hits: int = 0
    slot0_misses: int = 0
    slot1_hits: int = 0
    slot1_misses: int = 0
    exact_reuse: int = 0
    cross_slot_exact_reuse: int = 0
    similarity_reuse: int = 0
    similarity_checked: int = 0
    slot0_identical_ratio_min: float = -1.0
    slot0_identical_ratio_max: float = -1.0
    slot1_identical_ratio_min: float = -1.0
    slot1_identical_ratio_max: float = -1.0
    raw_cosine_mean_min: float = -1.0
    raw_cosine_mean_max: float = -1.0
    raw_cosine_ratio_min: float = -1.0
    raw_cosine_ratio_max: float = -1.0
    partial_vit_enabled: int = 0
    partial_vit_used: int = 0
    partial_vit_total_windows: int = 0
    partial_vit_reused_windows: int = 0
    partial_vit_computed_windows: int = 0
    partial_vit_reuse_ratio: float = -1.0
    partial_vit_window_cosine_min: float = -1.0
    partial_vit_window_cosine_mean: float = -1.0
    partial_vit_window_cosine_max: float = -1.0
    partial_vit_total_window_layers: int = 0
    partial_vit_reused_window_layer_windows: int = 0
    partial_vit_computed_window_layer_windows: int = 0
    partial_vit_total_tokens: int = 0
    partial_vit_reused_tokens: int = 0
    partial_vit_computed_tokens: int = 0
    partial_vit_token_reuse_ratio: float = -1.0
    partial_vit_token_cosine_min: float = -1.0
    partial_vit_token_cosine_mean: float = -1.0
    partial_vit_token_cosine_max: float = -1.0
    partial_vit_total_token_layers: int = 0
    partial_vit_reused_token_layer_tokens: int = 0
    partial_vit_computed_token_layer_tokens: int = 0
    partial_vit_merged_total_tokens: int = 0
    partial_vit_merged_reused_tokens: int = 0
    partial_vit_merged_token_reuse_ratio: float = -1.0
    partial_vit_merged_token_cosine_min: float = -1.0
    partial_vit_merged_token_cosine_mean: float = -1.0
    partial_vit_merged_token_cosine_max: float = -1.0
    partial_vit_fallback_reason: str = ""

    def to_log_fields(self) -> Dict[str, str]:
        return {
            "grpo_sim_cache_enabled": "1" if _GRPO_SIM_CACHE_ENABLED else "0",
            "grpo_sim_policy": _GRPO_SIM_REUSE_MODE,
            "grpo_sim_reuse_mode": _GRPO_SIM_REUSE_MODE,
            "grpo_sim_target_turns": _GRPO_SIM_TARGET_TURNS_RAW,
            "grpo_sim_agent_uid": self.agent_uid or "",
            "grpo_sim_agent_turn": str(self.agent_turn),
            "grpo_sim_global_step": str(self.global_step),
            "grpo_sim_vit_calls": str(self.vit_calls),
            "grpo_sim_vit_skipped": str(self.vit_skipped),
            "grpo_sim_slot0_skipped": str(self.slot0_skipped),
            "grpo_sim_slot1_skipped": str(self.slot1_skipped),
            "grpo_sim_cache_hits": str(self.cache_hits),
            "grpo_sim_cache_misses": str(self.cache_misses),
            "grpo_sim_slot0_hits": str(self.slot0_hits),
            "grpo_sim_slot0_misses": str(self.slot0_misses),
            "grpo_sim_slot1_hits": str(self.slot1_hits),
            "grpo_sim_slot1_misses": str(self.slot1_misses),
            "grpo_sim_exact_reuse": str(self.exact_reuse),
            "grpo_sim_cross_slot_exact_reuse": str(self.cross_slot_exact_reuse),
            "grpo_sim_similarity_reuse": str(self.similarity_reuse),
            "grpo_sim_similarity_checked": str(self.similarity_checked),
            "grpo_sim_slot0_identical_ratio_min": (
                "" if self.slot0_identical_ratio_min < 0 else f"{self.slot0_identical_ratio_min:.6f}"
            ),
            "grpo_sim_slot0_identical_ratio_max": (
                "" if self.slot0_identical_ratio_max < 0 else f"{self.slot0_identical_ratio_max:.6f}"
            ),
            "grpo_sim_slot1_identical_ratio_min": (
                "" if self.slot1_identical_ratio_min < 0 else f"{self.slot1_identical_ratio_min:.6f}"
            ),
            "grpo_sim_slot1_identical_ratio_max": (
                "" if self.slot1_identical_ratio_max < 0 else f"{self.slot1_identical_ratio_max:.6f}"
            ),
            "grpo_sim_raw_cosine_mean_min": (
                "" if self.raw_cosine_mean_min < 0 else f"{self.raw_cosine_mean_min:.6f}"
            ),
            "grpo_sim_raw_cosine_mean_max": (
                "" if self.raw_cosine_mean_max < 0 else f"{self.raw_cosine_mean_max:.6f}"
            ),
            "grpo_sim_raw_cosine_ratio_min": (
                "" if self.raw_cosine_ratio_min < 0 else f"{self.raw_cosine_ratio_min:.6f}"
            ),
            "grpo_sim_raw_cosine_ratio_max": (
                "" if self.raw_cosine_ratio_max < 0 else f"{self.raw_cosine_ratio_max:.6f}"
            ),
            "enable_partial_vit_reuse": "1" if _GRPO_SIM_PARTIAL_VIT_REUSE else "0",
            "partial_reuse_threshold": f"{_GRPO_SIM_PARTIAL_REUSE_THRESHOLD:.6f}",
            "partial_reuse_granularity": _GRPO_SIM_PARTIAL_REUSE_GRANULARITY,
            "partial_vit_used": str(self.partial_vit_used),
            "total_windows": str(self.partial_vit_total_windows),
            "reused_windows": str(self.partial_vit_reused_windows),
            "computed_windows": str(self.partial_vit_computed_windows),
            "reuse_ratio": "" if self.partial_vit_reuse_ratio < 0 else f"{self.partial_vit_reuse_ratio:.6f}",
            "partial_vit_window_cosine_min": (
                "" if self.partial_vit_window_cosine_min < 0 else f"{self.partial_vit_window_cosine_min:.6f}"
            ),
            "partial_vit_window_cosine_mean": (
                "" if self.partial_vit_window_cosine_mean < 0 else f"{self.partial_vit_window_cosine_mean:.6f}"
            ),
            "partial_vit_window_cosine_max": (
                "" if self.partial_vit_window_cosine_max < 0 else f"{self.partial_vit_window_cosine_max:.6f}"
            ),
            "partial_vit_total_window_layers": str(self.partial_vit_total_window_layers),
            "partial_vit_reused_window_layer_windows": str(self.partial_vit_reused_window_layer_windows),
            "partial_vit_computed_window_layer_windows": str(self.partial_vit_computed_window_layer_windows),
            "total_tokens": str(self.partial_vit_total_tokens),
            "reused_tokens": str(self.partial_vit_reused_tokens),
            "computed_tokens": str(self.partial_vit_computed_tokens),
            "token_reuse_ratio": (
                "" if self.partial_vit_token_reuse_ratio < 0 else f"{self.partial_vit_token_reuse_ratio:.6f}"
            ),
            "partial_vit_token_cosine_min": (
                "" if self.partial_vit_token_cosine_min < 0 else f"{self.partial_vit_token_cosine_min:.6f}"
            ),
            "partial_vit_token_cosine_mean": (
                "" if self.partial_vit_token_cosine_mean < 0 else f"{self.partial_vit_token_cosine_mean:.6f}"
            ),
            "partial_vit_token_cosine_max": (
                "" if self.partial_vit_token_cosine_max < 0 else f"{self.partial_vit_token_cosine_max:.6f}"
            ),
            "partial_vit_total_token_layers": str(self.partial_vit_total_token_layers),
            "partial_vit_reused_token_layer_tokens": str(self.partial_vit_reused_token_layer_tokens),
            "partial_vit_computed_token_layer_tokens": str(self.partial_vit_computed_token_layer_tokens),
            "merged_total_tokens": str(self.partial_vit_merged_total_tokens),
            "merged_reused_tokens": str(self.partial_vit_merged_reused_tokens),
            "merged_token_reuse_ratio": (
                ""
                if self.partial_vit_merged_token_reuse_ratio < 0
                else f"{self.partial_vit_merged_token_reuse_ratio:.6f}"
            ),
            "partial_vit_merged_token_cosine_min": (
                ""
                if self.partial_vit_merged_token_cosine_min < 0
                else f"{self.partial_vit_merged_token_cosine_min:.6f}"
            ),
            "partial_vit_merged_token_cosine_mean": (
                ""
                if self.partial_vit_merged_token_cosine_mean < 0
                else f"{self.partial_vit_merged_token_cosine_mean:.6f}"
            ),
            "partial_vit_merged_token_cosine_max": (
                ""
                if self.partial_vit_merged_token_cosine_max < 0
                else f"{self.partial_vit_merged_token_cosine_max:.6f}"
            ),
            "fallback_reason": self.partial_vit_fallback_reason,
        }


def grpo_sim_cache_enabled() -> bool:
    return _GRPO_SIM_CACHE_ENABLED


def grpo_partial_vit_reuse_enabled() -> bool:
    return (
        _GRPO_SIM_PARTIAL_VIT_REUSE
        and _GRPO_SIM_REUSE_MODE == "token_or_window_partial_reuse"
        and _GRPO_SIM_PARTIAL_REUSE_GRANULARITY
        in ("window", "token", "token_sparse", "merged", "merged_window", "merged-window")
    )


def grpo_whole_slot_reuse_enabled() -> bool:
    return _GRPO_SIM_REUSE_MODE == "whole_slot_reuse"


def _target_turn_enabled(turn: int) -> bool:
    raw = _GRPO_SIM_TARGET_TURNS_RAW
    if raw in ("*", "all", "any"):
        return True
    if raw in (">=1", "post0", "after0"):
        return turn >= 1
    try:
        return turn in {int(part.strip()) for part in raw.split(",") if part.strip()}
    except ValueError:
        return turn == 1


def register_request_meta(
    rid: str,
    *,
    agent_uid: Optional[str] = None,
    agent_turn: Optional[int] = None,
    agent_request_id: Optional[str] = None,
    global_step: Optional[int] = None,
) -> None:
    """Called from verl rollout before SGLang generate (same process)."""
    if not rid:
        return
    meta: Dict[str, Any] = {}
    if agent_uid:
        meta["agent_uid"] = str(agent_uid)
    if agent_turn is not None:
        meta["agent_turn"] = int(agent_turn)
    if agent_request_id:
        meta["agent_request_id"] = str(agent_request_id)
    if global_step is not None:
        meta["global_step"] = int(global_step)
    if not meta:
        return
    meta["_created_at"] = time.time()
    with _LOCK:
        _rotate_step_locked(meta.get("global_step"))
        _REQUEST_META[str(rid)] = meta
        _evict_request_meta_locked()


def lookup_request_meta(rid: Optional[str]) -> Optional[Dict[str, Any]]:
    if not rid:
        return None
    with _LOCK:
        return dict(_REQUEST_META.get(str(rid), {}) or {})


def parse_turn_from_rid(rid: str) -> Optional[int]:
    m = _RID_TURN_RE.search(str(rid))
    return int(m.group(1)) if m else None


def resolve_agent_meta_for_items(items, hash_to_rid: Dict[Any, str]) -> Optional[Dict[str, Any]]:
    if not items:
        return None
    rid = None
    for it in items:
        h = getattr(it, "hash", None)
        if h is not None and h in hash_to_rid:
            rid = hash_to_rid[h]
            break
    if rid is None:
        return None
    meta = lookup_request_meta(rid) or {}
    turn = meta.get("agent_turn")
    if turn is None:
        turn = parse_turn_from_rid(rid)
    uid = meta.get("agent_uid") or ""
    if not uid:
        return None
    return {
        "agent_uid": uid,
        "agent_turn": int(turn) if turn is not None else -1,
        "global_step": int(meta.get("global_step", -1)),
        "request_id": str(rid),
        "agent_request_id": meta.get("agent_request_id", ""),
    }


def split_images_by_grid(
    pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Split concatenated patch rows into per-image (patches, grid_row) segments."""
    if image_grid_thw.numel() == 0:
        return [(pixel_values, image_grid_thw)]
    grid = image_grid_thw.reshape(-1, 3)
    if grid.shape[0] <= 1:
        return [(pixel_values, grid if grid.dim() == 2 else image_grid_thw)]
    patch_counts = (grid[:, 0] * grid[:, 1] * grid[:, 2]).tolist()
    segments: List[Tuple[torch.Tensor, torch.Tensor]] = []
    start = 0
    for cnt in patch_counts:
        n = int(cnt)
        end = start + n
        segments.append((pixel_values[start:end], grid[len(segments) : len(segments) + 1]))
        start = end
    if start != pixel_values.shape[0]:
        # Fallback: treat as single image if patch arithmetic does not align.
        return [(pixel_values, grid)]
    return segments


def _grid_sig(grid_row: torch.Tensor) -> Tuple[int, int, int]:
    row = grid_row.reshape(-1, 3)[0].tolist()
    return int(row[0]), int(row[1]), int(row[2])


def _cache_key_with_step(
    global_step: int,
    agent_uid: str,
    agent_turn: int,
    image_slot: int,
    grid_sig: Tuple[int, int, int],
):
    return (global_step, agent_uid, agent_turn, image_slot, grid_sig)


def _patch_identical_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return 0.0
    if a.numel() == 0:
        return 1.0
    same = (a == b).all(dim=-1)
    return float(same.float().mean().item())


def _patch_cosine_stats(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    if a.shape != b.shape:
        return -1.0, 0.0
    if a.numel() == 0:
        return 1.0, 1.0
    cos = F.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-6)
    return float(cos.mean().item()), float((cos >= _GRPO_SIM_RAW_COSINE_THRESH).float().mean().item())


def _should_skip_with_donor(
    *,
    image_slot: int,
    donor: _GroupCacheEntry,
    target_patches: torch.Tensor,
) -> Tuple[bool, str, Dict[str, float]]:
    identical_ratio = _patch_identical_ratio(target_patches, donor.pixel_values)
    if identical_ratio >= 0.999:
        return True, "exact", {"identical_ratio": identical_ratio}
    if image_slot == 0:
        return False, "not_exact", {"identical_ratio": identical_ratio}

    mean_cos, high_ratio = _patch_cosine_stats(target_patches, donor.pixel_values)
    ok = mean_cos >= _GRPO_SIM_RAW_COSINE_THRESH and high_ratio >= _GRPO_SIM_RAW_COSINE_RATIO
    return ok, "similarity", {
        "identical_ratio": identical_ratio,
        "raw_cosine_mean": mean_cos,
        "raw_cosine_ratio": high_ratio,
    }


def _update_min_max(current_min: float, current_max: float, val: float) -> Tuple[float, float]:
    if val < 0:
        return current_min, current_max
    if current_min < 0:
        return val, val
    return min(current_min, val), max(current_max, val)


def _rotate_step_locked(global_step: Optional[int]) -> None:
    global _LAST_GLOBAL_STEP
    if global_step is None:
        return
    step = int(global_step)
    if step < 0:
        return
    if _LAST_GLOBAL_STEP is None:
        _LAST_GLOBAL_STEP = step
        return
    if step != _LAST_GLOBAL_STEP:
        _GROUP_CACHE.clear()
        _REQUEST_META.clear()
        _LAST_GLOBAL_STEP = step


def _evict_request_meta_locked() -> None:
    while len(_REQUEST_META) > _GRPO_SIM_MAX_REQUEST_META:
        oldest_key = min(
            _REQUEST_META.items(),
            key=lambda kv: float(kv[1].get("_created_at", 0.0)),
        )[0]
        _REQUEST_META.pop(oldest_key, None)


def _evict_if_needed() -> None:
    while len(_GROUP_CACHE) > _GRPO_SIM_MAX_GROUPS:
        oldest_key = min(_GROUP_CACHE.items(), key=lambda kv: kv[1].created_at)[0]
        _GROUP_CACHE.pop(oldest_key, None)


def _get_donor(key: Tuple) -> Optional[_GroupCacheEntry]:
    with _LOCK:
        return _GROUP_CACHE.get(key)


def _find_exact_donor_any_slot(
    *,
    global_step: int,
    agent_uid: str,
    agent_turn: int,
    image_slot: int,
    grid_sig: Tuple[int, int, int],
    target_patches: torch.Tensor,
) -> Tuple[Optional[_GroupCacheEntry], Optional[int], float]:
    """Find an exact donor in any slot of this GRPO group.

    SGLang packs all images of the same modality into one item. In some
    multi-turn paths, logical image order and feature segment order are not
    stable enough to rely on image_slot for exact reuse. Exact patch equality is
    content-safe, so we allow exact reuse across slots while keeping similarity
    reuse same-slot only.
    """
    prefix = (global_step, agent_uid, agent_turn)
    with _LOCK:
        candidates = [
            (key[3], entry)
            for key, entry in _GROUP_CACHE.items()
            if len(key) == 5
            and key[0:3] == prefix
            and key[3] != image_slot
            and key[4] == grid_sig
        ]
    for donor_slot, entry in candidates:
        identical_ratio = _patch_identical_ratio(target_patches, entry.pixel_values)
        if identical_ratio >= 0.999:
            return entry, int(donor_slot), identical_ratio
    return None, None, -1.0


def _put_donor_if_absent(
    key: Tuple,
    *,
    pixel_values: torch.Tensor,
    embedding: torch.Tensor,
    grid_sig: Tuple[int, int, int],
    donor_rid: str,
    partial_cache: Optional[Dict[str, Any]] = None,
) -> bool:
    stored_embedding = embedding.detach()
    with _LOCK:
        if key in _GROUP_CACHE:
            return False
        _GROUP_CACHE[key] = _GroupCacheEntry(
            pixel_values=pixel_values.detach().cpu(),
            embedding=stored_embedding,
            grid_sig=grid_sig,
            donor_rid=donor_rid,
            partial_cache=partial_cache,
        )
        _evict_if_needed()
    return True


def _merge_partial_stats(stats: GrpoEncodeStats, partial_stats: Optional[Dict[str, Any]]) -> None:
    if not partial_stats:
        return
    stats.partial_vit_enabled = 1
    stats.partial_vit_used += int(bool(partial_stats.get("used", False)))
    stats.partial_vit_total_windows += int(partial_stats.get("total_windows", 0) or 0)
    stats.partial_vit_reused_windows += int(partial_stats.get("reused_windows", 0) or 0)
    stats.partial_vit_computed_windows += int(partial_stats.get("computed_windows", 0) or 0)
    stats.partial_vit_total_window_layers += int(partial_stats.get("total_window_layers", 0) or 0)
    stats.partial_vit_reused_window_layer_windows += int(
        partial_stats.get("reused_window_layer_windows", 0) or 0
    )
    stats.partial_vit_computed_window_layer_windows += int(
        partial_stats.get("computed_window_layer_windows", 0) or 0
    )
    stats.partial_vit_total_tokens += int(partial_stats.get("total_tokens", 0) or 0)
    stats.partial_vit_reused_tokens += int(partial_stats.get("reused_tokens", 0) or 0)
    stats.partial_vit_computed_tokens += int(partial_stats.get("computed_tokens", 0) or 0)
    stats.partial_vit_total_token_layers += int(partial_stats.get("total_token_layers", 0) or 0)
    stats.partial_vit_reused_token_layer_tokens += int(
        partial_stats.get("reused_token_layer_tokens", 0) or 0
    )
    stats.partial_vit_computed_token_layer_tokens += int(
        partial_stats.get("computed_token_layer_tokens", 0) or 0
    )
    stats.partial_vit_merged_total_tokens += int(partial_stats.get("merged_total_tokens", 0) or 0)
    stats.partial_vit_merged_reused_tokens += int(partial_stats.get("merged_reused_tokens", 0) or 0)
    total_windows = stats.partial_vit_total_windows
    if total_windows > 0:
        stats.partial_vit_reuse_ratio = stats.partial_vit_reused_windows / total_windows
    total_tokens = stats.partial_vit_total_tokens
    if total_tokens > 0:
        stats.partial_vit_token_reuse_ratio = stats.partial_vit_reused_tokens / total_tokens
    merged_total_tokens = stats.partial_vit_merged_total_tokens
    if merged_total_tokens > 0:
        stats.partial_vit_merged_token_reuse_ratio = (
            stats.partial_vit_merged_reused_tokens / merged_total_tokens
        )
    for dst, key in (
        ("partial_vit_window_cosine_min", "window_cosine_min"),
        ("partial_vit_window_cosine_mean", "window_cosine_mean"),
        ("partial_vit_window_cosine_max", "window_cosine_max"),
        ("partial_vit_token_cosine_min", "token_cosine_min"),
        ("partial_vit_token_cosine_mean", "token_cosine_mean"),
        ("partial_vit_token_cosine_max", "token_cosine_max"),
        ("partial_vit_merged_token_cosine_min", "merged_token_cosine_min"),
        ("partial_vit_merged_token_cosine_mean", "merged_token_cosine_mean"),
        ("partial_vit_merged_token_cosine_max", "merged_token_cosine_max"),
    ):
        val = partial_stats.get(key)
        if val is None:
            continue
        try:
            val = float(val)
        except Exception:
            continue
        old = getattr(stats, dst)
        if old < 0:
            setattr(stats, dst, val)
        elif dst.endswith("_min"):
            setattr(stats, dst, min(old, val))
        elif dst.endswith("_max"):
            setattr(stats, dst, max(old, val))
        else:
            setattr(stats, dst, val)
    reason = str(partial_stats.get("fallback_reason", "") or "")
    if reason:
        if stats.partial_vit_fallback_reason:
            stats.partial_vit_fallback_reason += f";{reason}"
        else:
            stats.partial_vit_fallback_reason = reason


def encode_with_grpo_similarity_cache(
    *,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    items: list,
    hash_to_rid: Dict[Any, str],
    encode_single_image_fn,
    encode_partial_image_fn=None,
    output_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, GrpoEncodeStats, bool]:
    """Returns (embeddings, stats, used_cache_path).

    ViT pipeline stage map (Qwen2_5_VisionTransformer in qwen2_5_vl.py):
      §1 patch_embed          — partial: patch_hidden cosine 判定
      §2 prefull ViT blocks   — partial: window/token/token_sparse 省算
      §3 fullatt ViT blocks   — 不复用，全量计算
      §4/§5 merger            — granularity=merged 时输出层替换
      §6 LLM                  — 不在此函数，在 get_image_feature 返回后由 LLM forward 消费

    whole_slot_reuse 跳过 §1–§5，直接复用 donor 的最终 embedding。
    """
    stats = GrpoEncodeStats()
    meta = resolve_agent_meta_for_items(items, hash_to_rid)
    if meta is None:
        with _LOCK:
            _STATS["no_meta"] += 1
        return encode_single_image_fn(pixel_values, image_grid_thw), stats, False

    stats.agent_uid = meta["agent_uid"]
    stats.agent_turn = meta["agent_turn"]
    stats.global_step = meta.get("global_step", -1)
    stats.request_id = meta["request_id"]

    if not _target_turn_enabled(meta["agent_turn"]):
        return encode_single_image_fn(pixel_values, image_grid_thw), stats, False
    if stats.global_step < 0:
        with _LOCK:
            _STATS["no_step"] += 1
        return encode_single_image_fn(pixel_values, image_grid_thw), stats, False

    segments = split_images_by_grid(pixel_values, image_grid_thw)
    out_parts: List[torch.Tensor] = []
    device = output_device or pixel_values.device
    dtype = pixel_values.dtype

    for image_slot, (pv_seg, grid_row) in enumerate(segments):
        gs = _grid_sig(grid_row)
        key = _cache_key_with_step(
            stats.global_step,
            meta["agent_uid"],
            meta["agent_turn"],
            image_slot,
            gs,
        )
        same_slot_donor = _get_donor(key)
        donor = same_slot_donor

        skip_vit = False
        skip_reason = ""
        partial_attempted = False
        decision_metrics: Dict[str, float] = {}
        if same_slot_donor is not None:
            stats.cache_hits += 1
            if image_slot == 0:
                stats.slot0_hits += 1
            else:
                stats.slot1_hits += 1
            skip_vit, skip_reason, decision_metrics = _should_skip_with_donor(
                image_slot=image_slot,
                donor=same_slot_donor,
                target_patches=pv_seg.detach().cpu(),
            )
            identical_ratio = decision_metrics.get("identical_ratio", -1.0)
            if image_slot == 0:
                stats.slot0_identical_ratio_min, stats.slot0_identical_ratio_max = _update_min_max(
                    stats.slot0_identical_ratio_min,
                    stats.slot0_identical_ratio_max,
                    identical_ratio,
                )
            else:
                stats.slot1_identical_ratio_min, stats.slot1_identical_ratio_max = _update_min_max(
                    stats.slot1_identical_ratio_min,
                    stats.slot1_identical_ratio_max,
                    identical_ratio,
                )
            if skip_reason == "similarity":
                stats.similarity_checked += 1
                mean_cos = decision_metrics.get("raw_cosine_mean", -1.0)
                high_ratio = decision_metrics.get("raw_cosine_ratio", -1.0)
                stats.raw_cosine_mean_min, stats.raw_cosine_mean_max = _update_min_max(
                    stats.raw_cosine_mean_min,
                    stats.raw_cosine_mean_max,
                    mean_cos,
                )
                stats.raw_cosine_ratio_min, stats.raw_cosine_ratio_max = _update_min_max(
                    stats.raw_cosine_ratio_min,
                    stats.raw_cosine_ratio_max,
                    high_ratio,
                )
                if grpo_partial_vit_reuse_enabled() and encode_partial_image_fn is not None:
                    skip_vit = False
                    skip_reason = ""
                    partial_attempted = True

        if not skip_vit and not partial_attempted:
            cross_donor, _, cross_identical_ratio = _find_exact_donor_any_slot(
                global_step=stats.global_step,
                agent_uid=meta["agent_uid"],
                agent_turn=meta["agent_turn"],
                image_slot=image_slot,
                grid_sig=gs,
                target_patches=pv_seg.detach().cpu(),
            )
            if cross_donor is not None:
                if same_slot_donor is None:
                    stats.cache_hits += 1
                    if image_slot == 0:
                        stats.slot0_hits += 1
                    else:
                        stats.slot1_hits += 1
                donor = cross_donor
                skip_vit = True
                skip_reason = "cross_slot_exact"
                decision_metrics = {"identical_ratio": cross_identical_ratio}

        if same_slot_donor is None and not skip_vit:
            stats.cache_misses += 1
            if image_slot == 0:
                stats.slot0_misses += 1
            else:
                stats.slot1_misses += 1

        if partial_attempted and donor is not None and encode_partial_image_fn is not None:
            emb, partial_cache, partial_stats = encode_partial_image_fn(
                pv_seg,
                grid_row,
                donor_pixel_values=donor.pixel_values,
                donor_partial_cache=donor.partial_cache,
                donor_embedding=donor.embedding,
                threshold=_GRPO_SIM_PARTIAL_REUSE_THRESHOLD,
                granularity=_GRPO_SIM_PARTIAL_REUSE_GRANULARITY,
                capture_cache=False,
            )
            _merge_partial_stats(stats, partial_stats)
            stats.vit_calls += 1
            with _LOCK:
                _STATS["vit_calls"] += 1
            if same_slot_donor is None:
                _put_donor_if_absent(
                    key,
                    pixel_values=pv_seg.detach().cpu(),
                    embedding=emb,
                    grid_sig=gs,
                    donor_rid=meta["request_id"],
                    partial_cache=partial_cache,
                )
        elif skip_vit and donor is not None:
            emb = donor.embedding.to(device=device, dtype=dtype)
            stats.vit_skipped += 1
            if skip_reason in ("exact", "cross_slot_exact"):
                stats.exact_reuse += 1
                if skip_reason == "cross_slot_exact":
                    stats.cross_slot_exact_reuse += 1
            elif skip_reason == "similarity":
                stats.similarity_reuse += 1
            if image_slot == 0:
                stats.slot0_skipped += 1
            else:
                stats.slot1_skipped += 1
            with _LOCK:
                _STATS["vit_skipped"] += 1
                if skip_reason == "exact":
                    _STATS["exact_skipped"] += 1
                elif skip_reason == "cross_slot_exact":
                    _STATS["exact_skipped"] += 1
                elif skip_reason == "similarity":
                    _STATS["similarity_skipped"] += 1
                if image_slot == 0:
                    _STATS["slot0_skipped"] += 1
                else:
                    _STATS["slot1_skipped"] += 1
            if same_slot_donor is None:
                if not (grpo_partial_vit_reuse_enabled() and skip_reason == "cross_slot_exact"):
                    _put_donor_if_absent(
                        key,
                        pixel_values=pv_seg.detach().cpu(),
                        embedding=emb,
                        grid_sig=gs,
                        donor_rid=meta["request_id"],
                        partial_cache=donor.partial_cache,
                    )
        else:
            partial_cache = None
            if grpo_partial_vit_reuse_enabled() and encode_partial_image_fn is not None:
                emb, partial_cache, partial_stats = encode_partial_image_fn(
                    pv_seg,
                    grid_row,
                    donor_pixel_values=None,
                    donor_partial_cache=None,
                    threshold=_GRPO_SIM_PARTIAL_REUSE_THRESHOLD,
                    granularity=_GRPO_SIM_PARTIAL_REUSE_GRANULARITY,
                    capture_cache=True,
                )
                _merge_partial_stats(stats, partial_stats)
            else:
                emb = encode_single_image_fn(pv_seg, grid_row)
            stats.vit_calls += 1
            with _LOCK:
                _STATS["vit_calls"] += 1
            if same_slot_donor is None:
                _put_donor_if_absent(
                    key,
                    pixel_values=pv_seg.detach().cpu(),
                        embedding=emb,
                        grid_sig=gs,
                        donor_rid=meta["request_id"],
                        partial_cache=partial_cache,
                    )

        out_parts.append(emb)

    if len(out_parts) == 1:
        return out_parts[0], stats, True
    return torch.cat(out_parts, dim=0), stats, True


def global_stats_snapshot() -> Dict[str, int]:
    with _LOCK:
        return dict(_STATS)


def clear_caches() -> None:
    global _LAST_GLOBAL_STEP
    with _LOCK:
        _GROUP_CACHE.clear()
        _REQUEST_META.clear()
        _LAST_GLOBAL_STEP = None
