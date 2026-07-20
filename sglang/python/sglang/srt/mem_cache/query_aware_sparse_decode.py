"""Query-aware block plans for low-overhead sparse autoregressive decode.

The selector deliberately does *not* use CacheBlend's reuse mask.  It scores old
context blocks with the final prompt query at one probe layer, keeps the most relevant
blocks plus protected prefix/recent windows, and persists the plan for decode.  The
selection cost is paid once during prefill; token-by-token decode only compacts once
and incrementally appends new KV slots.

This is an approximate ``SKIP`` policy.  It never claims that the omitted K/V is an
equivalent reusable artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


QUERY_BLOCK_POLICY = "query-block-mass-prefill-v2"


@dataclass(frozen=True)
class QueryBlockSelection:
    drop_positions: torch.Tensor
    kept_block_indices: torch.Tensor
    dropped_block_indices: torch.Tensor
    block_scores: torch.Tensor
    block_size: int
    candidate_start: int
    candidate_blocks: int
    kept_blocks: int
    dropped_blocks: int
    # Keep this scalar on the scoring device until the caller materializes the
    # position plan.  Calling ``.item()`` here serializes every request in a
    # prefill batch; the production path instead copies the scalar together with
    # the selected positions in one host transfer.
    dropped_score_mass_tensor: torch.Tensor

    @property
    def dropped_score_mass(self) -> float:
        return float(self.dropped_score_mass_tensor.item())

    @property
    def candidate_tokens(self) -> int:
        return self.candidate_blocks * self.block_size

    @property
    def selected_tokens(self) -> int:
        return self.kept_blocks * self.block_size


def score_query_blocks(
    query: torch.Tensor,
    representative_keys: torch.Tensor,
) -> torch.Tensor:
    """Score blocks by the strongest query/key-head landmark interaction.

    Args:
        query: ``[q_heads, head_dim]`` or
            ``[query_representatives, q_heads, head_dim]`` prompt queries after
            RoPE.
        representative_keys: ``[blocks, representatives, kv_heads, head_dim]``.

    GQA query heads are grouped by their KV head. Taking the maximum across query
    heads and landmarks is conservative: a block is retained when any head finds one
    of its landmarks important.
    """

    if query.ndim == 2:
        query = query.unsqueeze(0)
    if query.ndim != 3 or representative_keys.ndim != 4:
        raise ValueError("query must be rank 2/3 and representative_keys rank 4")
    query_representatives, q_heads, head_dim = (
        int(value) for value in query.shape
    )
    blocks, representatives, kv_heads, key_dim = (
        int(value) for value in representative_keys.shape
    )
    if blocks <= 0 or representatives <= 0 or kv_heads <= 0:
        raise ValueError("representative_keys dimensions must be positive")
    if head_dim != key_dim:
        raise ValueError("query/key head dimensions differ")
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")

    group_size = q_heads // kv_heads
    grouped_query = query.reshape(
        query_representatives, kv_heads, group_size, head_dim
    ).float()
    keys = representative_keys.float()
    # [query_landmark, kv_head, gqa, dim] x
    # [block, key_landmark, kv_head, dim]
    # -> [query_landmark, block, key_landmark, kv_head, gqa]
    scores = torch.einsum("qhgd,brhd->qbrhg", grouped_query, keys)
    return scores.amax(dim=(0, 2, 3, 4))


def select_query_aware_blocks(
    *,
    query: torch.Tensor,
    representative_keys: torch.Tensor,
    candidate_start: int,
    block_size: int,
    max_drop_ratio: float,
    max_dropped_score_mass: float,
) -> QueryBlockSelection:
    """Choose blocks under both structural and query-score-mass error budgets.

    This deliberately avoids a fixed top-k quota. Low-score blocks are removed only
    while their cumulative softmax mass remains below the configured proxy error
    budget. Ambiguous/uniform scores therefore retain almost all context, whereas a
    clear irrelevant tail can use the full structural drop allowance.
    """

    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    candidate_start = max(0, int(candidate_start))
    max_drop_ratio = min(1.0, max(0.0, float(max_drop_ratio)))
    max_dropped_score_mass = min(
        1.0, max(0.0, float(max_dropped_score_mass))
    )
    scores = score_query_blocks(query, representative_keys)
    n_blocks = int(scores.numel())

    max_drop = min(n_blocks - 1, int(n_blocks * max_drop_ratio))
    # Scaling makes the mass proxy insensitive to head dimension. This is not a
    # mathematical error bound on full-layer attention (landmarks are sampled), so it
    # remains explicitly marked approximate in the runtime contract.
    score_mass = torch.softmax(scores / (query.shape[-1] ** 0.5), dim=0)
    ranked_low = torch.argsort(scores, descending=False, stable=True)
    cumulative_mass = torch.cumsum(score_mass.index_select(0, ranked_low), dim=0)
    within_mass = cumulative_mass <= max_dropped_score_mass
    # ``within_mass`` is a monotonic prefix because cumulative_mass is monotonic.
    # Keep the quota decision on-device instead of synchronizing for ``drop_count``.
    ranked_positions = torch.arange(n_blocks, device=scores.device)
    dropped = torch.sort(
        ranked_low[within_mass & (ranked_positions < max_drop)]
    ).values
    keep_mask = torch.zeros(n_blocks, dtype=torch.bool, device=scores.device)
    keep_mask[:] = True
    keep_mask[dropped] = False
    kept = torch.nonzero(keep_mask, as_tuple=False).flatten()
    dropped_score_mass = score_mass.index_select(0, dropped).sum()
    offsets = torch.arange(block_size, dtype=torch.long, device=scores.device)
    drop_positions = (
        candidate_start + dropped[:, None] * block_size + offsets[None, :]
    ).reshape(-1)
    return QueryBlockSelection(
        drop_positions=drop_positions,
        kept_block_indices=kept,
        dropped_block_indices=dropped,
        block_scores=scores,
        block_size=block_size,
        candidate_start=candidate_start,
        candidate_blocks=n_blocks,
        kept_blocks=int(kept.numel()),
        dropped_blocks=int(dropped.numel()),
        dropped_score_mass_tensor=dropped_score_mass,
    )


def _host_int_list(value: Any, count: int) -> Optional[list[int]]:
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                return None
            values = value.reshape(-1)[:count].tolist()
        else:
            values = list(value)[:count]
        if len(values) != count:
            return None
        return [int(item) for item in values]
    except (TypeError, ValueError):
        return None


def maybe_register_query_block_plans(
    *,
    forward_batch: Any,
    q: torch.Tensor,
    key_cache: torch.Tensor,
    layer_id: int,
    page_size: int,
) -> int:
    """Build per-request plans once at the configured prefill probe layer."""

    from sglang.srt.mem_cache import vlm_cacheblend

    cfg = vlm_cacheblend.get_config()
    mode = str(cfg.sparse_decode_mode or "query_blocks").strip().lower()
    if (
        not vlm_cacheblend.sparse_decode_enabled()
        or mode not in ("query", "query_blocks")
        or int(layer_id) != max(0, int(cfg.sparse_decode_probe_layer))
        or int(page_size) != 1
        or q.ndim != 3
        or key_cache.ndim != 4
        or bool(getattr(forward_batch, "_cacheblend_compact", False))
    ):
        return 0
    forward_mode = getattr(forward_batch, "forward_mode", None)
    if forward_mode is None or not forward_mode.is_extend():
        return 0

    batch_size = int(getattr(forward_batch, "batch_size", 0) or 0)
    if batch_size <= 0:
        return 0
    seq_lens = _host_int_list(
        getattr(forward_batch, "seq_lens_cpu", None), batch_size
    )
    extend_lens = _host_int_list(
        getattr(forward_batch, "extend_seq_lens_cpu", None), batch_size
    )
    if seq_lens is None or extend_lens is None:
        return 0

    req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
    req_to_token_pool = getattr(forward_batch, "req_to_token_pool", None)
    req_to_token = getattr(req_to_token_pool, "req_to_token", None)
    if req_pool_indices is None or req_to_token is None:
        return 0
    reqs = list(getattr(forward_batch, "reqs", None) or [])
    # The scheduler already owns request-pool indices on the host.  Prefer those
    # over a device-to-host copy at the probe layer; retain the tensor fallback for
    # synthetic/unit callers and unusual scheduler paths.
    if len(reqs) >= batch_size and all(
        getattr(req, "req_pool_idx", None) is not None
        for req in reqs[:batch_size]
    ):
        req_pool_cpu = [int(req.req_pool_idx) for req in reqs[:batch_size]]
    else:
        req_pool_cpu = (
            req_pool_indices.detach().to(device="cpu").reshape(-1).tolist()
        )
    if len(req_pool_cpu) < batch_size:
        return 0

    block_size = max(1, int(cfg.sparse_decode_block_size))
    keep_first = max(0, int(cfg.sparse_decode_keep_first))
    keep_recent = max(0, int(cfg.sparse_decode_keep_recent))
    min_context = max(1, int(cfg.sparse_decode_min_context_tokens))
    representatives = max(
        1, min(block_size, int(cfg.sparse_decode_score_representatives))
    )
    query_representatives = max(1, int(cfg.sparse_decode_query_representatives))
    rep_offsets = torch.div(
        (torch.arange(representatives, device=q.device, dtype=torch.long) * 2 + 1)
        * block_size,
        2 * representatives,
        rounding_mode="floor",
    ).clamp_(max=block_size - 1)

    q_cursor = 0
    registered = 0
    for row in range(batch_size):
        extend_len = max(0, int(extend_lens[row]))
        seq_len = max(0, int(seq_lens[row]))
        q_last = q_cursor + extend_len - 1
        q_cursor += extend_len
        req_pool_idx = int(req_pool_cpu[row])
        if (
            req_pool_idx < 0
            or req_pool_idx >= int(req_to_token.shape[0])
            or seq_len > int(req_to_token.shape[1])
        ):
            continue
        # A plan is conditional on the most recent prompt query. Never let an older
        # turn's drop decision survive a failed/ambiguous replan: clear first and only
        # install a replacement after every guard and score-mass check succeeds.
        vlm_cacheblend.clear_sparse_decode_plan(req_pool_idx)
        if extend_len <= 0 or seq_len < min_context or q_last >= int(q.shape[0]):
            continue
        # A handful of queries from the tail of the new prompt is a much safer
        # proxy for decode relevance than a single assistant-marker token.  The
        # selector still runs once; decode remains free of per-token scoring.
        query_window = min(32, extend_len)
        query_count = min(query_representatives, query_window)
        if query_count == 1:
            query_offsets = torch.tensor(
                [query_window - 1], device=q.device, dtype=torch.long
            )
        else:
            query_offsets = torch.div(
                torch.arange(query_count, device=q.device, dtype=torch.long)
                * (query_window - 1),
                query_count - 1,
                rounding_mode="floor",
            )
        query_indices = q_last - query_window + 1 + query_offsets
        candidate_start = min(seq_len, keep_first)
        candidate_end = max(candidate_start, seq_len - keep_recent)
        n_blocks = (candidate_end - candidate_start) // block_size
        if n_blocks < 2:
            continue

        block_ids = torch.arange(n_blocks, device=q.device, dtype=torch.long)
        representative_positions = (
            candidate_start
            + block_ids[:, None] * block_size
            + rep_offsets[None, :]
        )
        table_row = req_to_token[req_pool_idx, :seq_len]
        physical_locs = table_row.index_select(
            0, representative_positions.reshape(-1)
        ).reshape(n_blocks, representatives)
        invalid_locs = (
            (physical_locs < 0) | (physical_locs >= int(key_cache.shape[0]))
        ).any()
        # Avoid a request-by-request validity synchronization.  Clamp only to make
        # the speculative gather memory-safe; the packed host result below still
        # fails closed and refuses to register a plan if any location was invalid.
        safe_physical_locs = physical_locs.clamp(
            min=0, max=max(0, int(key_cache.shape[0]) - 1)
        )
        representative_keys = key_cache[safe_physical_locs, 0]
        try:
            selection = select_query_aware_blocks(
                query=q.index_select(0, query_indices),
                representative_keys=representative_keys,
                candidate_start=candidate_start,
                block_size=block_size,
                max_drop_ratio=float(cfg.sparse_decode_max_drop_ratio),
                max_dropped_score_mass=float(
                    cfg.sparse_decode_max_dropped_score_mass
                ),
            )
        except (RuntimeError, ValueError):
            continue
        if selection.dropped_blocks <= 0:
            continue

        request_id = str(row)
        if row < len(reqs):
            request_id = str(getattr(reqs[row], "rid", request_id) or request_id)
        # One transfer replaces the former validity ``.item()``, mass ``.item()``,
        # and positions copy.  Float64 exactly represents every supported context
        # position and carries the scalar metadata without a second synchronization.
        host_plan = torch.cat(
            (
                invalid_locs.reshape(1).to(dtype=torch.float64),
                selection.dropped_score_mass_tensor.reshape(1).to(
                    dtype=torch.float64
                ),
                selection.drop_positions.to(dtype=torch.float64),
            )
        ).to(device="cpu")
        if bool(host_plan[0].item()):
            continue
        drop_positions_cpu = host_plan[2:].to(dtype=torch.long)
        plan = vlm_cacheblend.register_sparse_decode_position_plan(
            request_id=request_id,
            req_pool_idx=req_pool_idx,
            req_to_token_row=table_row,
            drop_positions=drop_positions_cpu,
            mode="query_blocks",
            keep_recent=keep_recent,
            keep_first=keep_first,
            candidate_tokens=selection.candidate_tokens,
            selected_tokens=selection.selected_tokens,
            decision_reason="query_ranked_context_blocks",
            policy=QUERY_BLOCK_POLICY,
            error_bound=float(host_plan[1].item()),
            positions_are_unique_sorted=True,
        )
        registered += int(plan is not None)
    return registered
