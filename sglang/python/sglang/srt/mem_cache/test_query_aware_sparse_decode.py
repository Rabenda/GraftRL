from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache import vlm_cacheblend as cb
from sglang.srt.mem_cache.query_aware_sparse_decode import (
    maybe_register_query_block_plans,
    score_query_blocks,
    select_query_aware_blocks,
)
from sglang.srt.mem_cache.sparse_decode_kernels import (
    MAX_FUSED_SPARSE_CONTEXT_WIDTH,
    sparse_decode_warmup_widths,
)


def test_all_sparse_enable_aliases_default_to_query_blocks(monkeypatch):
    for name in (
        "SGLANG_ROLLOUT_SPARSE_DECODE",
        "SGLANG_ROLLOUT_SPARSE_DECODE_MODE",
        "SGLANG_VLM_CACHEBLEND_SPARSE_DECODE",
        "SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MODE",
    ):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("SGLANG_ROLLOUT_SPARSE_DECODE", "1")
    assert cb.VLMCacheBlendConfig.from_env().sparse_decode_mode == "query_blocks"
    monkeypatch.delenv("SGLANG_ROLLOUT_SPARSE_DECODE")
    monkeypatch.setenv("SGLANG_VLM_CACHEBLEND_SPARSE_DECODE", "1")
    assert cb.VLMCacheBlendConfig.from_env().sparse_decode_mode == "query_blocks"

    monkeypatch.setenv("SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MODE", "reuse")
    assert cb.VLMCacheBlendConfig.from_env().sparse_decode_mode == "reuse"


def test_sparse_warmup_clamps_model_context_to_fused_kernel_limit():
    assert sparse_decode_warmup_widths(128000, 4096) == (
        4096,
        8192,
        16384,
        32768,
        MAX_FUSED_SPARSE_CONTEXT_WIDTH,
    )
    assert sparse_decode_warmup_widths(5000, 4096) == (4096, 5000)
    assert sparse_decode_warmup_widths(128000, 70000) == ()


def test_query_block_scores_follow_query_not_reuse_identity():
    query = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    keys = torch.tensor(
        [
            [[[0.1, 0.0]]],
            [[[3.0, 0.0]]],
            [[[0.0, 2.0]]],
            [[[2.0, 0.0]]],
        ]
    )
    scores = score_query_blocks(query, keys)
    assert torch.allclose(scores, torch.tensor([0.1, 3.0, 0.0, 2.0]))

    selection = select_query_aware_blocks(
        query=query,
        representative_keys=keys,
        candidate_start=8,
        block_size=4,
        max_drop_ratio=0.5,
        max_dropped_score_mass=1.0,
    )
    assert selection.kept_block_indices.tolist() == [1, 3]
    assert selection.dropped_block_indices.tolist() == [0, 2]
    assert selection.drop_positions.tolist() == list(range(8, 12)) + list(
        range(16, 20)
    )


def test_query_block_scores_keep_a_block_used_by_any_prompt_landmark():
    queries = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ]
    )
    keys = torch.tensor(
        [
            [[[3.0, 0.0]]],
            [[[0.0, 4.0]]],
            [[[1.0, 1.0]]],
        ]
    )
    assert torch.allclose(
        score_query_blocks(queries, keys), torch.tensor([3.0, 4.0, 1.0])
    )


def test_query_block_structural_drop_ratio_is_a_hard_quality_guard():
    query = torch.ones((1, 2))
    keys = torch.arange(16, dtype=torch.float32).reshape(8, 1, 1, 2)
    selection = select_query_aware_blocks(
        query=query,
        representative_keys=keys,
        candidate_start=0,
        block_size=2,
        max_drop_ratio=0.25,
        max_dropped_score_mass=1.0,
    )
    assert selection.kept_blocks == 6
    assert selection.dropped_blocks == 2
    assert selection.drop_positions.numel() == 4


def test_query_block_mass_budget_fails_closed_for_ambiguous_scores():
    query = torch.ones((1, 2))
    keys = torch.ones((8, 1, 1, 2))
    selection = select_query_aware_blocks(
        query=query,
        representative_keys=keys,
        candidate_start=0,
        block_size=2,
        max_drop_ratio=0.70,
        max_dropped_score_mass=0.05,
    )
    assert selection.dropped_blocks == 0
    assert selection.dropped_score_mass == 0.0


def test_query_block_mass_budget_drops_only_low_mass_tail():
    query = torch.tensor([[1.0, 0.0]])
    keys = torch.tensor(
        [
            [[[8.0, 0.0]]],
            [[[7.0, 0.0]]],
            [[[0.0, 0.0]]],
            [[[-1.0, 0.0]]],
        ]
    )
    selection = select_query_aware_blocks(
        query=query,
        representative_keys=keys,
        candidate_start=0,
        block_size=2,
        max_drop_ratio=0.70,
        max_dropped_score_mass=0.05,
    )
    assert selection.dropped_block_indices.tolist() == [2, 3]
    assert selection.dropped_score_mass <= 0.05


def test_query_block_selection_keeps_mass_scalar_on_device_until_materialized():
    selection = select_query_aware_blocks(
        query=torch.tensor([[1.0, 0.0]]),
        representative_keys=torch.tensor(
            [[[[0.0, 0.0]]], [[[4.0, 0.0]]], [[[-1.0, 0.0]]]]
        ),
        candidate_start=0,
        block_size=2,
        max_drop_ratio=0.70,
        max_dropped_score_mass=0.25,
    )
    assert selection.dropped_score_mass_tensor.ndim == 0
    assert selection.dropped_score_mass_tensor.device == selection.block_scores.device


class _ExtendMode:
    @staticmethod
    def is_extend() -> bool:
        return True


def test_query_plan_registration_is_cacheblend_independent():
    old_config = cb._CONFIG
    old_store = cb._SPARSE_DECODE_STORE
    try:
        cb._CONFIG = dataclasses.replace(
            old_config,
            enabled=False,
            sparse_decode=True,
            sparse_decode_mode="query_blocks",
            sparse_decode_block_size=4,
            sparse_decode_max_drop_ratio=0.5,
            sparse_decode_max_dropped_score_mass=1.0,
            sparse_decode_probe_layer=1,
            sparse_decode_score_representatives=1,
            sparse_decode_query_representatives=2,
            sparse_decode_min_context_tokens=8,
            sparse_decode_keep_first=2,
            sparse_decode_keep_recent=2,
            sparse_decode_min_dropped_tokens=0,
            sparse_decode_min_drop_ratio=0.0,
        )
        cb._SPARSE_DECODE_STORE = cb.SparseDecodePlanStore(max_plans=4)

        seq_len = 16
        req_to_token = torch.arange(seq_len, dtype=torch.int32).reshape(1, -1)
        pool = SimpleNamespace(req_to_token=req_to_token)
        forward_batch = SimpleNamespace(
            forward_mode=_ExtendMode(),
            batch_size=1,
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
            extend_seq_lens_cpu=[seq_len],
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            req_to_token_pool=pool,
            reqs=[SimpleNamespace(rid="request-0")],
        )
        q = torch.zeros((seq_len, 2, 2), dtype=torch.float32)
        q[-1, :, 0] = 1.0
        key_cache = torch.zeros((seq_len, 1, 1, 2), dtype=torch.float32)
        key_cache[8, 0, 0, 0] = 1.0
        key_cache[12, 0, 0, 0] = 3.0

        assert not cb.cacheblend_enabled()
        assert cb.sparse_decode_enabled()
        assert (
            maybe_register_query_block_plans(
                forward_batch=forward_batch,
                q=q,
                key_cache=key_cache,
                layer_id=1,
                page_size=1,
            )
            == 1
        )
        plan = cb.get_sparse_decode_store().get(0)
        assert plan is not None
        assert plan.mode == "query_blocks"
        assert plan.policy == "query-block-mass-prefill-v2"
        assert plan.approximate_error_bound is not None
        assert plan.decision_action == "skip"
        assert plan.drop_positions_host == (2, 3, 4, 5)
        # Query-aware execution is position-only; it must not copy the complete
        # request routing row back to CPU merely to populate legacy diagnostics.
        assert plan.drop_locs.numel() == 0

        decode_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
            reqs=[SimpleNamespace(req_pool_idx=0)],
        )
        sparse = cb.build_sparse_decode_batch(
            decode_batch,
            page_table=req_to_token,
            cache_seqlens=torch.tensor([seq_len], dtype=torch.int32),
        )
        assert sparse is not None
        assert sparse.mode == "query_blocks"
        assert sparse.approximate_error_bound == plan.approximate_error_bound
        assert sparse.candidate_tokens == plan.candidate_tokens
        assert sparse.selected_tokens == plan.selected_tokens
        assert sparse.dropped_tokens == 4
        assert sparse.page_table[0, : sparse.cache_seqlens[0]].tolist() == list(
            range(0, 2)
        ) + list(range(6, seq_len))
    finally:
        cb._CONFIG = old_config
        cb._SPARSE_DECODE_STORE = old_store
        cb.clear_runtime_state()


def test_ambiguous_replan_clears_previous_turn_plan():
    old_config = cb._CONFIG
    old_store = cb._SPARSE_DECODE_STORE
    try:
        cb._CONFIG = dataclasses.replace(
            old_config,
            enabled=False,
            sparse_decode=True,
            sparse_decode_mode="query_blocks",
            sparse_decode_block_size=4,
            sparse_decode_max_drop_ratio=0.70,
            sparse_decode_max_dropped_score_mass=0.05,
            sparse_decode_probe_layer=1,
            sparse_decode_score_representatives=1,
            sparse_decode_query_representatives=2,
            sparse_decode_min_context_tokens=8,
            sparse_decode_keep_first=0,
            sparse_decode_keep_recent=0,
        )
        store = cb.SparseDecodePlanStore(max_plans=4)
        cb._SPARSE_DECODE_STORE = store
        stale_positions = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        store.put(
            cb.SparseDecodePlan(
                request_id="old-turn",
                req_pool_idx=0,
                drop_locs=stale_positions,
                n_image_tokens=0,
                n_drop_tokens=4,
                drop_positions=stale_positions,
                drop_positions_host=(0, 1, 2, 3),
                mode="query_blocks",
            )
        )

        seq_len = 16
        req_to_token = torch.arange(seq_len, dtype=torch.int32).reshape(1, -1)
        forward_batch = SimpleNamespace(
            forward_mode=_ExtendMode(),
            batch_size=1,
            seq_lens_cpu=[seq_len],
            extend_seq_lens_cpu=[seq_len],
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            reqs=[SimpleNamespace(rid="new-turn")],
        )
        # Equal Q/K scores exceed the 5% mass budget even for one block, so V2 must
        # choose dense and must not leave the old turn's sparse plan attached.
        q = torch.ones((seq_len, 1, 2), dtype=torch.float32)
        key_cache = torch.ones((seq_len, 1, 1, 2), dtype=torch.float32)
        assert (
            maybe_register_query_block_plans(
                forward_batch=forward_batch,
                q=q,
                key_cache=key_cache,
                layer_id=1,
                page_size=1,
            )
            == 0
        )
        assert store.get(0) is None
    finally:
        cb._CONFIG = old_config
        cb._SPARSE_DECODE_STORE = old_store
        cb.clear_runtime_state()
