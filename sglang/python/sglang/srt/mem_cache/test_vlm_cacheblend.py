"""Standalone correctness tests for vlm_cacheblend core ops.

Run:
    python -m sglang.srt.mem_cache.test_vlm_cacheblend
or
    python sglang_vision_profile/python/sglang/srt/mem_cache/test_vlm_cacheblend.py

These tests cover the pure-tensor algorithm (no GPU / SGLang runtime needed) and
encode the correctness invariants from the design doc §7.
"""

import os
import sys

import torch

# allow running as a plain script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from sglang.srt.mem_cache import vlm_cacheblend as cb  # noqa: E402


def _cfg(**kw):
    base = dict(
        enabled=True,
        pos_mode="same",
        select_mode="topr",
        recompute_ratio=0.15,
        sim_threshold=0.90,
        target_turn=1,
        max_groups=8,
        verbose=False,
    )
    base.update(kw)
    return cb.VLMCacheBlendConfig(**base)


def test_cacheblend_select_env_alias_and_precedence():
    old_select = os.environ.get("SGLANG_VLM_CACHEBLEND_SELECT")
    old_selector = os.environ.get("SGLANG_VLM_CACHEBLEND_SELECTOR")
    try:
        os.environ.pop("SGLANG_VLM_CACHEBLEND_SELECT", None)
        os.environ["SGLANG_VLM_CACHEBLEND_SELECTOR"] = "kvdev"
        assert cb.VLMCacheBlendConfig.from_env().select_mode == "kvdev"

        os.environ["SGLANG_VLM_CACHEBLEND_SELECT"] = "sim"
        assert cb.VLMCacheBlendConfig.from_env().select_mode == "sim"
    finally:
        if old_select is None:
            os.environ.pop("SGLANG_VLM_CACHEBLEND_SELECT", None)
        else:
            os.environ["SGLANG_VLM_CACHEBLEND_SELECT"] = old_select
        if old_selector is None:
            os.environ.pop("SGLANG_VLM_CACHEBLEND_SELECTOR", None)
        else:
            os.environ["SGLANG_VLM_CACHEBLEND_SELECTOR"] = old_selector
        cb.reload_config_from_env()
    print("ok test_cacheblend_select_env_alias_and_precedence")


def test_blend_kv():
    n, h, d = 6, 2, 4
    donor_k = torch.zeros(n, h, d)
    donor_v = torch.zeros(n, h, d)
    rec_k = torch.ones(n, h, d)
    rec_v = torch.ones(n, h, d) * 2
    mask = torch.tensor([True, False, True, False, False, True])
    k, v = cb.blend_kv(donor_k, donor_v, rec_k, rec_v, mask)
    assert torch.equal(k[mask], rec_k[mask])
    assert torch.equal(k[~mask], donor_k[~mask])
    assert torch.equal(v[mask], rec_v[mask])
    assert torch.equal(v[~mask], donor_v[~mask])
    print("ok test_blend_kv")


def test_select_full_recompute_equiv():
    # r=1.0 must select ALL tokens => equivalent to full prefill (design §7).
    cfg = _cfg(recompute_ratio=1.0)
    mask = cb.select_recompute_tokens(10, cfg, deviation=torch.rand(10))
    assert bool(mask.all()), "r=1.0 must recompute every token"
    print("ok test_select_full_recompute_equiv")


def test_select_topr_count():
    cfg = _cfg(select_mode="kvdev", recompute_ratio=0.2)
    dev = torch.tensor([0.0, 9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0])
    mask = cb.select_recompute_tokens(10, cfg, deviation=dev)
    assert int(mask.sum()) == 2, int(mask.sum())
    # highest-deviation tokens (idx 1 and 3) chosen
    assert mask[1] and mask[3]
    print("ok test_select_topr_count")


def test_select_sim_threshold():
    cfg = _cfg(select_mode="sim", sim_threshold=0.9, recompute_ratio=1.0)
    sim = torch.tensor([0.99, 0.5, 0.95, 0.2, 0.91])
    mask = cb.select_recompute_tokens(5, cfg, similarity=sim)
    # recompute_ratio=1.0 short-circuits to all-True; use <1 to test threshold path
    cfg2 = _cfg(select_mode="sim", sim_threshold=0.9, recompute_ratio=0.99)
    mask2 = cb.select_recompute_tokens(5, cfg2, similarity=sim)
    assert mask2.tolist() == [False, True, False, True, False], mask2.tolist()
    print("ok test_select_sim_threshold")


def test_target_turn_enabled_all_and_list():
    cfg_all = _cfg(target_turns="all")
    assert not cb.target_turn_enabled(0, cfg_all)
    assert cb.target_turn_enabled(1, cfg_all)
    assert cb.target_turn_enabled(15, cfg_all)

    cfg_list = _cfg(target_turns="1,3,15")
    assert cb.target_turn_enabled(1, cfg_list)
    assert not cb.target_turn_enabled(2, cfg_list)
    assert cb.target_turn_enabled(15, cfg_list)
    print("ok test_target_turn_enabled_all_and_list")


def test_positions_match():
    a = torch.arange(12).reshape(3, 4)
    b = a.clone()
    assert cb.positions_match(a, b)
    b[0, 0] += 1
    assert not cb.positions_match(a, b)
    assert not cb.positions_match(None, b)
    print("ok test_positions_match")


def test_image_token_mask_for_slot():
    ids = torch.tensor([1, 7, 7, 2, 7, 7, 7, 3, 7])
    last = cb.image_token_mask_for_slot(ids, 7, -1)
    assert last.tolist() == [False, False, False, False, False, False, False, False, True]
    first = cb.image_token_mask_for_slot(ids, 7, 0)
    assert first.tolist() == [False, True, True, False, False, False, False, False, False]
    second = cb.image_token_mask_for_slot(ids, 7, 1)
    assert second.tolist() == [False, False, False, False, True, True, True, False, False]
    missing = cb.image_token_mask_for_slot(ids, 7, 5)
    assert not bool(missing.any())
    locs = cb.image_token_locs(ids, torch.arange(ids.numel()), 7, target_image_slot=1)
    assert locs.tolist() == [4, 5, 6]
    print("ok test_image_token_mask_for_slot")


def test_image_token_mask_uses_pad_values():
    # Qwen2.5-VL padded input ids use per-image hash pad_values, not hf image_token_id.
    ids = torch.tensor([10, 101, 101, 11, 20, 202, 202, 202, 21, 30])
    mask = cb.image_token_mask_for_slot(
        ids,
        image_token_id=999,
        target_image_slot=1,
        image_token_values=(101, 202),
    )
    assert mask.tolist() == [False, False, False, False, False, True, True, True, False, False]
    locs = cb.image_token_locs(
        ids,
        torch.arange(100, 100 + ids.numel()),
        image_token_id=999,
        target_image_slot=0,
        image_token_values=(101, 202),
    )
    assert locs.tolist() == [101, 102]
    fallback = cb.image_token_mask_for_slot(
        torch.tensor([1, 999, 999, 2]),
        image_token_id=999,
        target_image_slot=0,
        image_token_values=(101, 202),
    )
    assert fallback.tolist() == [False, True, True, False]
    print("ok test_image_token_mask_uses_pad_values")


def test_select_grid_sig_aligns_current_image_spans_to_grid_suffix():
    class Item:
        def __init__(self, pad_value, grid):
            self.pad_value = pad_value
            self.image_grid_thw = torch.tensor([grid])

        def is_image(self):
            return True

    class MM:
        def __init__(self):
            self.mm_items = [
                Item(101, (1, 10, 10)),
                Item(102, (1, 11, 11)),
                Item(201, (1, 20, 20)),
                Item(202, (1, 21, 21)),
            ]

    class Req:
        multimodal_inputs = MM()
        # Current turn has only the two suffix images in fill_ids.
        fill_ids = [7, 201, 201, 8, 202, 202, 202, 9]

    slot, grid = cb.select_grid_sig_for_req(Req(), -1, image_token_id=151655)
    assert slot == 1
    assert grid == (1, 21, 21)

    slot, grid = cb.select_grid_sig_for_req(Req(), 3, image_token_id=151655)
    assert slot == 1
    assert grid == (1, 21, 21)

    slot, grid = cb.select_grid_sig_for_req(Req(), 0, image_token_id=151655)
    assert slot == 0
    assert grid == (1, 20, 20)
    print("ok test_select_grid_sig_aligns_current_image_spans_to_grid_suffix")


def test_resolve_target_slots_multi_slot():
    """Method A: 'all'/list selectors expand to per-slot indices (original+refocus)."""

    class Item:
        def __init__(self, pad_value, grid):
            self.pad_value = pad_value
            self.image_grid_thw = torch.tensor([grid])

        def is_image(self):
            return True

    class MM:
        def __init__(self):
            self.mm_items = [
                Item(201, (1, 20, 20)),
                Item(202, (1, 21, 21)),
            ]

    class Req:
        multimodal_inputs = MM()
        # Two image spans present in the current turn (slot0 original, slot1 refocus).
        fill_ids = [7, 201, 201, 8, 202, 202, 202, 9]

    tok = 151655
    # Legacy default -> single last slot (back-compat).
    legacy = cb._resolve_target_slots(Req(), _cfg(target_image_slots="-1"), tok)
    assert legacy == [1], legacy

    # "all" -> every span.
    all_slots = cb._resolve_target_slots(Req(), _cfg(target_image_slots="all"), tok)
    assert all_slots == [0, 1], all_slots

    # Explicit list, de-duplicated and order-preserving; -1 normalizes to last span.
    explicit = cb._resolve_target_slots(Req(), _cfg(target_image_slots="0,-1"), tok)
    assert explicit == [0, 1], explicit
    print("ok test_resolve_target_slots_multi_slot")


def test_get_recipient_blend_plan_for_is_slot_aware():
    """Multi-slot: per (request_id, group_key) lookup returns the right slot plan."""

    def _plan(group_key, reused):
        return cb.RecipientKVBlendPlan(
            request_id="r0",
            group_key=group_key,
            img_locs=torch.tensor([0, 1], dtype=torch.long),
            positions=None,
            recompute_mask=torch.tensor([False, False]),
            grid_sig=(1, 2, 2),
            n_image_tokens=2,
            reused_tokens=reused,
            recomputed_tokens=0,
            pos_mode="same",
            select_mode="kvdev",
        )

    gk0 = ("uid", 1, (1, 20, 20), 0, 0)
    gk1 = ("uid", 1, (1, 21, 21), 0, 1)
    cb.set_recipient_blend_plans((_plan(gk0, 5), _plan(gk1, 7)))
    try:
        assert cb.get_recipient_blend_plan_for("r0", gk1).reused_tokens == 7
        assert cb.get_recipient_blend_plan_for("r0", gk0).reused_tokens == 5
        # Unknown group_key falls back to first plan for the request.
        assert cb.get_recipient_blend_plan_for("r0", ("x",)).reused_tokens == 5
    finally:
        cb.clear_recipient_blend_plans()
    print("ok test_get_recipient_blend_plan_for_is_slot_aware")


def test_image_token_locs_from_request_uses_request_index():
    class Pool:
        req_to_token = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15],
                [20, 21, 22, 23, 24, 25],
            ],
            dtype=torch.long,
        )

    class Item:
        pad_value = 202

        def is_image(self):
            return True

    class MM:
        mm_items = [Item()]
        mrope_positions = torch.arange(18, dtype=torch.long).reshape(3, 6)

    class Req:
        def __init__(self, fill_ids):
            self.fill_ids = fill_ids
            self.origin_input_ids = fill_ids
            self.output_ids = []
            self.multimodal_inputs = MM()

    class Batch:
        reqs = [Req([1, 101, 101, 2]), Req([3, 202, 202, 202, 4])]
        req_to_token_pool = Pool()
        req_pool_indices = torch.tensor([0, 1], dtype=torch.long)
        out_cache_loc = torch.arange(8, dtype=torch.long)

    ctx = cb.RequestContext(
        group_key=("g",),
        role="donor",
        image_token_id=999,
        image_token_values=(202,),
        target_image_slot=0,
        request_index=1,
    )
    locs, positions, reason = cb.image_token_locs_from_request(Batch(), ctx)
    assert locs.tolist() == [21, 22, 23]
    assert positions.tolist() == [[1, 2, 3], [7, 8, 9], [13, 14, 15]]
    assert reason.startswith("full_request_ok:"), reason
    print("ok test_image_token_locs_from_request_uses_request_index")


def test_rerotate_inverse_identity():
    # rerotating donor->recipient with identical positions must be a no-op.
    torch.manual_seed(0)
    n, h, d = 5, 2, 8
    rotary_dim = d
    half = rotary_dim // 2
    max_pos = 64
    # build a neox cos_sin cache: [max_pos, rotary_dim] (cos half | sin half)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    t = torch.arange(max_pos).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    mrope_section = [2, 1, 1]  # sums to half=4
    pos = torch.randint(0, max_pos, (3, n))
    k = torch.randn(n, h, d)
    out = cb.rerotate_keys_mrope(k, pos, pos.clone(), cos_sin, mrope_section)
    assert torch.allclose(out, k, atol=1e-5), (out - k).abs().max()
    print("ok test_rerotate_inverse_identity")


def test_rerotate_relative_consistency():
    # rotating donor_k (defined at donor_pos) to recipient_pos must equal directly
    # rotating the raw key at recipient_pos.
    torch.manual_seed(1)
    n, h, d = 4, 1, 8
    rotary_dim = d
    max_pos = 128
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    t = torch.arange(max_pos).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    mrope_section = [2, 1, 1]
    raw = torch.randn(n, h, d)

    def rot_at(pos):
        half = cos_sin.shape[-1] // 2
        cos_c, sin_c = cos_sin[..., :half], cos_sin[..., half:]
        cos_parts, sin_parts, off = [], [], 0
        for si, sl in enumerate(mrope_section):
            idx = pos[si]
            cos_parts.append(cos_c.index_select(0, idx)[:, off:off + sl])
            sin_parts.append(sin_c.index_select(0, idx)[:, off:off + sl])
            off += sl
        cos = torch.cat(cos_parts, -1)
        sin = torch.cat(sin_parts, -1)
        return cb._rotate_neox(raw, cos, sin)

    donor_pos = torch.randint(0, max_pos, (3, n))
    rec_pos = torch.randint(0, max_pos, (3, n))
    donor_k = rot_at(donor_pos)
    expected = rot_at(rec_pos)
    got = cb.rerotate_keys_mrope(donor_k, donor_pos, rec_pos, cos_sin, mrope_section)
    assert torch.allclose(got, expected, atol=1e-4), (got - expected).abs().max()
    print("ok test_rerotate_relative_consistency")


def test_donor_store_lru():
    store = cb.DonorKVStore(max_groups=2)
    for i in range(3):
        store.get_or_create_donor((i,), n_image_tokens=4, grid_sig=((1, 2, 2),), positions=None)
    assert store.lookup((0,)) is None  # evicted
    assert store.lookup((1,)) is not None
    assert store.lookup((2,)) is not None
    e = store.get_or_create_donor((1,), 4, ((1, 2, 2),), None)
    e.record_layer(0, torch.zeros(4, 2, 3), torch.zeros(4, 2, 3))
    assert e.has_layer(0)
    print("ok test_donor_store_lru")


def test_kv_deviation_and_blend_full():
    # blend with full recompute mask must equal recomputed tensors entirely.
    n, h, d = 7, 2, 4
    donor_k, donor_v = torch.randn(n, h, d), torch.randn(n, h, d)
    rec_k, rec_v = torch.randn(n, h, d), torch.randn(n, h, d)
    full = torch.ones(n, dtype=torch.bool)
    k, v = cb.blend_kv(donor_k, donor_v, rec_k, rec_v, full)
    assert torch.equal(k, rec_k) and torch.equal(v, rec_v)
    dev = cb.kv_deviation(rec_k, donor_k)
    assert dev.shape == (n,)
    print("ok test_kv_deviation_and_blend_full")


def test_kv_cosine_similarity():
    donor_k = torch.tensor([[[1.0, 0.0]], [[1.0, 1.0]], [[0.0, 2.0]]])
    rec_k = torch.tensor([[[1.0, 0.0]], [[1.0, -1.0]], [[0.0, 4.0]]])
    sim = cb.kv_cosine_similarity(rec_k, donor_k)
    assert torch.allclose(sim, torch.tensor([1.0, 0.0, 1.0]), atol=1e-6), sim
    print("ok test_kv_cosine_similarity")


def test_apply_recipient_kv_blend_updates_pool_and_direct_tensors():
    cb._CONFIG = _cfg(fast_path=True)
    cb._DONOR_STORE = cb.DonorKVStore(max_groups=4)
    cb.clear_recipient_blend_plans()

    class Pool:
        def __init__(self):
            self.k = torch.zeros(16, 1, 2)
            self.v = torch.zeros(16, 1, 2)

        def get_key_buffer(self, layer_id):
            assert layer_id == 0
            return self.k

        def get_value_buffer(self, layer_id):
            assert layer_id == 0
            return self.v

    class Batch:
        token_to_kv_pool = Pool()

    donor_k = torch.tensor(
        [[[10.0, 11.0]], [[20.0, 21.0]], [[30.0, 31.0]]]
    )
    donor_v = donor_k + 100
    entry = cb.get_donor_store().get_or_create_donor(
        ("g",), n_image_tokens=3, grid_sig=(1, 1, 3), positions=None
    )
    entry.record_layer(0, donor_k, donor_v)
    cb.get_donor_store().mark_complete(("g",))

    plan = cb.RecipientKVBlendPlan(
        request_id="r1",
        group_key=("g",),
        img_locs=torch.tensor([3, 5, 8], dtype=torch.long),
        positions=None,
        recompute_mask=torch.tensor([False, True, False]),
        grid_sig=(1, 1, 3),
        n_image_tokens=3,
        reused_tokens=2,
        recomputed_tokens=1,
        pos_mode="same",
        select_mode="topr",
    )
    cb.set_recipient_blend_plans((plan,))

    cache_locs = torch.tensor([2, 3, 5, 8, 9], dtype=torch.long)
    k = torch.ones(5, 1, 2)
    v = torch.ones(5, 1, 2) * 2
    Batch.token_to_kv_pool.k[cache_locs] = k
    Batch.token_to_kv_pool.v[cache_locs] = v

    reused = cb.apply_recipient_kv_blend_for_layer(
        forward_batch=Batch(),
        layer_id=0,
        cache_locs=cache_locs,
        k=k,
        v=v,
    )
    assert reused == 2
    assert cb.recipient_blend_was_used("r1")
    assert torch.equal(Batch.token_to_kv_pool.k[3], donor_k[0])
    assert torch.equal(Batch.token_to_kv_pool.v[3], donor_v[0])
    assert torch.equal(Batch.token_to_kv_pool.k[8], donor_k[2])
    assert torch.equal(Batch.token_to_kv_pool.v[8], donor_v[2])
    assert torch.equal(Batch.token_to_kv_pool.k[5], torch.ones(1, 2))
    assert torch.equal(k[1], donor_k[0])
    assert torch.equal(v[1], donor_v[0])
    assert torch.equal(k[3], donor_k[2])
    assert torch.equal(v[3], donor_v[2])
    assert torch.equal(k[2], torch.ones(1, 2))
    cb.clear_recipient_blend_plans()
    print("ok test_apply_recipient_kv_blend_updates_pool_and_direct_tensors")


def _apply_fixture(fast_apply):
    """Build donor/plan/pool fixture for apply parity tests. Returns (Batch, plan)."""
    cb._CONFIG = _cfg(fast_path=True, fast_apply=fast_apply)
    cb._DONOR_STORE = cb.DonorKVStore(max_groups=4)
    cb.clear_recipient_blend_plans()

    class Pool:
        def __init__(self):
            self.k = torch.zeros(16, 1, 2)
            self.v = torch.zeros(16, 1, 2)

        def get_key_buffer(self, layer_id):
            return self.k

        def get_value_buffer(self, layer_id):
            return self.v

    class Batch:
        token_to_kv_pool = Pool()

    donor_k = torch.tensor([[[10.0, 11.0]], [[20.0, 21.0]], [[30.0, 31.0]]])
    donor_v = donor_k + 100
    entry = cb.get_donor_store().get_or_create_donor(
        ("g",), n_image_tokens=3, grid_sig=(1, 1, 3), positions=None
    )
    entry.record_layer(0, donor_k, donor_v)
    cb.get_donor_store().mark_complete(("g",))

    plan = cb.RecipientKVBlendPlan(
        request_id="r1",
        group_key=("g",),
        img_locs=torch.tensor([3, 5, 8], dtype=torch.long),
        positions=None,
        recompute_mask=torch.tensor([False, True, False]),
        grid_sig=(1, 1, 3),
        n_image_tokens=3,
        reused_tokens=2,
        recomputed_tokens=1,
        pos_mode="same",
        select_mode="topr",
    )
    cb.set_recipient_blend_plans((plan,))
    return Batch, plan, donor_k, donor_v


def test_fast_apply_matches_slow_path():
    """Method B: fast_apply must be bit-for-bit identical to the slow apply path."""

    def run(fast_apply):
        Batch, plan, donor_k, donor_v = _apply_fixture(fast_apply)
        cache_locs = torch.tensor([2, 3, 5, 8, 9], dtype=torch.long)
        k = torch.ones(5, 1, 2)
        v = torch.ones(5, 1, 2) * 2
        Batch.token_to_kv_pool.k[cache_locs] = k
        Batch.token_to_kv_pool.v[cache_locs] = v
        # Call the same layer twice to exercise the per-forward cache reuse.
        for _ in range(2):
            cb.apply_recipient_kv_blend_for_layer(
                forward_batch=Batch(), layer_id=0, cache_locs=cache_locs, k=k, v=v
            )
        cb.clear_recipient_blend_plans()
        return (
            Batch.token_to_kv_pool.k.clone(),
            Batch.token_to_kv_pool.v.clone(),
            k.clone(),
            v.clone(),
        )

    slow = run(False)
    fast = run(True)
    for a, b in zip(slow, fast):
        assert torch.equal(a, b), "fast_apply diverged from slow path"
    print("ok test_fast_apply_matches_slow_path")


def test_fast_apply_skips_direct_write_for_pool_reading_backend():
    """Method B: reads_kv_from_pool skips the dead direct-tensor write (FA3).

    The pool must still receive donor K/V (that is what FA3 reads); the direct k/v
    tensors must be left untouched, and the pool result must match the full path.
    """

    def run(reads_kv_from_pool):
        Batch, plan, donor_k, donor_v = _apply_fixture(fast_apply=True)
        cache_locs = torch.tensor([2, 3, 5, 8, 9], dtype=torch.long)
        k = torch.ones(5, 1, 2)
        v = torch.ones(5, 1, 2) * 2
        Batch.token_to_kv_pool.k[cache_locs] = k
        Batch.token_to_kv_pool.v[cache_locs] = v
        cb.apply_recipient_kv_blend_for_layer(
            forward_batch=Batch(),
            layer_id=0,
            cache_locs=cache_locs,
            k=k,
            v=v,
            reads_kv_from_pool=reads_kv_from_pool,
        )
        cb.clear_recipient_blend_plans()
        return Batch.token_to_kv_pool.k.clone(), k.clone()

    pool_full, k_full = run(False)
    pool_skip, k_skip = run(True)
    # Pool write identical regardless of the direct-write skip (FA3 reads the pool).
    assert torch.equal(pool_full, pool_skip), "pool diverged when skipping direct write"
    # With skip, the direct k tensor keeps its original recipient values (untouched).
    assert torch.equal(k_skip, torch.ones(5, 1, 2)), "direct tensor should be untouched"
    # Without skip, the direct k tensor was overwritten with donor K at reused slots.
    assert not torch.equal(k_full, torch.ones(5, 1, 2)), "direct write expected off-skip"
    print("ok test_fast_apply_skips_direct_write_for_pool_reading_backend")


def test_fast_apply_reuse_indices_cache_invalidates_on_mask_change():
    """Cache keyed by mask sum: finalizing recompute mask must recompute indices."""
    cb._CONFIG = _cfg(fast_path=True, fast_apply=True)
    cb.clear_recipient_blend_plans()
    plan = cb.RecipientKVBlendPlan(
        request_id="rb",
        group_key=("g",),
        img_locs=torch.tensor([11, 13, 17, 19], dtype=torch.long),
        positions=None,
        recompute_mask=torch.tensor([True, True, True, True]),  # bootstrap: all recompute
        grid_sig=(1, 1, 4),
        n_image_tokens=4,
        reused_tokens=0,
        recomputed_tokens=4,
        pos_mode="same",
        select_mode="kvdev",
    )
    cb.set_recipient_blend_plans((plan,))
    cache_locs = torch.tensor([7, 11, 12, 13, 17, 18, 19], dtype=torch.long)
    idx0 = cb.recipient_reuse_token_indices(cache_locs)
    assert idx0.tolist() == [], idx0.tolist()
    # Bootstrap finalizes: only index 1 recomputes now -> reuse the other three.
    plan.recompute_mask = torch.tensor([False, True, False, False])
    idx1 = cb.recipient_reuse_token_indices(cache_locs)
    assert idx1.tolist() == [1, 4, 6], idx1.tolist()
    cb.clear_recipient_blend_plans()
    print("ok test_fast_apply_reuse_indices_cache_invalidates_on_mask_change")


def test_recipient_reuse_token_indices():
    cb.clear_recipient_blend_plans()
    plan = cb.RecipientKVBlendPlan(
        request_id="r2",
        group_key=("g",),
        img_locs=torch.tensor([11, 13, 17, 19], dtype=torch.long),
        positions=None,
        recompute_mask=torch.tensor([False, True, False, False]),
        grid_sig=(1, 1, 4),
        n_image_tokens=4,
        reused_tokens=3,
        recomputed_tokens=1,
        pos_mode="same",
        select_mode="topr",
    )
    cb.set_recipient_blend_plans((plan,))
    cache_locs = torch.tensor([7, 11, 12, 13, 17, 18, 19], dtype=torch.long)
    idx = cb.recipient_reuse_token_indices(cache_locs)
    assert idx.tolist() == [1, 4, 6], idx.tolist()
    cb.clear_recipient_blend_plans()
    print("ok test_recipient_reuse_token_indices")


def test_recipient_active_query_ranges():
    cb.clear_recipient_blend_plans()
    plan = cb.RecipientKVBlendPlan(
        request_id="r3",
        group_key=("g",),
        img_locs=torch.tensor([101, 102, 202], dtype=torch.long),
        positions=None,
        recompute_mask=torch.tensor([False, False, False]),
        grid_sig=(1, 1, 3),
        n_image_tokens=3,
        reused_tokens=3,
        recomputed_tokens=0,
        pos_mode="same",
        select_mode="topr",
    )
    cb.set_recipient_blend_plans((plan,))

    class Batch:
        out_cache_loc = torch.tensor(
            [100, 101, 102, 103, 104, 200, 201, 202, 203], dtype=torch.long
        )
        extend_seq_lens = torch.tensor([5, 4], dtype=torch.int32)
        extend_seq_lens_cpu = [5, 4]
        # Prefix lengths are 10 and 20.
        seq_lens = torch.tensor([15, 24], dtype=torch.int32)
        req_pool_indices = torch.tensor([7, 8], dtype=torch.long)

    ranges = cb.recipient_active_query_ranges(Batch())
    assert ranges is not None
    assert cb.recipient_active_query_ranges(Batch()) is ranges
    assert ranges.query_indices.tolist() == [0, 3, 4, 5, 6, 8]
    assert ranges.cu_seqlens_q.tolist() == [0, 1, 3, 5, 6]
    assert ranges.max_seqlen_q == 2
    # Request 0 active ranges end at local offsets 1 and 5 => seq lens 11 and 15.
    # Request 1 active ranges end at local offsets 2 and 4 => seq lens 22 and 24.
    assert ranges.range_end_lens.tolist() == [11, 15, 22, 24]
    assert ranges.range_batch_indices.tolist() == [0, 0, 1, 1]
    assert ranges.range_req_indices.tolist() == [7, 7, 8, 8]
    cb.clear_recipient_blend_plans()
    assert cb.recipient_active_query_ranges(Batch()) is None
    print("ok test_recipient_active_query_ranges")


def test_compact_active_indices_match_ranges_when_finalized():
    # Method C: compact prefill must drop exactly the reused image tokens, i.e. the
    # active indices must equal recipient_active_query_ranges().query_indices.
    cb._CONFIG = _cfg(fast_path=True, compact_prefill=True)
    cb.clear_recipient_blend_plans()
    plan = cb.RecipientKVBlendPlan(
        request_id="rc",
        group_key=("g",),
        img_locs=torch.tensor([101, 102, 202], dtype=torch.long),
        positions=None,
        recompute_mask=torch.tensor([False, False, False]),  # all reused (finalized)
        grid_sig=(1, 1, 3),
        n_image_tokens=3,
        reused_tokens=3,
        recomputed_tokens=0,
        pos_mode="same",
        select_mode="topr",
        pending_deviation=False,
    )
    cb.set_recipient_blend_plans((plan,))

    class Batch:
        out_cache_loc = torch.tensor(
            [100, 101, 102, 103, 104, 200, 201, 202, 203], dtype=torch.long
        )
        extend_seq_lens = torch.tensor([5, 4], dtype=torch.int32)
        extend_seq_lens_cpu = [5, 4]
        seq_lens = torch.tensor([15, 24], dtype=torch.int32)
        req_pool_indices = torch.tensor([7, 8], dtype=torch.long)

    assert cb.recipient_plans_finalized() is True
    ranges = cb.recipient_active_query_ranges(Batch())
    active = cb.recipient_compact_active_indices(Batch())
    assert active is not None
    assert active.tolist() == ranges.query_indices.tolist() == [0, 3, 4, 5, 6, 8]

    # Disabled flag => no compaction even though reuse exists.
    cb._CONFIG = _cfg(fast_path=True, compact_prefill=False)
    assert cb.recipient_compact_active_indices(Batch()) is None

    cb.clear_recipient_blend_plans()
    cb._CONFIG = _cfg(fast_path=True)
    print("ok test_compact_active_indices_match_ranges_when_finalized")


def test_compact_active_indices_deferred_until_finalized():
    # kvdev/sim defer the mask to a bootstrap layer; compact prefill must return None
    # while pending (mask is all-recompute anyway) and never drop tokens too early.
    cb._CONFIG = _cfg(fast_path=True, compact_prefill=True, select_mode="kvdev")
    cb.clear_recipient_blend_plans()
    pending_plan = cb.RecipientKVBlendPlan(
        request_id="rd",
        group_key=("g",),
        img_locs=torch.tensor([101, 102, 202], dtype=torch.long),
        positions=None,
        recompute_mask=torch.ones(3, dtype=torch.bool),  # pending: mark all recompute
        grid_sig=(1, 1, 3),
        n_image_tokens=3,
        reused_tokens=0,
        recomputed_tokens=3,
        pos_mode="same",
        select_mode="kvdev",
        bootstrap_layer_id=0,
        pending_deviation=True,
    )
    cb.set_recipient_blend_plans((pending_plan,))

    class Batch:
        out_cache_loc = torch.tensor(
            [100, 101, 102, 103, 104, 200, 201, 202, 203], dtype=torch.long
        )
        extend_seq_lens = torch.tensor([5, 4], dtype=torch.int32)
        extend_seq_lens_cpu = [5, 4]
        seq_lens = torch.tensor([15, 24], dtype=torch.int32)
        req_pool_indices = torch.tensor([7, 8], dtype=torch.long)

    assert cb.recipient_plans_finalized() is False
    assert cb.recipient_compact_active_indices(Batch()) is None

    cb.clear_recipient_blend_plans()
    cb._CONFIG = _cfg(fast_path=True)
    print("ok test_compact_active_indices_deferred_until_finalized")


def test_recipient_attention_skip_stats():
    cb.clear_recipient_blend_plans()
    plan = cb.RecipientKVBlendPlan(
        request_id="r4",
        group_key=("g",),
        img_locs=torch.tensor([11, 12, 13], dtype=torch.long),
        positions=None,
        recompute_mask=torch.tensor([False, True, False]),
        grid_sig=(1, 1, 3),
        n_image_tokens=3,
        reused_tokens=2,
        recomputed_tokens=1,
        pos_mode="same",
        select_mode="topr",
    )
    cb.set_recipient_blend_plans((plan,))
    ranges = cb.RecipientActiveQueryRanges(
        query_indices=torch.tensor([0, 2], dtype=torch.long),
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        max_seqlen_q=1,
        range_end_lens=torch.tensor([5, 7], dtype=torch.int32),
        range_batch_indices=torch.tensor([0, 0], dtype=torch.long),
        range_req_indices=torch.tensor([3, 3], dtype=torch.long),
    )
    cb.mark_recipient_attention_skip_used(ranges)
    assert cb.recipient_attention_skip_stats("r4") == (2, 2)
    stats = cb.CacheBlendStats(
        role="recipient",
        request_id="r4",
        attention_skipped_tokens=2,
        attention_active_ranges=2,
        plan_wall_ms=1.0,
        apply_wall_ms=2.5,
        applied_layers=3,
        gate_reason="",
    )
    d = stats.finalize().to_dict()
    assert d["cacheblend_attention_skipped_tokens"] == "2"
    assert d["cacheblend_attention_active_ranges"] == "2"
    assert d["cacheblend_extend_wall_ms"] == "3.500"
    assert d["cacheblend_plan_wall_ms"] == "1.000"
    assert d["cacheblend_apply_wall_ms"] == "2.500"
    assert d["cacheblend_applied_layers"] == "3"
    cb.clear_recipient_blend_plans()
    print("ok test_recipient_attention_skip_stats")


def test_finalize_recipient_plan_deviation_picks_high_deviation():
    # kvdev finalize must recompute the highest-KV-deviation tokens, reuse the rest.
    cfg = _cfg(select_mode="kvdev", recompute_ratio=0.25)  # 2 of 8 tokens
    n, h, d = 8, 1, 2
    donor_k = torch.zeros(n, h, d)
    recipient_k = torch.zeros(n, h, d)
    recipient_k[2] = 9.0  # largest deviation
    recipient_k[6] = 8.0  # second largest
    recipient_k[5] = 1.0  # small deviation, must NOT be selected
    plan = cb.RecipientKVBlendPlan(
        request_id="rk",
        group_key=("g",),
        img_locs=torch.arange(n, dtype=torch.long),
        positions=None,
        recompute_mask=torch.ones(n, dtype=torch.bool),  # pending => all recompute
        grid_sig=(1, 1, n),
        n_image_tokens=n,
        reused_tokens=0,
        recomputed_tokens=n,
        pos_mode="same",
        select_mode="kvdev",
        bootstrap_layer_id=0,
        pending_deviation=True,
    )
    dev = cb.finalize_recipient_plan_deviation(plan, recipient_k, donor_k, cfg)
    assert dev.shape == (n,)
    assert plan.pending_deviation is False
    assert int(plan.recompute_mask.sum()) == 2, int(plan.recompute_mask.sum())
    assert bool(plan.recompute_mask[2]) and bool(plan.recompute_mask[6])
    assert not bool(plan.recompute_mask[5])
    assert plan.recomputed_tokens == 2 and plan.reused_tokens == n - 2
    print("ok test_finalize_recipient_plan_deviation_picks_high_deviation")


def test_low_value_gate_disables_small_reuse_set():
    cfg = _cfg(select_mode="kvdev", recompute_ratio=0.5, min_reused_tokens=5)
    n, h, d = 8, 1, 2
    donor_k = torch.zeros(n, h, d)
    recipient_k = torch.arange(n * h * d, dtype=torch.float).reshape(n, h, d)
    plan = cb.RecipientKVBlendPlan(
        request_id="rg",
        group_key=("g",),
        img_locs=torch.arange(n, dtype=torch.long),
        positions=None,
        recompute_mask=torch.ones(n, dtype=torch.bool),
        grid_sig=(1, 1, n),
        n_image_tokens=n,
        reused_tokens=0,
        recomputed_tokens=n,
        pos_mode="same",
        select_mode="kvdev",
        bootstrap_layer_id=0,
        pending_deviation=True,
    )
    cb.finalize_recipient_plan_deviation(plan, recipient_k, donor_k, cfg)
    assert plan.pending_deviation is False
    assert bool(plan.recompute_mask.all())
    assert plan.recomputed_tokens == n
    assert plan.reused_tokens == 0
    assert plan.gate_reason.startswith("low_value_gate:")
    print("ok test_low_value_gate_disables_small_reuse_set")


def test_finalize_recipient_plan_similarity_picks_low_cosine():
    # sim finalize must recompute low-cosine tokens and reuse high-cosine tokens.
    cfg = _cfg(select_mode="sim", sim_threshold=0.5, recompute_ratio=0.5)
    n = 4
    donor_k = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, 1.0]],
            [[2.0, 0.0]],
        ]
    )
    recipient_k = torch.tensor(
        [
            [[1.0, 0.0]],   # cos 1.0
            [[0.0, -1.0]],  # cos -1.0
            [[-1.0, -1.0]], # cos -1.0
            [[3.0, 0.0]],   # cos 1.0
        ]
    )
    plan = cb.RecipientKVBlendPlan(
        request_id="rs",
        group_key=("g",),
        img_locs=torch.arange(n, dtype=torch.long),
        positions=None,
        recompute_mask=torch.ones(n, dtype=torch.bool),
        grid_sig=(1, 1, n),
        n_image_tokens=n,
        reused_tokens=0,
        recomputed_tokens=n,
        pos_mode="same",
        select_mode="sim",
        bootstrap_layer_id=0,
        pending_deviation=True,
    )
    sim = cb.finalize_recipient_plan_similarity(plan, recipient_k, donor_k, cfg)
    assert sim.shape == (n,)
    assert plan.pending_deviation is False
    assert int(plan.recompute_mask.sum()) == 2, int(plan.recompute_mask.sum())
    assert bool(plan.recompute_mask[1]) and bool(plan.recompute_mask[2])
    assert not bool(plan.recompute_mask[0]) and not bool(plan.recompute_mask[3])
    assert plan.recomputed_tokens == 2 and plan.reused_tokens == 2
    print("ok test_finalize_recipient_plan_similarity_picks_low_cosine")


def test_build_plan_deferred_for_sim():
    cb._CONFIG = _cfg(select_mode="sim", recompute_ratio=0.25, fast_path=True)
    cb._DONOR_STORE = cb.DonorKVStore(max_groups=4)
    cb.clear_recipient_blend_plans()

    ctx = cb.RequestContext(
        group_key=("g",),
        role="recipient",
        image_token_id=999,
        request_id="rs2",
        agent_turn=1,
        grid_sig=(1, 1, 4),
    )
    positions = torch.arange(12, dtype=torch.long).reshape(3, 4)
    entry = cb.get_donor_store().get_or_create_donor(
        ("g",), n_image_tokens=4, grid_sig=(1, 1, 4), positions=positions
    )
    entry.record_layer(0, torch.zeros(4, 1, 2), torch.zeros(4, 1, 2))
    cb.get_donor_store().mark_complete(("g",))

    plan, stats = cb.build_recipient_kv_blend_plan(
        ctx,
        torch.arange(4, dtype=torch.long),
        positions,
    )
    assert plan is not None
    assert plan.pending_deviation is True
    assert plan.select_mode == "sim"
    assert bool(plan.recompute_mask.all())
    assert stats.fallback_reason == "recipient_kv_blend_deferred_sim"
    cb.clear_recipient_blend_plans()
    print("ok test_build_plan_deferred_for_sim")


def test_apply_kvdev_bootstrap_finalizes_then_reuses_next_layer():
    # End-to-end: a deferred kvdev plan must (1) at the bootstrap layer measure deviation
    # from the recipient's own K, finalize the high-deviation recompute set, AND write
    # donor K/V for the reused tokens on that same layer (so the same-layer attention/MLP
    # skip stays consistent), then (2) keep reusing on the following layer.
    cb._CONFIG = _cfg(select_mode="kvdev", recompute_ratio=0.5)  # 2 of 4 recompute
    cb._DONOR_STORE = cb.DonorKVStore(max_groups=4)
    cb.clear_recipient_blend_plans()
    n = 4

    class Pool:
        def __init__(self):
            self.k = {0: torch.zeros(16, 1, 2), 1: torch.zeros(16, 1, 2)}
            self.v = {0: torch.zeros(16, 1, 2), 1: torch.zeros(16, 1, 2)}

        def get_key_buffer(self, layer_id):
            return self.k[layer_id]

        def get_value_buffer(self, layer_id):
            return self.v[layer_id]

    pool = Pool()

    class Batch:
        token_to_kv_pool = pool

    img_locs = torch.tensor([3, 5, 8, 9], dtype=torch.long)
    donor_k0 = torch.arange(n * 2, dtype=torch.float).reshape(n, 1, 2)
    donor_v0 = donor_k0 + 100
    donor_k1 = donor_k0 + 1000
    donor_v1 = donor_v0 + 1000
    entry = cb.get_donor_store().get_or_create_donor(
        ("g",), n_image_tokens=n, grid_sig=(1, 1, n), positions=None
    )
    entry.record_layer(0, donor_k0, donor_v0)
    entry.record_layer(1, donor_k1, donor_v1)
    cb.get_donor_store().mark_complete(("g",))

    plan = cb.RecipientKVBlendPlan(
        request_id="rk2",
        group_key=("g",),
        img_locs=img_locs,
        positions=None,
        recompute_mask=torch.ones(n, dtype=torch.bool),  # pending
        grid_sig=(1, 1, n),
        n_image_tokens=n,
        reused_tokens=0,
        recomputed_tokens=n,
        pos_mode="same",
        select_mode="kvdev",
        bootstrap_layer_id=0,
        pending_deviation=True,
    )
    cb.set_recipient_blend_plans((plan,))

    # recipient's own layer-0 K: tokens 1 and 2 deviate most from the donor.
    rec_k0 = donor_k0.clone()
    rec_k0[1] += 50.0
    rec_k0[2] += 40.0
    pool.k[0][img_locs] = rec_k0
    pool.v[0][img_locs] = donor_v0

    reused0 = cb.apply_recipient_kv_blend_for_layer(forward_batch=Batch(), layer_id=0)
    assert reused0 == 2, "bootstrap layer must finalize AND reuse the 2 low-dev tokens"
    assert plan.pending_deviation is False
    assert int(plan.recompute_mask.sum()) == 2
    assert bool(plan.recompute_mask[1]) and bool(plan.recompute_mask[2])
    # deviation was measured before the overwrite, so the high-dev tokens (1,2) are the
    # recompute set and keep the recipient's own K; low-dev tokens (0,3) take donor K.
    assert torch.equal(pool.k[0][img_locs[0]], donor_k0[0])
    assert torch.equal(pool.k[0][img_locs[3]], donor_k0[3])
    assert torch.equal(pool.k[0][img_locs[1]], rec_k0[1])
    assert torch.equal(pool.k[0][img_locs[2]], rec_k0[2])

    reused1 = cb.apply_recipient_kv_blend_for_layer(forward_batch=Batch(), layer_id=1)
    assert reused1 == 2, "next layer must reuse the 2 low-deviation tokens"
    # low-deviation tokens 0 and 3 take donor layer-1 K; recompute tokens 1,2 untouched.
    assert torch.equal(pool.k[1][img_locs[0]], donor_k1[0])
    assert torch.equal(pool.k[1][img_locs[3]], donor_k1[3])
    assert torch.equal(pool.k[1][img_locs[1]], torch.zeros(1, 2))
    assert torch.equal(pool.k[1][img_locs[2]], torch.zeros(1, 2))
    cb.clear_recipient_blend_plans()
    print("ok test_apply_kvdev_bootstrap_finalizes_then_reuses_next_layer")


def test_sparse_decode_plan_register_and_build_batch():
    """Sparse decode: register reuse drop locs, then shorten decode page table."""
    os.environ["SGLANG_VLM_CACHEBLEND"] = "1"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE"] = "1"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_RECENT"] = "2"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_FIRST"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROPPED_TOKENS"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROP_RATIO"] = "0"
    cb.reload_config_from_env()
    assert cb.sparse_decode_enabled()

    # Sequence of 8 token locs; image tokens at positions 1,2,3 with pool locs 101,102,103.
    # Recompute only token 2 => drop 101 and 103.
    img_locs = torch.tensor([101, 102, 103], dtype=torch.long)
    recompute_mask = torch.tensor([False, True, False])
    plan = cb.RecipientKVBlendPlan(
        request_id="reqA",
        group_key=("g",),
        img_locs=img_locs,
        positions=None,
        recompute_mask=recompute_mask,
        grid_sig=(1, 1, 1),
        n_image_tokens=3,
        reused_tokens=2,
        recomputed_tokens=1,
        pos_mode="same",
        select_mode="kvdev",
        pending_deviation=False,
    )
    sparse = cb.register_sparse_decode_plan_from_blend(plan, req_pool_idx=7)
    assert sparse is not None
    assert sparse.n_drop_tokens == 2
    assert set(sparse.drop_locs.tolist()) == {101, 103}

    page_table = torch.tensor(
        [[10, 101, 102, 103, 20, 21, 22, 23]], dtype=torch.int32
    )
    cache_seqlens = torch.tensor([8], dtype=torch.int32)

    class Batch:
        req_pool_indices = torch.tensor([7], dtype=torch.long)

    batch = cb.build_sparse_decode_batch(
        Batch(),
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        page_size=1,
    )
    assert batch is not None
    # keep_recent=2 keeps last two (22,23). Drop 101,103 from earlier context.
    # Kept: 10, 102, 20, 21, 22, 23  (dropped 101,103)
    assert int(batch.cache_seqlens[0]) == 6
    assert batch.dropped_tokens == 2
    assert batch.kept_tokens == 6
    kept = batch.page_table[0, :6].tolist()
    assert 101 not in kept and 103 not in kept
    assert kept[-2:] == [22, 23]

    # page_size != 1 must fall back
    assert (
        cb.build_sparse_decode_batch(
            Batch(),
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            page_size=16,
        )
        is None
    )

    cb.clear_sparse_decode_plan(7)
    assert cb.get_sparse_decode_store().get(7) is None
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE"] = "0"
    cb.reload_config_from_env()
    print("ok test_sparse_decode_plan_register_and_build_batch")


def test_sparse_decode_keep_recent_protects_tail():
    os.environ["SGLANG_VLM_CACHEBLEND"] = "1"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE"] = "1"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_RECENT"] = "4"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_FIRST"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROPPED_TOKENS"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROP_RATIO"] = "0"
    cb.reload_config_from_env()

    # All tokens are drop candidates, but keep_recent=4 must retain the tail.
    img_locs = torch.arange(8, dtype=torch.long)
    recompute_mask = torch.zeros(8, dtype=torch.bool)
    plan = cb.RecipientKVBlendPlan(
        request_id="reqB",
        group_key=("g",),
        img_locs=img_locs,
        positions=None,
        recompute_mask=recompute_mask,
        grid_sig=(1, 1, 1),
        n_image_tokens=8,
        reused_tokens=8,
        recomputed_tokens=0,
        pos_mode="same",
        select_mode="kvdev",
    )
    cb.register_sparse_decode_plan_from_blend(plan, req_pool_idx=3)
    page_table = torch.arange(8, dtype=torch.int32).unsqueeze(0)
    cache_seqlens = torch.tensor([8], dtype=torch.int32)

    class Batch:
        req_pool_indices = torch.tensor([3], dtype=torch.long)

    batch = cb.build_sparse_decode_batch(
        Batch(),
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        page_size=1,
    )
    assert batch is not None
    assert int(batch.cache_seqlens[0]) == 4
    assert batch.page_table[0, :4].tolist() == [4, 5, 6, 7]
    assert batch.dropped_tokens == 4
    cb.get_sparse_decode_store().clear()
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE"] = "0"
    cb.reload_config_from_env()
    print("ok test_sparse_decode_keep_recent_protects_tail")


def test_sparse_decode_keep_first_and_min_drop_gate():
    os.environ["SGLANG_VLM_CACHEBLEND"] = "1"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE"] = "1"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_RECENT"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_FIRST"] = "2"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROPPED_TOKENS"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROP_RATIO"] = "0"
    cb.reload_config_from_env()

    plan = cb.RecipientKVBlendPlan(
        request_id="reqD",
        group_key=("g",),
        img_locs=torch.arange(6, dtype=torch.long),
        positions=None,
        recompute_mask=torch.zeros(6, dtype=torch.bool),
        grid_sig=(1, 1, 1),
        n_image_tokens=6,
        reused_tokens=6,
        recomputed_tokens=0,
        pos_mode="same",
        select_mode="kvdev",
    )
    cb.register_sparse_decode_plan_from_blend(plan, req_pool_idx=4)
    page_table = torch.arange(6, dtype=torch.int32).unsqueeze(0)
    cache_seqlens = torch.tensor([6], dtype=torch.int32)

    class Batch:
        req_pool_indices = torch.tensor([4], dtype=torch.long)

    batch = cb.build_sparse_decode_batch(
        Batch(),
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        page_size=1,
    )
    assert batch is not None
    assert batch.page_table[0, :2].tolist() == [0, 1]
    assert int(batch.cache_seqlens[0]) == 2
    assert batch.dropped_tokens == 4

    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROPPED_TOKENS"] = "5"
    cb.reload_config_from_env()
    cb.register_sparse_decode_plan_from_blend(plan, req_pool_idx=4)
    assert (
        cb.build_sparse_decode_batch(
            Batch(),
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            page_size=1,
        )
        is None
    )
    cb.get_sparse_decode_store().clear()
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE"] = "0"
    cb.reload_config_from_env()
    print("ok test_sparse_decode_keep_first_and_min_drop_gate")


def test_sparse_decode_batched_rows_keep_request_isolation():
    os.environ["SGLANG_VLM_CACHEBLEND"] = "1"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE"] = "1"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROPPED_TOKENS"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROP_RATIO"] = "0"
    cb.reload_config_from_env()
    cb.get_sparse_decode_store().clear()

    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_RECENT"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_FIRST"] = "0"
    cb.reload_config_from_env()
    plan_a = cb.RecipientKVBlendPlan(
        request_id="req_iso_a",
        group_key=("g",),
        img_locs=torch.tensor([101, 102, 103], dtype=torch.long),
        positions=None,
        recompute_mask=torch.tensor([False, True, False]),
        grid_sig=(1, 1, 1),
        n_image_tokens=3,
        reused_tokens=2,
        recomputed_tokens=1,
        pos_mode="same",
        select_mode="kvdev",
    )
    cb.register_sparse_decode_plan_from_blend(plan_a, req_pool_idx=10)

    plan_b = cb.RecipientKVBlendPlan(
        request_id="req_iso_b",
        group_key=("g",),
        img_locs=torch.tensor([201, 202, 204], dtype=torch.long),
        positions=None,
        recompute_mask=torch.tensor([True, False, False]),
        grid_sig=(1, 1, 1),
        n_image_tokens=3,
        reused_tokens=2,
        recomputed_tokens=1,
        pos_mode="same",
        select_mode="kvdev",
    )
    cb.register_sparse_decode_plan_from_blend(plan_b, req_pool_idx=20)

    page_table = torch.tensor(
        [
            [1, 101, 102, 103, 9],
            [201, 202, 203, 204, 205],
            [101, 301, 302, 303, 0],
        ],
        dtype=torch.int32,
    )
    cache_seqlens = torch.tensor([5, 5, 4], dtype=torch.int32)

    class Batch:
        req_pool_indices = torch.tensor([10, 20, 30], dtype=torch.long)

    batch = cb.build_sparse_decode_batch(
        Batch(),
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        page_size=1,
    )
    assert batch is not None
    assert batch.dropped_tokens == 4
    assert batch.used_requests == 2
    assert batch.cache_seqlens.tolist() == [3, 3, 4]
    assert batch.page_table[0, :3].tolist() == [1, 102, 9]
    assert batch.page_table[1, :3].tolist() == [201, 203, 205]
    # Row 2 has no sparse plan. Its 101 must not be dropped just because row 0 drops 101.
    assert batch.page_table[2, :4].tolist() == [101, 301, 302, 303]

    cb.get_sparse_decode_store().clear()
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE"] = "0"
    cb.reload_config_from_env()
    print("ok test_sparse_decode_batched_rows_keep_request_isolation")


def test_sparse_decode_disabled_is_noop():
    os.environ["SGLANG_VLM_CACHEBLEND"] = "1"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE"] = "0"
    cb.reload_config_from_env()
    assert not cb.sparse_decode_enabled()
    plan = cb.RecipientKVBlendPlan(
        request_id="reqC",
        group_key=("g",),
        img_locs=torch.tensor([1, 2]),
        positions=None,
        recompute_mask=torch.tensor([False, False]),
        grid_sig=(1, 1, 1),
        n_image_tokens=2,
        reused_tokens=2,
        recomputed_tokens=0,
        pos_mode="same",
        select_mode="kvdev",
    )
    assert cb.register_sparse_decode_plan_from_blend(plan, req_pool_idx=1) is None
    print("ok test_sparse_decode_disabled_is_noop")


def test_sparse_decode_stats_to_dict():
    stats = cb.CacheBlendStats(
        sparse_decode_used=True,
        sparse_decode_kept_tokens=12,
        sparse_decode_dropped_tokens=4,
    )
    d = stats.to_dict()
    assert d["cacheblend_sparse_decode_used"] == "1"
    assert d["cacheblend_sparse_decode_kept_tokens"] == "12"
    assert d["cacheblend_sparse_decode_dropped_tokens"] == "4"
    print("ok test_sparse_decode_stats_to_dict")


def test_sparse_decode_min_drop_gate_bails_on_upper_bound():
    """[P0.5-a] The min-drop gate bails using an upper bound before heavy tensor work,
    and the decision matches the exact computation (bail iff exact would also bail)."""
    os.environ["SGLANG_VLM_CACHEBLEND"] = "1"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE"] = "1"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_RECENT"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_FIRST"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROPPED_TOKENS"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROP_RATIO"] = "0"
    cb.reload_config_from_env()

    page_table = torch.tensor(
        [[10, 101, 102, 103, 20, 21, 22, 23]], dtype=torch.int32
    )
    cache_seqlens = torch.tensor([8], dtype=torch.int32)

    class Batch:
        req_pool_indices = torch.tensor([50], dtype=torch.long)

    def build_with_gate(min_ratio, min_dropped):
        # reload_config_from_env resets the sparse-decode store, so register the plan
        # after configuring gates. drop {101,103} => n_drop=2, seqlen 8 => bound 0.25.
        os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROP_RATIO"] = str(min_ratio)
        os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROPPED_TOKENS"] = str(
            min_dropped
        )
        cb.reload_config_from_env()
        cb.get_sparse_decode_store().clear()
        plan = cb.RecipientKVBlendPlan(
            request_id="reqUB",
            group_key=("g",),
            img_locs=torch.tensor([101, 102, 103], dtype=torch.long),
            positions=None,
            recompute_mask=torch.tensor([False, True, False]),
            grid_sig=(1, 1, 1),
            n_image_tokens=3,
            reused_tokens=2,
            recomputed_tokens=1,
            pos_mode="same",
            select_mode="kvdev",
        )
        cb.register_sparse_decode_plan_from_blend(plan, req_pool_idx=50)
        return cb.build_sparse_decode_batch(
            Batch(), page_table=page_table, cache_seqlens=cache_seqlens, page_size=1
        )

    # ratio gate above the 0.25 upper bound -> bail (None).
    assert build_with_gate(0.5, 0) is None
    # ratio gate below the upper bound and below the exact 0.25 ratio -> keep.
    b = build_with_gate(0.2, 0)
    assert b is not None and b.dropped_tokens == 2
    # dropped-count gate above the upper bound of 2 -> bail (None).
    assert build_with_gate(0, 3) is None
    # dropped-count gate at the exact bound -> keep.
    b = build_with_gate(0, 2)
    assert b is not None and b.dropped_tokens == 2

    cb.get_sparse_decode_store().clear()
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROPPED_TOKENS"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE"] = "0"
    cb.reload_config_from_env()
    print("ok test_sparse_decode_min_drop_gate_bails_on_upper_bound")


def test_sparse_decode_candidate_restriction_mixed_batch():
    """[P0.5-b] Candidate-restricted isin scatters back correctly across a zero-length
    row, a non-candidate row, and a keep_first-protected candidate row."""
    os.environ["SGLANG_VLM_CACHEBLEND"] = "1"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE"] = "1"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_RECENT"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_FIRST"] = "1"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROPPED_TOKENS"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROP_RATIO"] = "0"
    cb.reload_config_from_env()
    cb.get_sparse_decode_store().clear()

    # Candidate row: all three image tokens reusable, keep_first=1 protects col 0 (101).
    plan = cb.RecipientKVBlendPlan(
        request_id="reqMix",
        group_key=("g",),
        img_locs=torch.tensor([101, 102, 103], dtype=torch.long),
        positions=None,
        recompute_mask=torch.tensor([False, False, False]),
        grid_sig=(1, 1, 1),
        n_image_tokens=3,
        reused_tokens=3,
        recomputed_tokens=0,
        pos_mode="same",
        select_mode="kvdev",
    )
    cb.register_sparse_decode_plan_from_blend(plan, req_pool_idx=60)

    page_table = torch.tensor(
        [
            [101, 102, 103, 55, 66],  # req60 candidate: keep 101 (first), drop 102,103
            [201, 202, 0, 0, 0],      # req99 no plan: pass through whole
            [9, 9, 9, 9, 9],          # zero-length row: kept empty
        ],
        dtype=torch.int32,
    )
    cache_seqlens = torch.tensor([5, 2, 0], dtype=torch.int32)

    class Batch:
        req_pool_indices = torch.tensor([60, 99, 70], dtype=torch.long)

    batch = cb.build_sparse_decode_batch(
        Batch(), page_table=page_table, cache_seqlens=cache_seqlens, page_size=1
    )
    assert batch is not None
    assert batch.used_requests == 1
    assert batch.dropped_tokens == 2
    assert batch.cache_seqlens.tolist() == [3, 2, 0]
    assert batch.page_table[0, :3].tolist() == [101, 55, 66]
    assert batch.page_table[1, :2].tolist() == [201, 202]

    cb.get_sparse_decode_store().clear()
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_FIRST"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE"] = "0"
    cb.reload_config_from_env()
    print("ok test_sparse_decode_candidate_restriction_mixed_batch")


def _run_incremental_decode_sequence(
    *, full_row, steps, keep_recent, keep_first, incremental, req_pool_idx=7
):
    """Register one reuse plan (drop image slots 101 & 103) then build a sparse
    decode batch at each seqlen in ``steps``, simulating autoregressive decode over a
    fixed page-table row. Returns a comparable summary per step (or None when the
    build bails). Used to assert the incremental path matches the full recompute."""
    os.environ["SGLANG_VLM_CACHEBLEND"] = "1"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE"] = "1"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_RECENT"] = str(keep_recent)
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_FIRST"] = str(keep_first)
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROPPED_TOKENS"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROP_RATIO"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_INCREMENTAL"] = (
        "1" if incremental else "0"
    )
    cb.reload_config_from_env()
    cb.get_sparse_decode_store().clear()

    img_locs = torch.tensor([101, 102, 103], dtype=torch.long)
    recompute_mask = torch.tensor([False, True, False])  # reuse (=> drop) 101 & 103
    plan = cb.RecipientKVBlendPlan(
        request_id="reqInc",
        group_key=("g",),
        img_locs=img_locs,
        positions=None,
        recompute_mask=recompute_mask,
        grid_sig=(1, 1, 1),
        n_image_tokens=3,
        reused_tokens=2,
        recomputed_tokens=1,
        pos_mode="same",
        select_mode="kvdev",
        pending_deviation=False,
    )
    cb.register_sparse_decode_plan_from_blend(plan, req_pool_idx=req_pool_idx)

    outs = []
    for seq_len in steps:
        page_table = torch.tensor([full_row], dtype=torch.int32)
        cache_seqlens = torch.tensor([seq_len], dtype=torch.int32)

        class Batch:
            req_pool_indices = torch.tensor([req_pool_idx], dtype=torch.long)

        batch = cb.build_sparse_decode_batch(
            Batch(),
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            page_size=1,
        )
        if batch is None:
            outs.append(None)
        else:
            outs.append(
                (
                    batch.cache_seqlens.tolist(),
                    batch.page_table.tolist(),
                    int(batch.dropped_tokens),
                    int(batch.kept_tokens),
                    int(batch.used_requests),
                )
            )
    return outs


def test_sparse_decode_incremental_matches_full_recompute():
    """The incremental drop-column cache must produce a bit-identical batch to the
    full-context recompute at every decode step, including when keep_recent initially
    overlaps the drop columns and then slides past them."""
    row = [10, 101, 102, 103, 20, 21, 30, 31, 32, 33]

    # Case A: steady state, recent window never reaches the drop columns.
    a_inc = _run_incremental_decode_sequence(
        full_row=row, steps=[6, 7, 8, 9, 10], keep_recent=2, keep_first=1,
        incremental=True,
    )
    a_ref = _run_incremental_decode_sequence(
        full_row=row, steps=[6, 7, 8, 9, 10], keep_recent=2, keep_first=1,
        incremental=False,
    )
    assert a_inc == a_ref, f"case A mismatch:\n{a_inc}\n{a_ref}"

    # Case B: large recent window protects the drops early (build bails -> None), then
    # the window slides forward and the drops re-appear.
    b_inc = _run_incremental_decode_sequence(
        full_row=row, steps=[4, 5, 6, 7, 8, 9, 10], keep_recent=6, keep_first=0,
        incremental=True,
    )
    b_ref = _run_incremental_decode_sequence(
        full_row=row, steps=[4, 5, 6, 7, 8, 9, 10], keep_recent=6, keep_first=0,
        incremental=False,
    )
    assert b_inc == b_ref, f"case B mismatch:\n{b_inc}\n{b_ref}"
    # Sanity: case B actually exercises both a bail and a real drop.
    assert None in b_inc and any(o is not None for o in b_inc)

    cb.get_sparse_decode_store().clear()
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_INCREMENTAL"] = "1"
    cb.reload_config_from_env()
    print("ok test_sparse_decode_incremental_matches_full_recompute")


def test_sparse_decode_incremental_resets_on_seqlen_regression():
    """If the same plan object is reused but the sequence length regresses (e.g. an
    unexpected pool-index reuse), the incremental cache must reset and recompute
    membership against the new page-table row rather than reuse stale columns."""
    os.environ["SGLANG_VLM_CACHEBLEND"] = "1"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE"] = "1"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_RECENT"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_FIRST"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROPPED_TOKENS"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROP_RATIO"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_INCREMENTAL"] = "1"
    cb.reload_config_from_env()
    cb.get_sparse_decode_store().clear()

    img_locs = torch.tensor([101, 102, 103], dtype=torch.long)
    recompute_mask = torch.tensor([False, True, False])  # drop 101 & 103
    plan = cb.RecipientKVBlendPlan(
        request_id="reqReset",
        group_key=("g",),
        img_locs=img_locs,
        positions=None,
        recompute_mask=recompute_mask,
        grid_sig=(1, 1, 1),
        n_image_tokens=3,
        reused_tokens=2,
        recomputed_tokens=1,
        pos_mode="same",
        select_mode="kvdev",
        pending_deviation=False,
    )
    cb.register_sparse_decode_plan_from_blend(plan, req_pool_idx=5)

    class BatchA:
        req_pool_indices = torch.tensor([5], dtype=torch.long)

    # Grow to len 8 with 101 & 103 at cols 1 & 3 -> cache columns {1, 3}, seen=8.
    grow = cb.build_sparse_decode_batch(
        BatchA(),
        page_table=torch.tensor([[10, 101, 102, 103, 20, 21, 22, 23]], dtype=torch.int32),
        cache_seqlens=torch.tensor([8], dtype=torch.int32),
        page_size=1,
    )
    assert grow is not None and grow.dropped_tokens == 2

    # Regress to len 4 with a DIFFERENT row: 101 now sits at col 2, 103 absent. The
    # cache must reset (seen 8 > 4) and recompute -> drop only col 2.
    regress_row = torch.tensor([[10, 55, 101, 66, 77, 88, 99, 44]], dtype=torch.int32)
    regress = cb.build_sparse_decode_batch(
        BatchA(),
        page_table=regress_row,
        cache_seqlens=torch.tensor([4], dtype=torch.int32),
        page_size=1,
    )
    assert regress is not None
    assert regress.dropped_tokens == 1, regress.dropped_tokens
    kept = regress.page_table[0, : int(regress.cache_seqlens[0])].tolist()
    assert 101 not in kept and kept == [10, 55, 66], kept

    cb.get_sparse_decode_store().clear()
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE"] = "0"
    cb.reload_config_from_env()
    print("ok test_sparse_decode_incremental_resets_on_seqlen_regression")


def test_sparse_decode_incremental_same_seqlen_rewrite_resets():
    """If the page-table row is rewritten under an unchanged seqlen (slot relocation
    from preemption / replay / KV compaction / pool reuse), the anchor-value check must
    invalidate the cache so the drop mask matches a fresh full recompute instead of
    dropping stale columns."""
    os.environ["SGLANG_VLM_CACHEBLEND"] = "1"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE"] = "1"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_RECENT"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_KEEP_FIRST"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROPPED_TOKENS"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_MIN_DROP_RATIO"] = "0"
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE_INCREMENTAL"] = "1"
    cb.reload_config_from_env()
    cb.get_sparse_decode_store().clear()

    img_locs = torch.tensor([101, 102, 103], dtype=torch.long)
    recompute_mask = torch.tensor([False, True, False])  # drop 101 & 103
    plan = cb.RecipientKVBlendPlan(
        request_id="reqRewrite",
        group_key=("g",),
        img_locs=img_locs,
        positions=None,
        recompute_mask=recompute_mask,
        grid_sig=(1, 1, 1),
        n_image_tokens=3,
        reused_tokens=2,
        recomputed_tokens=1,
        pos_mode="same",
        select_mode="kvdev",
        pending_deviation=False,
    )
    cb.register_sparse_decode_plan_from_blend(plan, req_pool_idx=9)

    class Batch:
        req_pool_indices = torch.tensor([9], dtype=torch.long)

    # First build: 101 & 103 present at cols 1 & 3 -> cache {1, 3} with anchor vals.
    first = cb.build_sparse_decode_batch(
        Batch(),
        page_table=torch.tensor([[10, 101, 20, 103, 30]], dtype=torch.int32),
        cache_seqlens=torch.tensor([5], dtype=torch.int32),
        page_size=1,
    )
    assert first is not None and first.dropped_tokens == 2

    # Same seqlen, rewritten row: cols 1 & 3 no longer hold drop slots. The anchor check
    # must reset the cache; nothing is droppable now -> build bails (None), exactly like
    # a full recompute would.
    rewritten = cb.build_sparse_decode_batch(
        Batch(),
        page_table=torch.tensor([[10, 55, 20, 66, 30]], dtype=torch.int32),
        cache_seqlens=torch.tensor([5], dtype=torch.int32),
        page_size=1,
    )
    assert rewritten is None, "stale cache dropped rewritten columns"

    # After the reset, normal append-only growth still detects a drop slot at a newly
    # appended position (col 5 holds 103).
    grow = cb.build_sparse_decode_batch(
        Batch(),
        page_table=torch.tensor([[10, 55, 20, 66, 30, 103]], dtype=torch.int32),
        cache_seqlens=torch.tensor([6], dtype=torch.int32),
        page_size=1,
    )
    assert grow is not None and grow.dropped_tokens == 1
    kept = grow.page_table[0, : int(grow.cache_seqlens[0])].tolist()
    assert kept == [10, 55, 20, 66, 30], kept

    cb.get_sparse_decode_store().clear()
    os.environ["SGLANG_VLM_CACHEBLEND_SPARSE_DECODE"] = "0"
    cb.reload_config_from_env()
    print("ok test_sparse_decode_incremental_same_seqlen_rewrite_resets")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()
