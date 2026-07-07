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


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()
