import asyncio

import pytest

from verl.experimental.rollout_reuse import (
    ArtifactIdentity,
    ExecutionAction,
    ReuseContext,
    ReuseRegistry,
    SharingScope,
    group_preserving_slices,
)


def _identity(**extra):
    return ArtifactIdentity.from_dependencies(
        operator_id="vision_encoder",
        representation_stage="vit_embedding",
        content_id="image:abc",
        dependencies={
            "processed_pixels": "abc",
            "processor": "proc-v1",
            "model": "model-v1",
            **extra,
        },
    )


def _context(branch: int, *, group: str = "group-a", epoch: str = "7"):
    return ReuseContext(
        group_id=group,
        policy_epoch=epoch,
        branch_id=str(branch),
        turn_id=1,
    )


@pytest.mark.asyncio
async def test_four_concurrent_branches_compute_once():
    registry = ReuseRegistry[int]()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return 42

    tasks = [
        asyncio.create_task(
            registry.get_or_compute(
                identity=_identity(),
                scope=SharingScope.GROUP,
                context=_context(branch),
                compute=compute,
            )
        )
        for branch in range(4)
    ]
    await started.wait()
    release.set()
    results = await asyncio.gather(*tasks)

    assert calls == 1
    assert [result.value for result in results] == [42] * 4
    assert sum(result.action is ExecutionAction.LOCAL for result in results) == 1
    assert sum(result.action is ExecutionAction.EXACT for result in results) == 3
    assert {result.artifact.producer.branch_id for result in results} == {"0"}


@pytest.mark.asyncio
async def test_equivalence_is_separate_from_group_scope():
    registry = ReuseRegistry[int]()
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        return calls

    first = await registry.get_or_compute(
        identity=_identity(),
        scope=SharingScope.GROUP,
        context=_context(0, group="a"),
        compute=compute,
    )
    second = await registry.get_or_compute(
        identity=_identity(),
        scope=SharingScope.GROUP,
        context=_context(0, group="b"),
        compute=compute,
    )
    cross_group = await registry.get_or_compute(
        identity=_identity(),
        scope=SharingScope.POLICY_EPOCH,
        context=_context(1, group="b"),
        compute=compute,
    )
    cross_group_hit = await registry.get_or_compute(
        identity=_identity(),
        scope=SharingScope.POLICY_EPOCH,
        context=_context(2, group="c"),
        compute=compute,
    )

    assert first.value == 1
    assert second.value == 2
    assert cross_group.value == 3
    assert cross_group_hit.value == 3
    assert cross_group_hit.action is ExecutionAction.EXACT


@pytest.mark.asyncio
async def test_dependency_or_epoch_change_misses():
    registry = ReuseRegistry[str]()
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        return f"value-{calls}"

    values = []
    for identity, context in (
        (_identity(), _context(0, epoch="1")),
        (_identity(processor="proc-v2"), _context(1, epoch="1")),
        (_identity(), _context(2, epoch="2")),
    ):
        result = await registry.get_or_compute(
            identity=identity,
            scope=SharingScope.GROUP,
            context=context,
            compute=compute,
        )
        values.append(result.value)

    assert values == ["value-1", "value-2", "value-3"]


@pytest.mark.asyncio
async def test_failure_is_not_cached_and_waiters_recover():
    registry = ReuseRegistry[int]()
    calls = 0

    async def fail():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        raise RuntimeError("producer failed")

    results = await asyncio.gather(
        *[
            registry.get_or_compute(
                identity=_identity(),
                scope=SharingScope.GROUP,
                context=_context(branch),
                compute=fail,
            )
            for branch in range(4)
        ],
        return_exceptions=True,
    )
    assert calls == 1
    assert all(isinstance(result, RuntimeError) for result in results)

    recovered = await registry.get_or_compute(
        identity=_identity(),
        scope=SharingScope.GROUP,
        context=_context(1),
        compute=lambda: 9,
    )
    assert recovered.value == 9
    assert recovered.action is ExecutionAction.LOCAL


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_shared_producer():
    registry = ReuseRegistry[int]()
    started = asyncio.Event()
    release = asyncio.Event()

    async def compute():
        started.set()
        await release.wait()
        return 5

    producer_caller = asyncio.create_task(
        registry.get_or_compute(
            identity=_identity(),
            scope=SharingScope.GROUP,
            context=_context(0),
            compute=compute,
        )
    )
    await started.wait()
    waiter = asyncio.create_task(
        registry.get_or_compute(
            identity=_identity(),
            scope=SharingScope.GROUP,
            context=_context(1),
            compute=compute,
        )
    )
    producer_caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await producer_caller
    release.set()

    result = await waiter
    assert result.value == 5
    assert result.action is ExecutionAction.EXACT
    stats = await registry.stats()
    assert stats["caller_cancellations"] == 1
    assert stats["inflight"] == 0


@pytest.mark.asyncio
async def test_cancelled_inflight_owner_is_removed_and_not_cached():
    registry = ReuseRegistry[int]()
    started = asyncio.Event()

    async def compute():
        started.set()
        await asyncio.Event().wait()

    request = asyncio.create_task(
        registry.get_or_compute(
            identity=_identity(),
            scope=SharingScope.GROUP,
            context=_context(0),
            compute=compute,
        )
    )
    await started.wait()
    await registry.clear(cancel_inflight=True)
    with pytest.raises(asyncio.CancelledError):
        await request

    recovered = await registry.get_or_compute(
        identity=_identity(),
        scope=SharingScope.GROUP,
        context=_context(1),
        compute=lambda: 11,
    )
    assert recovered.value == 11
    assert (await registry.stats())["inflight"] == 0


def test_group_preserving_slices_never_split_a_group():
    groups = ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4
    slices = group_preserving_slices(groups, 3)

    parts = [groups[group_slice] for group_slice in slices]
    assert [item for part in parts for item in part] == groups
    owners = {}
    for partition_index, part in enumerate(parts):
        for group_id in part:
            owners.setdefault(group_id, set()).add(partition_index)
    assert all(len(partitions) == 1 for partitions in owners.values())


def test_group_preserving_slices_reject_non_contiguous_groups():
    with pytest.raises(ValueError, match="contiguous"):
        group_preserving_slices(["a", "b", "a"], 2)
