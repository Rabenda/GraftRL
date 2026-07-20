from __future__ import annotations

import pytest

from verl.experimental.rollout_reuse import (
    ArtifactIdentity,
    ExecutionAction,
    ExecutionDecision,
    ReuseContext,
    RolloutReuseRuntime,
    SharingScope,
)


def test_execution_decision_enforces_exactness_and_unit_accounting():
    with pytest.raises(ValueError, match="EXACT"):
        ExecutionDecision(
            action=ExecutionAction.EXACT,
            operator_id="op",
            representation_stage="stage",
            reason="bad",
            eligible_units=1,
            applied_units=1,
            approximate=True,
        )
    with pytest.raises(ValueError, match="exceed"):
        ExecutionDecision(
            action=ExecutionAction.SKIP,
            operator_id="op",
            representation_stage="stage",
            reason="bad",
            eligible_units=1,
            applied_units=2,
            approximate=True,
        )
    with pytest.raises(ValueError, match="every eligible"):
        ExecutionDecision(
            action=ExecutionAction.EXACT,
            operator_id="op",
            representation_stage="stage",
            reason="bad",
            eligible_units=2,
            applied_units=1,
        )


@pytest.mark.asyncio
async def test_runtime_unifies_exact_local_and_approximate_skip_trace():
    runtime: RolloutReuseRuntime[int] = RolloutReuseRuntime()
    identity = ArtifactIdentity.from_dependencies(
        operator_id="search",
        representation_stage="tool_result",
        content_id="query:a",
        dependencies={"query": "a"},
    )
    context = ReuseContext(
        group_id="g", policy_epoch="1", branch_id="0", turn_id=1
    )
    local = await runtime.get_or_compute(
        identity=identity,
        scope=SharingScope.GROUP,
        context=context,
        compute=lambda: 7,
    )
    exact = await runtime.get_or_compute(
        identity=identity,
        scope=SharingScope.GROUP,
        context=context,
        compute=lambda: 9,
    )
    skip = runtime.decide(
        action=ExecutionAction.SKIP,
        operator_id="decode_attention",
        representation_stage="decode_kv_blocks",
        reason="query_ranked_context_blocks",
        eligible_units=16,
        applied_units=8,
        approximate=True,
        policy="query-block-mass-prefill-v2",
    )

    assert local.action is ExecutionAction.LOCAL
    assert exact.action is ExecutionAction.EXACT
    assert skip.event_fields("decode")["decode_action"] == "skip"
    snapshot = runtime.snapshot()
    assert snapshot["counts"] == {"local": 1, "exact": 1, "skip": 1}
    assert snapshot["applied_units"] == {"local": 0, "exact": 1, "skip": 8}


def test_runtime_ingests_backend_wire_decision_with_the_same_contract():
    runtime: RolloutReuseRuntime[object] = RolloutReuseRuntime()
    backend_fields = {
        "reuse_action_schema": "rollout-reuse-action-v1",
        "reuse_action": "partial",
        "reuse_operator_id": "vlm_cacheblend.prefill",
        "reuse_representation_stage": "llm_prefill_kv",
        "reuse_reason": "recipient_kv_graft",
        "reuse_eligible_units": "128",
        "reuse_applied_units": "96",
        "reuse_approximate": "1",
        "reuse_error_bound": "",
        "reuse_policy": "kvdev",
    }
    decision = runtime.record_event_fields(backend_fields)

    assert decision.action is ExecutionAction.PARTIAL
    assert decision.applied_units == 96
    assert decision.approximate
    assert runtime.snapshot()["counts"] == {"partial": 1}


def test_runtime_rejects_unknown_backend_schema():
    runtime: RolloutReuseRuntime[object] = RolloutReuseRuntime()
    with pytest.raises(ValueError, match="unsupported action schema"):
        runtime.record_event_fields(
            {
                "reuse_action_schema": "unknown",
                "reuse_action": "local",
            }
        )
