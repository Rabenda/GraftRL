from __future__ import annotations

import pytest

from sglang.srt.mem_cache.rollout_reuse_backend import BackendExecutionDecision


def test_backend_decision_emits_versioned_portable_fields():
    decision = BackendExecutionDecision(
        action="skip",
        operator_id="sparse_decode.context_attention",
        representation_stage="decode_kv_blocks",
        reason="query_ranked_context_blocks",
        eligible_units=100,
        applied_units=50,
        approximate=True,
        policy="query-block-mass-prefill-v2",
    )
    fields = decision.event_fields()
    assert fields["reuse_action_schema"] == "rollout-reuse-action-v1"
    assert fields["reuse_action"] == "skip"
    assert fields["reuse_applied_units"] == "50"
    assert fields["reuse_approximate"] == "1"


def test_backend_decision_rejects_false_exact_and_local_savings():
    with pytest.raises(ValueError, match="EXACT"):
        BackendExecutionDecision(
            action="exact",
            operator_id="op",
            representation_stage="stage",
            reason="bad",
            eligible_units=2,
            applied_units=1,
        )
    with pytest.raises(ValueError, match="LOCAL"):
        BackendExecutionDecision(
            action="local",
            operator_id="op",
            representation_stage="stage",
            reason="bad",
            eligible_units=2,
            applied_units=1,
        )
