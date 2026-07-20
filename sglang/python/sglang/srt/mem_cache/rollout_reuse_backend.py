"""Portable SGLang adapter for the rollout reuse action contract.

The agent control plane and the inference backend are separate Python processes and
packages.  This module keeps backend execution mechanics independent while forcing
every emitted decision through the same versioned EXACT/PARTIAL/SKIP/LOCAL schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ACTION_SCHEMA_VERSION = "rollout-reuse-action-v1"
VALID_ACTIONS = frozenset(("exact", "partial", "skip", "local"))


@dataclass(frozen=True)
class BackendExecutionDecision:
    action: str
    operator_id: str
    representation_stage: str
    reason: str
    eligible_units: int = 0
    applied_units: int = 0
    approximate: bool = False
    error_bound: float | None = None
    policy: str = ""

    def __post_init__(self) -> None:
        if self.action not in VALID_ACTIONS:
            raise ValueError(f"unsupported execution action: {self.action!r}")
        if not self.operator_id or not self.representation_stage or not self.reason:
            raise ValueError("operator, representation stage, and reason are required")
        if self.eligible_units < 0 or self.applied_units < 0:
            raise ValueError("decision unit counts must be non-negative")
        if self.applied_units > self.eligible_units:
            raise ValueError("applied_units cannot exceed eligible_units")
        if self.action == "exact" and self.approximate:
            raise ValueError("EXACT decisions cannot be approximate")
        if self.action == "exact" and self.applied_units != self.eligible_units:
            raise ValueError("EXACT decisions must apply every eligible unit")
        if self.action == "local" and self.applied_units:
            raise ValueError("LOCAL decisions cannot report reused/skipped units")
        if self.error_bound is not None and self.error_bound < 0:
            raise ValueError("error_bound must be non-negative")
        if self.error_bound is not None and not self.approximate:
            raise ValueError("error_bound is only valid for approximate decisions")

    def event_fields(self, prefix: str = "reuse") -> dict[str, Any]:
        return {
            f"{prefix}_action_schema": ACTION_SCHEMA_VERSION,
            f"{prefix}_action": self.action,
            f"{prefix}_operator_id": self.operator_id,
            f"{prefix}_representation_stage": self.representation_stage,
            f"{prefix}_reason": self.reason,
            f"{prefix}_eligible_units": str(self.eligible_units),
            f"{prefix}_applied_units": str(self.applied_units),
            f"{prefix}_approximate": "1" if self.approximate else "0",
            f"{prefix}_error_bound": (
                "" if self.error_bound is None else str(self.error_bound)
            ),
            f"{prefix}_policy": self.policy,
        }
