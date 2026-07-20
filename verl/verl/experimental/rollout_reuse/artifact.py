"""Artifact identity and execution decisions for rollout reuse.

Equivalence and sharing permission are deliberately separate:

* :class:`ArtifactIdentity` describes whether two operator executions compute
  the same value.
* :class:`SharingScope` describes where an equivalent value may be shared.

In particular, ``group_id`` and ``policy_epoch`` never alter the dependency
digest.  They only select a registry namespace.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Mapping, TypeVar


T = TypeVar("T")


class ExecutionAction(str, Enum):
    EXACT = "exact"
    PARTIAL = "partial"
    SKIP = "skip"
    LOCAL = "local"


class SharingScope(str, Enum):
    """Permission/lifetime namespace for an otherwise equivalent artifact."""

    GROUP = "group"
    POLICY_EPOCH = "policy_epoch"
    GLOBAL = "global"


ACTION_SCHEMA_VERSION = "rollout-reuse-action-v1"


@dataclass(frozen=True)
class ExecutionDecision:
    """One runtime decision under the EXACT/PARTIAL/SKIP/LOCAL contract.

    The action and its accounting live together so a backend cannot report saved
    work without also declaring whether it reused an equivalent artifact or made an
    approximate skip.  ``eligible_units`` is the work considered by the policy and
    ``applied_units`` is the subset actually reused/skipped.
    """

    action: ExecutionAction
    operator_id: str
    representation_stage: str
    reason: str
    eligible_units: int = 0
    applied_units: int = 0
    approximate: bool = False
    error_bound: float | None = None
    policy: str = ""

    def __post_init__(self) -> None:
        if not self.operator_id or not self.representation_stage:
            raise ValueError("operator_id and representation_stage are required")
        if not self.reason:
            raise ValueError("decision reason is required")
        if self.eligible_units < 0 or self.applied_units < 0:
            raise ValueError("decision unit counts must be non-negative")
        if self.applied_units > self.eligible_units:
            raise ValueError("applied_units cannot exceed eligible_units")
        if self.action is ExecutionAction.EXACT and self.approximate:
            raise ValueError("EXACT decisions cannot be approximate")
        if (
            self.action is ExecutionAction.EXACT
            and self.applied_units != self.eligible_units
        ):
            raise ValueError("EXACT decisions must apply every eligible unit")
        if self.action is ExecutionAction.LOCAL and self.applied_units:
            raise ValueError("LOCAL decisions cannot report reused/skipped units")
        if self.error_bound is not None and self.error_bound < 0:
            raise ValueError("error_bound must be non-negative")
        if self.error_bound is not None and not self.approximate:
            raise ValueError("error_bound is only valid for approximate decisions")

    def event_fields(self, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_action_schema": ACTION_SCHEMA_VERSION,
            f"{prefix}_action": self.action.value,
            f"{prefix}_operator_id": self.operator_id,
            f"{prefix}_representation_stage": self.representation_stage,
            f"{prefix}_reason": self.reason,
            f"{prefix}_eligible_units": self.eligible_units,
            f"{prefix}_applied_units": self.applied_units,
            f"{prefix}_approximate": self.approximate,
            f"{prefix}_error_bound": self.error_bound,
            f"{prefix}_policy": self.policy,
        }

    @classmethod
    def from_event_fields(
        cls, fields: Mapping[str, Any], prefix: str
    ) -> "ExecutionDecision":
        """Decode the portable event schema emitted by a remote backend.

        SGLang runs in a separate process and deliberately does not import the verl
        control plane.  This parser is the adapter boundary that turns its wire fields
        back into the same validated decision used by agent-side operators.
        """

        def value(name: str, default: Any = None) -> Any:
            return fields.get(f"{prefix}_{name}", default)

        schema = str(value("action_schema", ""))
        if schema != ACTION_SCHEMA_VERSION:
            raise ValueError(f"unsupported action schema: {schema!r}")

        approximate_raw = value("approximate", False)
        if isinstance(approximate_raw, str):
            approximate = approximate_raw.strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        else:
            approximate = bool(approximate_raw)
        error_bound_raw = value("error_bound")
        error_bound = (
            None
            if error_bound_raw in (None, "")
            else float(error_bound_raw)
        )
        return cls(
            action=ExecutionAction(str(value("action"))),
            operator_id=str(value("operator_id", "")),
            representation_stage=str(value("representation_stage", "")),
            reason=str(value("reason", "")),
            eligible_units=int(value("eligible_units", 0)),
            applied_units=int(value("applied_units", 0)),
            approximate=approximate,
            error_bound=error_bound,
            policy=str(value("policy", "")),
        )


def _canonicalize(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _canonicalize(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest(), "nbytes": len(value)}
    raise TypeError(
        f"Dependency values must be deterministic JSON-like data, got {type(value).__name__}"
    )


def dependency_digest(dependencies: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        _canonicalize(dependencies),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(b"rollout-reuse-dependencies-v1\0" + canonical).hexdigest()


@dataclass(frozen=True)
class ArtifactIdentity:
    """Mathematical identity of one context-derived operator output."""

    operator_id: str
    representation_stage: str
    content_id: str
    dependency_signature: str

    @classmethod
    def from_dependencies(
        cls,
        *,
        operator_id: str,
        representation_stage: str,
        content_id: str,
        dependencies: Mapping[str, Any],
    ) -> "ArtifactIdentity":
        if not operator_id or not representation_stage or not content_id:
            raise ValueError("operator_id, representation_stage and content_id are required")
        return cls(
            operator_id=str(operator_id),
            representation_stage=str(representation_stage),
            content_id=str(content_id),
            dependency_signature=dependency_digest(dependencies),
        )


@dataclass(frozen=True)
class ReuseContext:
    """Consumer/producer coordinates; these do not define equivalence."""

    group_id: str
    policy_epoch: str
    branch_id: str
    turn_id: int

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id is required")
        if not self.policy_epoch:
            raise ValueError("policy_epoch is required")


@dataclass(frozen=True)
class ProducerMetadata:
    group_id: str
    policy_epoch: str
    branch_id: str
    turn_id: int

    @classmethod
    def from_context(cls, context: ReuseContext) -> "ProducerMetadata":
        return cls(**dataclasses.asdict(context))


@dataclass(frozen=True)
class ReuseArtifact(Generic[T]):
    identity: ArtifactIdentity
    sharing_scope: SharingScope
    producer: ProducerMetadata
    value: T


@dataclass(frozen=True)
class ReuseResult(Generic[T]):
    artifact: ReuseArtifact[T]
    action: ExecutionAction
    source: str

    @property
    def value(self) -> T:
        return self.artifact.value

    @property
    def exact_reuse(self) -> bool:
        return self.action is ExecutionAction.EXACT

    @property
    def decision(self) -> ExecutionDecision:
        identity = self.artifact.identity
        return ExecutionDecision(
            action=self.action,
            operator_id=identity.operator_id,
            representation_stage=identity.representation_stage,
            reason=self.source,
            eligible_units=1,
            applied_units=1 if self.action is ExecutionAction.EXACT else 0,
            approximate=False,
            policy="content-addressed-single-flight",
        )

    def event_fields(self, prefix: str) -> dict[str, Any]:
        producer = self.artifact.producer
        return {
            **self.decision.event_fields(prefix),
            f"{prefix}_source": self.source,
            f"{prefix}_exact_reuse": self.exact_reuse,
            f"{prefix}_producer_group_id": producer.group_id,
            f"{prefix}_producer_policy_epoch": producer.policy_epoch,
            f"{prefix}_producer_branch_id": producer.branch_id,
            f"{prefix}_producer_turn_id": producer.turn_id,
            f"{prefix}_content_id": self.artifact.identity.content_id,
            f"{prefix}_dependency_signature": self.artifact.identity.dependency_signature,
        }
