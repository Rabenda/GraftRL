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

    def event_fields(self, prefix: str) -> dict[str, Any]:
        producer = self.artifact.producer
        return {
            f"{prefix}_action": self.action.value,
            f"{prefix}_source": self.source,
            f"{prefix}_exact_reuse": self.exact_reuse,
            f"{prefix}_producer_group_id": producer.group_id,
            f"{prefix}_producer_policy_epoch": producer.policy_epoch,
            f"{prefix}_producer_branch_id": producer.branch_id,
            f"{prefix}_producer_turn_id": producer.turn_id,
            f"{prefix}_content_id": self.artifact.identity.content_id,
            f"{prefix}_dependency_signature": self.artifact.identity.dependency_signature,
        }
