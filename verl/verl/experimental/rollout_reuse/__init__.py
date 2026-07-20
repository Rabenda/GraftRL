"""Dependency-aware reuse primitives for grouped rollout computation."""

from .artifact import (
    ACTION_SCHEMA_VERSION,
    ArtifactIdentity,
    ExecutionAction,
    ExecutionDecision,
    ProducerMetadata,
    ReuseArtifact,
    ReuseContext,
    ReuseResult,
    SharingScope,
)
from .registry import ReuseRegistry
from .routing import group_preserving_slices
from .runtime import RolloutReuseRuntime

__all__ = [
    "ACTION_SCHEMA_VERSION",
    "ArtifactIdentity",
    "ExecutionAction",
    "ExecutionDecision",
    "ProducerMetadata",
    "ReuseArtifact",
    "ReuseContext",
    "ReuseRegistry",
    "ReuseResult",
    "RolloutReuseRuntime",
    "SharingScope",
    "group_preserving_slices",
]
