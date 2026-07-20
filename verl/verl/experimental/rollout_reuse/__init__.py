"""Dependency-aware reuse primitives for grouped rollout computation."""

from .artifact import (
    ArtifactIdentity,
    ExecutionAction,
    ProducerMetadata,
    ReuseArtifact,
    ReuseContext,
    ReuseResult,
    SharingScope,
)
from .registry import ReuseRegistry
from .routing import group_preserving_slices

__all__ = [
    "ArtifactIdentity",
    "ExecutionAction",
    "ProducerMetadata",
    "ReuseArtifact",
    "ReuseContext",
    "ReuseRegistry",
    "ReuseResult",
    "SharingScope",
    "group_preserving_slices",
]
