"""Public execution runtime for EXACT/PARTIAL/SKIP/LOCAL decisions."""

from __future__ import annotations

import threading
from collections import Counter, deque
from collections.abc import Awaitable, Callable
from typing import Any, Generic, Mapping, TypeVar

from .artifact import (
    ArtifactIdentity,
    ExecutionAction,
    ExecutionDecision,
    ReuseContext,
    ReuseResult,
    SharingScope,
)
from .registry import ReuseRegistry


T = TypeVar("T")


class RolloutReuseRuntime(Generic[T]):
    """Execute exact reuse and account for every other execution action.

    Backends remain responsible for the mechanics of PARTIAL and SKIP, but they must
    obtain and record an :class:`ExecutionDecision` here.  This keeps exact artifact
    reuse, approximate pruning, and correctness fallbacks in one trace rather than in
    workload-specific counters.
    """

    def __init__(
        self,
        registry: ReuseRegistry[T] | None = None,
        *,
        max_trace_events: int = 8192,
    ) -> None:
        self.registry = registry or ReuseRegistry[T]()
        self._events: deque[ExecutionDecision] = deque(
            maxlen=max(1, int(max_trace_events))
        )
        self._counts: Counter[str] = Counter()
        self._eligible_units: Counter[str] = Counter()
        self._applied_units: Counter[str] = Counter()
        self._lock = threading.Lock()

    def record(self, decision: ExecutionDecision) -> ExecutionDecision:
        with self._lock:
            action = decision.action.value
            self._events.append(decision)
            self._counts[action] += 1
            self._eligible_units[action] += decision.eligible_units
            self._applied_units[action] += decision.applied_units
        return decision

    def decide(
        self,
        *,
        action: ExecutionAction,
        operator_id: str,
        representation_stage: str,
        reason: str,
        eligible_units: int = 0,
        applied_units: int = 0,
        approximate: bool = False,
        error_bound: float | None = None,
        policy: str = "",
    ) -> ExecutionDecision:
        return self.record(
            ExecutionDecision(
                action=action,
                operator_id=operator_id,
                representation_stage=representation_stage,
                reason=reason,
                eligible_units=eligible_units,
                applied_units=applied_units,
                approximate=approximate,
                error_bound=error_bound,
                policy=policy,
            )
        )

    def record_event_fields(
        self, fields: Mapping[str, Any], *, prefix: str = "reuse"
    ) -> ExecutionDecision:
        """Validate and ingest a decision emitted by a backend process."""

        return self.record(ExecutionDecision.from_event_fields(fields, prefix))

    async def get_or_compute(
        self,
        *,
        identity: ArtifactIdentity,
        scope: SharingScope,
        context: ReuseContext,
        compute: Callable[[], T | Awaitable[T]],
    ) -> ReuseResult[T]:
        result = await self.registry.get_or_compute(
            identity=identity,
            scope=scope,
            context=context,
            compute=compute,
        )
        self.record(result.decision)
        return result

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "counts": dict(self._counts),
                "eligible_units": dict(self._eligible_units),
                "applied_units": dict(self._applied_units),
                "events": tuple(self._events),
            }

    def clear_trace(self) -> None:
        with self._lock:
            self._events.clear()
            self._counts.clear()
            self._eligible_units.clear()
            self._applied_units.clear()
