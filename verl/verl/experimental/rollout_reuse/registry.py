"""In-owner artifact registry with failure-safe async single-flight."""

from __future__ import annotations

import asyncio
import inspect
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar, cast

from .artifact import (
    ArtifactIdentity,
    ExecutionAction,
    ProducerMetadata,
    ReuseArtifact,
    ReuseContext,
    ReuseResult,
    SharingScope,
)


T = TypeVar("T")
RegistryKey = tuple[ArtifactIdentity, tuple[str, ...]]


class ReuseRegistry(Generic[T]):
    """Own artifacts and coalesce concurrent equivalent computations.

    This object provides single-flight within the process/actor that owns it.
    Cross-process correctness is obtained by routing a GRPO group to the same
    owner; GPU artifacts remain owned by the SGLang replica cache.
    """

    def __init__(self, max_entries: int = 4096):
        self.max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[RegistryKey, ReuseArtifact[T]] = OrderedDict()
        self._flights: dict[RegistryKey, asyncio.Task[ReuseArtifact[T]]] = {}
        self._lock = asyncio.Lock()
        self._stats = {
            "lookups": 0,
            "local_computes": 0,
            "cache_hits": 0,
            "coalesced_hits": 0,
            "failures": 0,
            "caller_cancellations": 0,
            "evictions": 0,
        }

    @staticmethod
    def _namespace(scope: SharingScope, context: ReuseContext) -> tuple[str, ...]:
        if scope is SharingScope.GROUP:
            return (scope.value, context.policy_epoch, context.group_id)
        if scope is SharingScope.POLICY_EPOCH:
            return (scope.value, context.policy_epoch)
        if scope is SharingScope.GLOBAL:
            return (scope.value,)
        raise ValueError(f"Unsupported sharing scope: {scope}")

    def _key(
        self,
        identity: ArtifactIdentity,
        scope: SharingScope,
        context: ReuseContext,
    ) -> RegistryKey:
        return identity, self._namespace(scope, context)

    @staticmethod
    def _consume_task_exception(task: asyncio.Task[Any]) -> None:
        # If every caller is cancelled, retrieving the exception here prevents
        # an unobserved-task warning. Active waiters still receive the same error.
        if not task.cancelled():
            task.exception()

    async def _produce(
        self,
        *,
        key: RegistryKey,
        identity: ArtifactIdentity,
        scope: SharingScope,
        context: ReuseContext,
        compute: Callable[[], T | Awaitable[T]],
    ) -> ReuseArtifact[T]:
        try:
            value_or_awaitable = compute()
            if inspect.isawaitable(value_or_awaitable):
                value = await cast(Awaitable[T], value_or_awaitable)
            else:
                value = cast(T, value_or_awaitable)
            artifact = ReuseArtifact(
                identity=identity,
                sharing_scope=scope,
                producer=ProducerMetadata.from_context(context),
                value=value,
            )
            async with self._lock:
                self._entries[key] = artifact
                self._entries.move_to_end(key)
                self._stats["local_computes"] += 1
                while len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
                    self._stats["evictions"] += 1
            return artifact
        except asyncio.CancelledError:
            raise
        except BaseException:
            async with self._lock:
                self._stats["failures"] += 1
            raise
        finally:
            async with self._lock:
                current = self._flights.get(key)
                if current is asyncio.current_task():
                    self._flights.pop(key, None)

    async def get_or_compute(
        self,
        *,
        identity: ArtifactIdentity,
        scope: SharingScope,
        context: ReuseContext,
        compute: Callable[[], T | Awaitable[T]],
    ) -> ReuseResult[T]:
        key = self._key(identity, scope, context)
        async with self._lock:
            self._stats["lookups"] += 1
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self._stats["cache_hits"] += 1
                return ReuseResult(cached, ExecutionAction.EXACT, "cache")

            task = self._flights.get(key)
            producer = task is None
            if task is None:
                task = asyncio.create_task(
                    self._produce(
                        key=key,
                        identity=identity,
                        scope=scope,
                        context=context,
                        compute=compute,
                    )
                )
                task.add_done_callback(self._consume_task_exception)
                self._flights[key] = task

        try:
            # A caller cancellation must not cancel work required by other branches.
            artifact = await asyncio.shield(task)
        except asyncio.CancelledError:
            async with self._lock:
                self._stats["caller_cancellations"] += 1
            raise

        if producer:
            return ReuseResult(artifact, ExecutionAction.LOCAL, "computed")
        async with self._lock:
            self._stats["coalesced_hits"] += 1
        return ReuseResult(artifact, ExecutionAction.EXACT, "single_flight")

    async def clear(self, *, cancel_inflight: bool = True) -> None:
        async with self._lock:
            self._entries.clear()
            tasks = list(self._flights.values()) if cancel_inflight else []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def invalidate_policy_epoch(self, policy_epoch: str | int) -> int:
        epoch = str(policy_epoch)
        async with self._lock:
            keys = [
                key
                for key, artifact in self._entries.items()
                if artifact.producer.policy_epoch == epoch
            ]
            for key in keys:
                self._entries.pop(key, None)
            tasks = [
                task
                for (_, namespace), task in self._flights.items()
                if len(namespace) > 1 and namespace[1] == epoch
            ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(keys)

    async def stats(self) -> dict[str, int]:
        async with self._lock:
            return {
                **self._stats,
                "entries": len(self._entries),
                "inflight": len(self._flights),
            }
