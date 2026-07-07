# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utility classes for manage and request LLM servers:
- LLMServerManager: manage life-cycle of LLM servers, including launch, tear-down replicas.
- LLMServerClient: proxy client to request LLM servers, used by AgentLoopWorker.
- GlobalRequestLoadBalancer: global load balancer for LLMServerClient.
"""

import asyncio
import csv
import fcntl
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import uuid4

import ray
from cachetools import LRUCache
from omegaconf import DictConfig

from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import RolloutReplica, TokenOutput, get_rollout_replica_class
from verl.workers.rollout.utils import update_prometheus_config

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000


def _grpo_sim_cache_enabled() -> bool:
    return os.environ.get("SGLANG_GRPO_SIM_CACHE", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _vlm_cacheblend_enabled() -> bool:
    return os.environ.get("SGLANG_VLM_CACHEBLEND", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _vlm_cacheblend_warmup_barrier_enabled() -> bool:
    return os.environ.get("SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER", "1").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _vlm_cacheblend_target_turn() -> int:
    try:
        return int(os.environ.get("SGLANG_VLM_CACHEBLEND_TARGET_TURN", "1"))
    except (TypeError, ValueError):
        return 1


def _vlm_cacheblend_target_turns_spec() -> str:
    return os.environ.get(
        "SGLANG_VLM_CACHEBLEND_TARGET_TURNS",
        os.environ.get("SGLANG_VLM_CACHEBLEND_TARGET_TURN", "1"),
    ).strip().lower()


def _vlm_cacheblend_turn_enabled(agent_turn: int) -> bool:
    if agent_turn < 1:
        return False
    spec = _vlm_cacheblend_target_turns_spec()
    if spec in ("all", "*"):
        return True
    turns: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            turns.add(int(part))
        except ValueError:
            logger.warning("Invalid SGLANG_VLM_CACHEBLEND_TARGET_TURNS entry: %s", part)
    if turns:
        return agent_turn in turns
    return agent_turn == _vlm_cacheblend_target_turn()


def _vlm_cacheblend_prefix_warmup_enabled() -> bool:
    return os.environ.get("SGLANG_VLM_CACHEBLEND_PREFIX_WARMUP_BARRIER", "1").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _vlm_cacheblend_warmup_keep_steps() -> int:
    try:
        return max(1, int(os.environ.get("SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_KEEP_STEPS", "4")))
    except (TypeError, ValueError):
        return 4


def _vlm_cacheblend_warmup_timeout_s() -> float:
    try:
        return max(
            0.0,
            float(os.environ.get("SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_TIMEOUT_S", "300")),
        )
    except (TypeError, ValueError):
        return 300.0


def _vlm_cacheblend_warmup_timeout_action() -> str:
    action = os.environ.get("SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_TIMEOUT_ACTION", "fail").strip().lower()
    if action in ("fallback", "continue", "warn"):
        return "fallback"
    return "fail"


def _vlm_cacheblend_warmup_wait_policy() -> str:
    policy = os.environ.get("SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_WAIT_POLICY", "strict").strip().lower()
    if policy in ("bounded", "soft", "fail_open", "fail-open"):
        return "bounded"
    return "strict"


def _vlm_cacheblend_warmup_max_wait_s() -> float:
    try:
        return max(
            0.0,
            float(os.environ.get("SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_MAX_WAIT_S", "0.05")),
        )
    except (TypeError, ValueError):
        return 0.05


def _vlm_cacheblend_warmup_wait_timeout_s() -> float:
    if _vlm_cacheblend_warmup_wait_policy() == "bounded":
        return _vlm_cacheblend_warmup_max_wait_s()
    return _vlm_cacheblend_warmup_timeout_s()


def _vlm_cacheblend_prefill_donor_enabled() -> bool:
    return os.environ.get("SGLANG_VLM_CACHEBLEND_PREFILL_DONOR", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _vlm_cacheblend_prefill_donor_max_new_tokens() -> int:
    try:
        return max(1, int(os.environ.get("SGLANG_VLM_CACHEBLEND_PREFILL_DONOR_MAX_NEW_TOKENS", "1")))
    except (TypeError, ValueError):
        return 1


def _vlm_cacheblend_prefill_donor_wait_timeout_s() -> float:
    try:
        return max(0.0, float(os.environ.get("SGLANG_VLM_CACHEBLEND_PREFILL_DONOR_WAIT_TIMEOUT_S", "10")))
    except (TypeError, ValueError):
        return 10.0


def _vlm_cacheblend_coordinator_actor_name() -> str:
    name = os.environ.get("SGLANG_VLM_CACHEBLEND_COORDINATOR_ACTOR", "vlm_cacheblend_coordinator").strip()
    return name or "vlm_cacheblend_coordinator"


def _vlm_cacheblend_coordinator_actor_namespace() -> str:
    namespace = os.environ.get("SGLANG_VLM_CACHEBLEND_COORDINATOR_NAMESPACE", "vlm_cacheblend").strip()
    return namespace or "vlm_cacheblend"


def _normalize_global_step(global_step: Optional[Any]) -> Optional[int]:
    if global_step is None:
        return None
    try:
        return int(global_step)
    except (TypeError, ValueError):
        return None


def _cacheblend_group_key(agent_uid: str, global_step: Optional[Any]) -> str:
    step = _normalize_global_step(global_step)
    if step is None:
        return str(agent_uid)
    return f"{step}:{agent_uid}"


def _warmup_key_matches_step(key: str, step: int) -> bool:
    return key.startswith(f"{step}:") or f":{step}:" in key


def _prune_warmed_uids_for_step(
    warmed_uids: set[str],
    recent_steps: list[int],
    current_step: int,
    *,
    keep_steps: int,
) -> None:
    """Drop warmed keys for training steps older than the last ``keep_steps`` seen."""
    if current_step not in recent_steps:
        recent_steps.append(current_step)
    while len(recent_steps) > keep_steps:
        old_step = recent_steps.pop(0)
        drop = {key for key in warmed_uids if _warmup_key_matches_step(key, old_step)}
        warmed_uids.difference_update(drop)


BarrierRole = Literal["donor", "recipient", "bypass", "prefill_donor", "prefill_wait", "prefill_skip"]
WarmupAcquireRole = Literal["donor", "recipient", "recipient_wait"]


@ray.remote(max_concurrency=1000)
class GlobalCacheBlendCoordinator:
    """Global CacheBlend rollout-side coordinator shared by all AgentLoopWorkers.

    The actor does not move or store KV. It only serializes the first target-turn
    request for a GRPO group so that the selected SGLang replica can capture donor
    KV before recipient branches are sent to the same sticky route.
    """

    def __init__(self, keep_steps: int = 4):
        self._lock = asyncio.Lock()
        self._changed = asyncio.Condition(self._lock)
        self._states: dict[str, str] = {}
        self._recent_steps: list[int] = []
        self._keep_steps = max(1, int(keep_steps))

    def _prune_locked(self, global_step: Optional[Any]) -> None:
        step = _normalize_global_step(global_step)
        if step is None:
            return
        if step not in self._recent_steps:
            self._recent_steps.append(step)
        while len(self._recent_steps) > self._keep_steps:
            old_step = self._recent_steps.pop(0)
            for key in [key for key in self._states if _warmup_key_matches_step(key, old_step)]:
                self._states.pop(key, None)

    async def begin_warmup(self, key: str, global_step: Optional[Any]) -> WarmupAcquireRole:
        async with self._changed:
            self._prune_locked(global_step)
            state = self._states.get(key)
            if state == "ready":
                return "recipient"
            if state == "in_progress":
                return "recipient_wait"
            self._states[key] = "in_progress"
            self._changed.notify_all()
            return "donor"

    async def mark_ready(self, key: str) -> None:
        async with self._changed:
            self._states[key] = "ready"
            self._changed.notify_all()

    async def mark_failed(self, key: str) -> None:
        async with self._changed:
            if self._states.get(key) == "in_progress":
                self._states[key] = "failed"
            self._changed.notify_all()

    async def wait_until_ready(self, key: str, timeout_s: float) -> bool:
        async with self._changed:
            if timeout_s <= 0:
                return self._states.get(key) == "ready"
            deadline = time.monotonic() + timeout_s
            while self._states.get(key) == "in_progress":
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                try:
                    await asyncio.wait_for(self._changed.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return False
            return self._states.get(key) == "ready"

    async def get_status(self) -> dict[str, Any]:
        async with self._changed:
            counts: dict[str, int] = {}
            for state in self._states.values():
                counts[state] = counts.get(state, 0) + 1
            return {
                "states": counts,
                "num_keys": len(self._states),
                "recent_steps": list(self._recent_steps),
            }


@ray.remote
class GlobalRequestLoadBalancer:
    """Global sticky-session + in-flight load balancer shared by all AgentLoopWorkers.

    When a sticky session points to a removed server, the cache entry is
    automatically invalidated and a new server is selected.

    Key features:
    - **Atomic acquire**: ``acquire_server()`` returns ``(server_id, handle)``
    - **Sticky Session**: Uses LRUCache to map request_id → server_id, ensuring
      multi-turn conversations route to the same server.
    - **Least-loaded Selection**: When no sticky session exists, selects the
      server with the fewest in-flight requests.
    - **Dynamic Server Management**: Supports add/remove servers at runtime
      for hybrid scaling.
    """

    def __init__(self, servers: dict[str, ray.actor.ActorHandle], max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE):
        if not servers:
            raise ValueError("servers must be non-empty")

        self._servers: dict[str, ray.actor.ActorHandle] = dict(servers)
        self._inflight_requests: dict[str, int] = {sid: 0 for sid in servers}
        self._request_id_to_server: LRUCache = LRUCache(maxsize=max_cache_size)

    def acquire_server(self, request_id: str) -> tuple[str, ray.actor.ActorHandle]:
        """Acquire a server for the given request (sticky + least-loaded).

        Returns:
            A tuple of ``(server_id, actor_handle)`` in a single atomic call.
        """
        # Try sticky session first
        if request_id in self._request_id_to_server:
            server_id = self._request_id_to_server[request_id]
            # Check if server is still in the active pool
            if server_id in self._inflight_requests:
                self._inflight_requests[server_id] += 1
                return server_id, self._servers[server_id]
            # Server was removed, clear stale cache entry and re-select
            del self._request_id_to_server[request_id]

        # Select new server (least-loaded among available)
        if not self._inflight_requests:
            raise RuntimeError("No available servers in load balancer")

        server_id = min(self._inflight_requests, key=self._inflight_requests.get)
        self._request_id_to_server[request_id] = server_id
        self._inflight_requests[server_id] += 1
        return server_id, self._servers[server_id]

    def release_server(self, server_id: str) -> None:
        """Release a server after a request completes."""
        if server_id not in self._inflight_requests:
            return
        if self._inflight_requests[server_id] > 0:
            self._inflight_requests[server_id] -= 1

    def add_servers(self, servers: dict[str, ray.actor.ActorHandle]) -> None:
        """Atomically add multiple servers to the load balancer pool.

        This is more efficient than calling :meth:`add_server` in a loop
        because it performs a single bulk update on the internal state.

        Args:
            servers: Dict mapping server_id → actor_handle for all servers
                to register.
        """
        for sid, handle in servers.items():
            self._inflight_requests[sid] = 0
            self._servers[sid] = handle
        logger.info(f"[GlobalLoadBalancer] added {len(servers)} servers")

    def remove_servers(self, server_ids: list[str]) -> None:
        """Atomically remove multiple servers from the load balancer pool.

        More efficient than calling :meth:`remove_server` in a loop.

        Args:
            server_ids: List of server identifiers to remove.
        """
        for sid in server_ids:
            self._inflight_requests.pop(sid, None)
            self._servers.pop(sid, None)
        logger.info(f"[GlobalLoadBalancer] removed {len(server_ids)} servers")

    def get_inflight_count(self, server_id: str) -> int:
        """Get number of in-flight requests for a server."""
        return self._inflight_requests.get(server_id, 0)

    def get_all_servers(self) -> list[str]:
        """Get list of all active server IDs."""
        return list(self._inflight_requests.keys())

    def get_status(self) -> dict:
        """Return current load balancer state for debugging."""
        return {
            "servers": dict(self._inflight_requests),
            "total_inflight": sum(self._inflight_requests.values()),
            "active_servers": len(self._inflight_requests),
            "registered_handles": list(self._servers.keys()),
        }


class LLMServerClient:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least in-flight requests load balancing via global coordination
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(
        self,
        config: DictConfig,
        load_balancer_handle: ray.actor.ActorHandle,
        cacheblend_coordinator_handle: Optional[ray.actor.ActorHandle] = None,
        **kwargs,
    ):
        """Initialize the LLMServerClient.

        Args:
            config (DictConfig): whole config for main entrypoint.
            load_balancer_handle (ray.actor.ActorHandle): shared global load balancer actor
                that also holds the server-handle registry.
        """
        self.config = config
        self._load_balancer = load_balancer_handle
        self._cacheblend_coordinator = cacheblend_coordinator_handle
        self._vlm_cacheblend_warmed_uids: set[str] = set()
        self._vlm_cacheblend_warmup_locks: dict[str, asyncio.Lock] = {}
        self._vlm_cacheblend_warmup_recent_steps: list[int] = []
        self._vlm_cacheblend_barrier_log_lock = asyncio.Lock()
        self._vlm_cacheblend_barrier_log_ready = False
        self._vlm_cacheblend_missing_step_warned = False

    def _vlm_cacheblend_warmup_key(
        self,
        agent_uid: Optional[str],
        agent_turn: Optional[int],
        global_step: Optional[Any] = None,
    ) -> Optional[str]:
        if not (
            _vlm_cacheblend_enabled()
            and _vlm_cacheblend_warmup_barrier_enabled()
            and agent_uid
        ):
            return None
        try:
            turn = int(agent_turn)
        except (TypeError, ValueError):
            return None
        if turn == 0 and _vlm_cacheblend_prefix_warmup_enabled():
            return f"prefix:{_cacheblend_group_key(str(agent_uid), global_step)}:turn0"
        if not _vlm_cacheblend_turn_enabled(turn):
            return None
        return f"cacheblend:{_cacheblend_group_key(str(agent_uid), global_step)}:turn{turn}"

    def _vlm_cacheblend_warmup_lock(self, key: str) -> asyncio.Lock:
        lock = self._vlm_cacheblend_warmup_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._vlm_cacheblend_warmup_locks[key] = lock
        return lock

    def _maybe_prune_warmed_uids(self, global_step: Optional[Any]) -> None:
        if global_step is None:
            return
        try:
            step = int(global_step)
        except (TypeError, ValueError):
            return
        _prune_warmed_uids_for_step(
            self._vlm_cacheblend_warmed_uids,
            self._vlm_cacheblend_warmup_recent_steps,
            step,
            keep_steps=_vlm_cacheblend_warmup_keep_steps(),
        )

    def _barrier_log_path(self) -> Optional[Path]:
        log_dir = os.environ.get("SGLANG_INFERENCE_LOG_DIR", "").strip()
        if not log_dir:
            return None
        suffix = os.environ.get("SGLANG_INFERENCE_LOG_SUFFIX", "").strip()
        name = "cacheblend_barrier_log"
        if suffix:
            name = f"{name}_{suffix}"
        return Path(log_dir) / f"{name}.csv"

    async def _log_cacheblend_barrier_event(
        self,
        *,
        request_id: str,
        agent_uid: Optional[str],
        agent_turn: Optional[int],
        rollout_idx: Optional[str],
        global_step: Optional[Any],
        warmup_key: Optional[str],
        barrier_role: BarrierRole,
        wait_ms: float,
        donor_ready: bool,
        barrier_wait_ms: Optional[float] = None,
        server_call_ms: Optional[float] = None,
        wait_policy: Optional[str] = None,
        routing_request_id: Optional[str] = None,
        server_id: Optional[str] = None,
    ) -> None:
        target_turn = _vlm_cacheblend_target_turns_spec()
        barrier_enabled = _vlm_cacheblend_enabled() and _vlm_cacheblend_warmup_barrier_enabled()
        if barrier_wait_ms is None:
            barrier_wait_ms = wait_ms
        if server_call_ms is None:
            server_call_ms = 0.0
        if wait_policy is None:
            wait_policy = _vlm_cacheblend_warmup_wait_policy()
        fields = {
            "barrier_enabled": barrier_enabled,
            "barrier_role": barrier_role,
            "agent_uid": agent_uid or "",
            "agent_turn": agent_turn if agent_turn is not None else "",
            "target_turn": target_turn,
            "rollout_idx": rollout_idx or "",
            "global_step": global_step if global_step is not None else "",
            "request_id": request_id,
            "warmup_key": warmup_key or "",
            "routing_request_id": routing_request_id or "",
            "server_id": server_id or "",
            "wait_ms": f"{wait_ms:.3f}",
            "barrier_wait_ms": f"{barrier_wait_ms:.3f}",
            "server_call_ms": f"{server_call_ms:.3f}",
            "wait_policy": wait_policy,
            "donor_ready": donor_ready,
        }
        logger.info("[VLMCacheBlendBarrier] %s", " ".join(f"{k}={v}" for k, v in fields.items()))

        log_path = self._barrier_log_path()
        if log_path is None:
            return
        row = {
            "timestamp": f"{time.time():.6f}",
            **fields,
            "wait_ms": f"{wait_ms:.3f}",
            "barrier_wait_ms": f"{barrier_wait_ms:.3f}",
            "server_call_ms": f"{server_call_ms:.3f}",
            "donor_ready": "1" if donor_ready else "0",
            "barrier_enabled": "1" if barrier_enabled else "0",
        }
        async with self._vlm_cacheblend_barrier_log_lock:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a+", encoding="utf-8", newline="") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.seek(0, os.SEEK_END)
                write_header = handle.tell() == 0
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            self._vlm_cacheblend_barrier_log_ready = True

    async def _acquire_server(self, request_id: str) -> tuple[str, ray.actor.ActorHandle]:
        # Atomic acquire: returns (server_id, handle) in one Ray RPC.
        return await self._load_balancer.acquire_server.remote(request_id=request_id)

    def _release_server(self, server_id: str) -> None:
        # Fire-and-forget: release is just a counter decrement, no need to await.
        # Awaiting here risks blocking the finally clause if the LB actor is unresponsive.
        self._load_balancer.release_server.remote(server_id=server_id)

    @rollout_trace_op
    async def cacheblend_prefill_warmup(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        agent_turn: Optional[int] = None,
        agent_uid: Optional[str] = None,
        rollout_idx: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """Prime VLM CacheBlend with a short synthetic donor request.

        The existing warmup barrier makes the first real branch of a group serve as
        donor, so donor decode can delay recipient readiness. This method moves that
        donor work into a one-token synthetic request: the first worker for a group
        submits it, the rest only wait for the coordinator to become ready.
        """
        warmup_global_step = kwargs.pop("training_global_step", kwargs.pop("global_step", None))
        if warmup_global_step is not None:
            kwargs["training_global_step"] = warmup_global_step
        if not (_vlm_cacheblend_enabled() and _vlm_cacheblend_prefill_donor_enabled()):
            return False
        if not agent_uid or agent_turn is None:
            return False
        warmup_key = self._vlm_cacheblend_warmup_key(agent_uid, agent_turn, warmup_global_step)
        if warmup_key is None or self._cacheblend_coordinator is None:
            return False

        routing_request_id = f"cacheblend_group:{_cacheblend_group_key(str(agent_uid), warmup_global_step)}"
        wait_start = time.perf_counter()
        acquire_role = await self._cacheblend_coordinator.begin_warmup.remote(
            warmup_key,
            warmup_global_step,
        )
        barrier_wait_ms = (time.perf_counter() - wait_start) * 1000.0

        if acquire_role == "donor":
            server_id = ""
            server_call_ms = 0.0
            try:
                params = dict(sampling_params)
                params.pop("max_tokens", None)
                params["max_new_tokens"] = _vlm_cacheblend_prefill_donor_max_new_tokens()
                server_id, server = await self._acquire_server(routing_request_id)
                call_start = time.perf_counter()
                multimodal_kwargs = {}
                if audio_data is not None:
                    multimodal_kwargs["audio_data"] = audio_data
                if mm_processor_kwargs:
                    multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
                await server.generate.remote(
                    request_id=f"{request_id}_cacheblend_prefill_donor_t{int(agent_turn)}",
                    prompt_ids=prompt_ids,
                    sampling_params=params,
                    image_data=image_data,
                    video_data=video_data,
                    agent_request_id=request_id,
                    agent_turn=agent_turn,
                    agent_uid=agent_uid,
                    rollout_idx=rollout_idx,
                    **multimodal_kwargs,
                    **kwargs,
                )
                server_call_ms = (time.perf_counter() - call_start) * 1000.0
                await self._cacheblend_coordinator.mark_ready.remote(warmup_key)
                total_ms = (time.perf_counter() - wait_start) * 1000.0
                await self._log_cacheblend_barrier_event(
                    request_id=str(request_id),
                    agent_uid=agent_uid,
                    agent_turn=agent_turn,
                    rollout_idx=rollout_idx,
                    global_step=warmup_global_step,
                    warmup_key=warmup_key,
                    barrier_role="prefill_donor",
                    wait_ms=total_ms,
                    barrier_wait_ms=barrier_wait_ms,
                    server_call_ms=server_call_ms,
                    donor_ready=True,
                    routing_request_id=str(routing_request_id),
                    server_id=str(server_id),
                )
                return True
            except Exception:
                await self._cacheblend_coordinator.mark_failed.remote(warmup_key)
                raise
            finally:
                if server_id:
                    self._release_server(server_id)

        donor_ready = True
        if acquire_role == "recipient_wait":
            donor_ready = await self._cacheblend_coordinator.wait_until_ready.remote(
                warmup_key,
                _vlm_cacheblend_prefill_donor_wait_timeout_s(),
            )
            barrier_wait_ms = (time.perf_counter() - wait_start) * 1000.0
        total_ms = (time.perf_counter() - wait_start) * 1000.0
        await self._log_cacheblend_barrier_event(
            request_id=str(request_id),
            agent_uid=agent_uid,
            agent_turn=agent_turn,
            rollout_idx=rollout_idx,
            global_step=warmup_global_step,
            warmup_key=warmup_key,
            barrier_role="prefill_wait" if donor_ready else "prefill_skip",
            wait_ms=total_ms,
            barrier_wait_ms=barrier_wait_ms,
            server_call_ms=0.0,
            donor_ready=donor_ready,
            routing_request_id=str(routing_request_id),
            server_id="",
        )
        return bool(donor_ready)

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        agent_turn: Optional[int] = None,
        agent_uid: Optional[str] = None,
        rollout_idx: Optional[str] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput | DiffusionOutput: token or diffusion output
        """
        warmup_global_step = kwargs.pop("training_global_step", kwargs.pop("global_step", None))
        if warmup_global_step is not None:
            kwargs["training_global_step"] = warmup_global_step
        elif _vlm_cacheblend_enabled() and agent_uid and not self._vlm_cacheblend_missing_step_warned:
            self._vlm_cacheblend_missing_step_warned = True
            logger.warning(
                "VLM CacheBlend request has agent_uid=%s but no training_global_step/global_step; "
                "routing and warmup keys will fall back to agent_uid only.",
                agent_uid,
            )
        routing_request_id = request_id
        if _vlm_cacheblend_enabled() and agent_uid:
            routing_request_id = f"cacheblend_group:{_cacheblend_group_key(str(agent_uid), warmup_global_step)}"
        elif _grpo_sim_cache_enabled() and agent_uid:
            routing_request_id = f"grpo_agent_uid:{agent_uid}"

        self._maybe_prune_warmed_uids(warmup_global_step)
        warmup_key = self._vlm_cacheblend_warmup_key(agent_uid, agent_turn, warmup_global_step)

        async def _call_server() -> tuple[TokenOutput, str]:
            server_id, server = await self._acquire_server(routing_request_id)
            try:
                multimodal_kwargs = {}
                if audio_data is not None:
                    multimodal_kwargs["audio_data"] = audio_data
                if mm_processor_kwargs:
                    multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
                # Sticky LB uses agent `request_id`; SGLang rid is per-turn for profiling correlation.
                if agent_turn is not None:
                    sglang_request_id = f"{request_id}_t{int(agent_turn)}"
                else:
                    sglang_request_id = uuid4().hex

                output = await server.generate.remote(
                    request_id=sglang_request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=image_data,
                    video_data=video_data,
                    agent_request_id=request_id,
                    agent_turn=agent_turn,
                    agent_uid=agent_uid,
                    rollout_idx=rollout_idx,
                    **multimodal_kwargs,
                    **kwargs,
                )
                return output, str(server_id)
            finally:
                self._release_server(server_id)

        if warmup_key is None:
            call_start = time.perf_counter()
            output, server_id = await _call_server()
            server_call_ms = (time.perf_counter() - call_start) * 1000.0
            await self._log_cacheblend_barrier_event(
                request_id=str(request_id),
                agent_uid=agent_uid,
                agent_turn=agent_turn,
                rollout_idx=rollout_idx,
                global_step=warmup_global_step,
                warmup_key=None,
                barrier_role="bypass",
                wait_ms=0.0,
                barrier_wait_ms=0.0,
                server_call_ms=server_call_ms,
                donor_ready=False,
                routing_request_id=str(routing_request_id),
                server_id=server_id,
            )
            return output

        if self._cacheblend_coordinator is not None:
            wait_start = time.perf_counter()
            acquire_role = await self._cacheblend_coordinator.begin_warmup.remote(
                warmup_key,
                warmup_global_step,
            )
            barrier_wait_ms = (time.perf_counter() - wait_start) * 1000.0
            if acquire_role == "donor":
                call_start = time.perf_counter()
                try:
                    output, server_id = await _call_server()
                except Exception:
                    await self._cacheblend_coordinator.mark_failed.remote(warmup_key)
                    raise
                server_call_ms = (time.perf_counter() - call_start) * 1000.0
                await self._cacheblend_coordinator.mark_ready.remote(warmup_key)
                total_ms = (time.perf_counter() - wait_start) * 1000.0
                await self._log_cacheblend_barrier_event(
                    request_id=str(request_id),
                    agent_uid=agent_uid,
                    agent_turn=agent_turn,
                    rollout_idx=rollout_idx,
                    global_step=warmup_global_step,
                    warmup_key=warmup_key,
                    barrier_role="donor",
                    wait_ms=total_ms,
                    barrier_wait_ms=barrier_wait_ms,
                    server_call_ms=server_call_ms,
                    donor_ready=True,
                    routing_request_id=str(routing_request_id),
                    server_id=server_id,
                )
                return output

            donor_ready = True
            if acquire_role == "recipient_wait":
                donor_ready = await self._cacheblend_coordinator.wait_until_ready.remote(
                    warmup_key,
                    _vlm_cacheblend_warmup_wait_timeout_s(),
                )
                barrier_wait_ms = (time.perf_counter() - wait_start) * 1000.0
                if not donor_ready:
                    message = (
                        "VLM CacheBlend warmup barrier timed out before donor KV was ready "
                        f"(request_id={request_id}, agent_uid={agent_uid}, agent_turn={agent_turn}, "
                        f"rollout_idx={rollout_idx}, global_step={warmup_global_step}, warmup_key={warmup_key}, "
                        f"wait_ms={barrier_wait_ms:.3f}, "
                        f"wait_policy={_vlm_cacheblend_warmup_wait_policy()})"
                    )
                    if (
                        _vlm_cacheblend_warmup_wait_policy() == "strict"
                        and _vlm_cacheblend_warmup_timeout_action() == "fail"
                    ):
                        await self._cacheblend_coordinator.mark_failed.remote(warmup_key)
                        await self._log_cacheblend_barrier_event(
                            request_id=str(request_id),
                            agent_uid=agent_uid,
                            agent_turn=agent_turn,
                            rollout_idx=rollout_idx,
                            global_step=warmup_global_step,
                            warmup_key=warmup_key,
                            barrier_role="recipient",
                            wait_ms=barrier_wait_ms,
                            barrier_wait_ms=barrier_wait_ms,
                            server_call_ms=0.0,
                            donor_ready=False,
                            routing_request_id=str(routing_request_id),
                            server_id="",
                        )
                        raise TimeoutError(message)
                    if _vlm_cacheblend_warmup_wait_policy() == "strict":
                        logger.warning(
                            "%s; continuing because SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_TIMEOUT_ACTION=fallback",
                            message,
                        )
                    else:
                        logger.info("%s; continuing with bounded CacheBlend barrier", message)
            call_start = time.perf_counter()
            output, server_id = await _call_server()
            server_call_ms = (time.perf_counter() - call_start) * 1000.0
            total_ms = (time.perf_counter() - wait_start) * 1000.0
            await self._log_cacheblend_barrier_event(
                request_id=str(request_id),
                agent_uid=agent_uid,
                agent_turn=agent_turn,
                rollout_idx=rollout_idx,
                global_step=warmup_global_step,
                warmup_key=warmup_key,
                barrier_role="recipient",
                wait_ms=total_ms,
                barrier_wait_ms=barrier_wait_ms,
                server_call_ms=server_call_ms,
                donor_ready=donor_ready,
                routing_request_id=str(routing_request_id),
                server_id=server_id,
            )
            return output

        if warmup_key in self._vlm_cacheblend_warmed_uids:
            call_start = time.perf_counter()
            output, server_id = await _call_server()
            server_call_ms = (time.perf_counter() - call_start) * 1000.0
            await self._log_cacheblend_barrier_event(
                request_id=str(request_id),
                agent_uid=agent_uid,
                agent_turn=agent_turn,
                rollout_idx=rollout_idx,
                global_step=warmup_global_step,
                warmup_key=warmup_key,
                barrier_role="recipient",
                wait_ms=0.0,
                barrier_wait_ms=0.0,
                server_call_ms=server_call_ms,
                donor_ready=True,
                routing_request_id=str(routing_request_id),
                server_id=server_id,
            )
            return output

        wait_start = time.perf_counter()
        async with self._vlm_cacheblend_warmup_lock(warmup_key):
            wait_ms = (time.perf_counter() - wait_start) * 1000.0
            if warmup_key not in self._vlm_cacheblend_warmed_uids:
                call_start = time.perf_counter()
                output, server_id = await _call_server()
                server_call_ms = (time.perf_counter() - call_start) * 1000.0
                self._vlm_cacheblend_warmed_uids.add(warmup_key)
                await self._log_cacheblend_barrier_event(
                    request_id=str(request_id),
                    agent_uid=agent_uid,
                    agent_turn=agent_turn,
                    rollout_idx=rollout_idx,
                    global_step=warmup_global_step,
                    warmup_key=warmup_key,
                    barrier_role="donor",
                    wait_ms=wait_ms,
                    barrier_wait_ms=wait_ms,
                    server_call_ms=server_call_ms,
                    donor_ready=True,
                    routing_request_id=str(routing_request_id),
                    server_id=server_id,
                )
                return output
            call_start = time.perf_counter()
            output, server_id = await _call_server()
            server_call_ms = (time.perf_counter() - call_start) * 1000.0
            await self._log_cacheblend_barrier_event(
                request_id=str(request_id),
                agent_uid=agent_uid,
                agent_turn=agent_turn,
                rollout_idx=rollout_idx,
                global_step=warmup_global_step,
                warmup_key=warmup_key,
                barrier_role="recipient",
                wait_ms=wait_ms,
                barrier_wait_ms=wait_ms,
                server_call_ms=server_call_ms,
                donor_ready=True,
                routing_request_id=str(routing_request_id),
                server_id=server_id,
            )
            return output


class LLMServerManager:
    """LLMServerManager is responsible for:
    - Launch server replicas
    - Launch global load balancer
    - Elastic launch/tear-down new replicas

    Args:
        config (DictConfig): Config for the trainer entrypoint.
        worker_group (RayWorkerGroup): Worker group for the server replicas. If not none, init hybrid server,
            else init standalone server with a new resource pool.
        rollout_resource_pool (RayResourcePool): Resource pool for the server replicas, only needed for TensorRT-LLM.
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.worker_group = worker_group
        self.rollout_resource_pool = rollout_resource_pool

        assert worker_group is not None or self.rollout_config.nnodes > 0, "nnodes must be > 0 in standalone mode"

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(
                self.rollout_config.name,
                disaggregation_enabled=self.rollout_config.disaggregation.enabled,
            )

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create the LLMServerManager."""
        instance = cls(*args, **kwargs)
        await instance._initialize_llm_servers()
        await instance._init_global_load_balancer()
        return instance

    async def _initialize_llm_servers(self, start_rank: int = 0):
        """Initialize the LLM server replicas.

        Args:
            start_rank: First ``replica_rank`` to assign.  Defaults to 0 so that
                existing callers are unaffected.  Subclasses (e.g.
                ``FullyAsyncLLMServerManager``) may pass a non-zero value to avoid
                Ray named-actor collisions when hybrid and standalone replicas
                coexist.
        """
        rollout_world_size = (
            self.rollout_config.tensor_model_parallel_size
            * self.rollout_config.data_parallel_size
            * self.rollout_config.pipeline_model_parallel_size
        )
        # PD inflates per-replica footprint; miss this and init_hybrid slices
        # past worker_group → empty workers on replica_rank>=1.
        disagg = getattr(self.rollout_config, "disaggregation", None)
        if disagg is not None and getattr(disagg, "enabled", False):
            prefill_tp = self.rollout_config.tensor_model_parallel_size
            # Inline decode_tp default: OmegaConf/Ray serialization drops dataclass methods.
            decode_tp = (
                disagg.decode_tensor_model_parallel_size
                if disagg.decode_tensor_model_parallel_size is not None
                else prefill_tp
            )
            rollout_world_size = (
                (prefill_tp * disagg.prefill_replicas + decode_tp * disagg.decode_replicas)
                * self.rollout_config.data_parallel_size
                * self.rollout_config.pipeline_model_parallel_size
            )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.rollout_config.n_gpus_per_node * self.rollout_config.nnodes
        )
        num_replicas = world_size // rollout_world_size

        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=start_rank + replica_rank,
                config=self.rollout_config,
                model_config=self.model_config,
                gpus_per_node=self.rollout_config.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group and self.rollout_config.name != "trtllm":
            await asyncio.gather(*[server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        # TODO: unify trtllm to init_hybrid
        elif self.worker_group and self.rollout_config.name == "trtllm":
            await asyncio.gather(
                *[
                    server.init_hybrid_colocated(self.worker_group, self.rollout_resource_pool)
                    for server in self.rollout_replicas
                ]
            )
        else:
            await asyncio.gather(*[server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]
        print(f"LLMServerManager: {self.server_addresses}")

        # Update Prometheus configuration with server addresses
        if self.rollout_config.prometheus.enable:
            if self.rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(self.rollout_config.prometheus, self.server_addresses, self.rollout_config.name)

    async def _init_global_load_balancer(self) -> None:
        self.global_load_balancer = GlobalRequestLoadBalancer.remote(
            servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
            max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
        )
        self.global_cacheblend_coordinator = GlobalCacheBlendCoordinator.options(
            name=_vlm_cacheblend_coordinator_actor_name(),
            namespace=_vlm_cacheblend_coordinator_actor_namespace(),
            lifetime="detached",
            get_if_exists=True,
        ).remote(
            keep_steps=_vlm_cacheblend_warmup_keep_steps(),
        )

    def get_client(self, client_cls=LLMServerClient, **kwargs) -> LLMServerClient:
        """Get the LLMServerClient to request LLM server replicas.

        Args:
            client_cls: The client class to instantiate (default: ``LLMServerClient``).
                Pass ``FullyAsyncLLMServerClient`` for abort-resume support.
            **kwargs: Forwarded to the client constructor.
        """
        return client_cls(
            config=self.config,
            load_balancer_handle=self.global_load_balancer,
            cacheblend_coordinator_handle=self.global_cacheblend_coordinator,
            **kwargs,
        )

    def get_addresses(self) -> list[str]:
        """Get the OpenAI chat completion API http addresses of the LLM server replicas."""
        return self.server_addresses

    def get_replicas(self) -> list[RolloutReplica]:
        """Get the LLM server replicas."""
        return self.rollout_replicas

    @auto_await
    async def start_profile(self, **kwargs):
        """Start profiling on all rollout replicas."""
        await asyncio.gather(*[replica.start_profile(**kwargs) for replica in self.rollout_replicas])

    @auto_await
    async def stop_profile(self):
        """Stop profiling on all rollout replicas."""
        await asyncio.gather(*[replica.stop_profile() for replica in self.rollout_replicas])
