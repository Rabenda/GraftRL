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


def _vlm_cacheblend_warmup_keep_steps() -> int:
    try:
        return max(1, int(os.environ.get("SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_KEEP_STEPS", "4")))
    except (TypeError, ValueError):
        return 4


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
        prefix = f"{old_step}:"
        drop = {key for key in warmed_uids if key.startswith(prefix)}
        warmed_uids.difference_update(drop)


BarrierRole = Literal["donor", "recipient", "bypass"]


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
        self._vlm_cacheblend_warmed_uids: set[str] = set()
        self._vlm_cacheblend_warmup_locks: dict[str, asyncio.Lock] = {}
        self._vlm_cacheblend_warmup_recent_steps: list[int] = []
        self._vlm_cacheblend_barrier_log_lock = asyncio.Lock()
        self._vlm_cacheblend_barrier_log_ready = False

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
        if turn != _vlm_cacheblend_target_turn():
            return None
        if global_step is None:
            return str(agent_uid)
        return f"{global_step}:{agent_uid}"

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
    ) -> None:
        target_turn = _vlm_cacheblend_target_turn()
        barrier_enabled = _vlm_cacheblend_enabled() and _vlm_cacheblend_warmup_barrier_enabled()
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
            "wait_ms": f"{wait_ms:.3f}",
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
            "donor_ready": "1" if donor_ready else "0",
            "barrier_enabled": "1" if barrier_enabled else "0",
        }
        async with self._vlm_cacheblend_barrier_log_lock:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not self._vlm_cacheblend_barrier_log_ready and not log_path.exists()
            with log_path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            self._vlm_cacheblend_barrier_log_ready = True

    async def _acquire_server(self, request_id: str) -> tuple[str, ray.actor.ActorHandle]:
        # Atomic acquire: returns (server_id, handle) in one Ray RPC.
        return await self._load_balancer.acquire_server.remote(request_id=request_id)

    def _release_server(self, server_id: str) -> None:
        # Fire-and-forget: release is just a counter decrement, no need to await.
        # Awaiting here risks blocking the finally clause if the LB actor is unresponsive.
        self._load_balancer.release_server.remote(server_id=server_id)

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
        routing_request_id = request_id
        if (_grpo_sim_cache_enabled() or _vlm_cacheblend_enabled()) and agent_uid:
            routing_request_id = f"grpo_agent_uid:{agent_uid}"

        warmup_global_step = kwargs.get("training_global_step", kwargs.get("global_step"))
        self._maybe_prune_warmed_uids(warmup_global_step)
        warmup_key = self._vlm_cacheblend_warmup_key(agent_uid, agent_turn, warmup_global_step)

        async def _call_server() -> TokenOutput:
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

                return await server.generate.remote(
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
            finally:
                self._release_server(server_id)

        if warmup_key is None:
            await self._log_cacheblend_barrier_event(
                request_id=str(request_id),
                agent_uid=agent_uid,
                agent_turn=agent_turn,
                rollout_idx=rollout_idx,
                global_step=warmup_global_step,
                warmup_key=None,
                barrier_role="bypass",
                wait_ms=0.0,
                donor_ready=False,
            )
            return await _call_server()

        if warmup_key in self._vlm_cacheblend_warmed_uids:
            await self._log_cacheblend_barrier_event(
                request_id=str(request_id),
                agent_uid=agent_uid,
                agent_turn=agent_turn,
                rollout_idx=rollout_idx,
                global_step=warmup_global_step,
                warmup_key=warmup_key,
                barrier_role="recipient",
                wait_ms=0.0,
                donor_ready=True,
            )
            return await _call_server()

        wait_start = time.perf_counter()
        async with self._vlm_cacheblend_warmup_lock(warmup_key):
            wait_ms = (time.perf_counter() - wait_start) * 1000.0
            if warmup_key not in self._vlm_cacheblend_warmed_uids:
                output = await _call_server()
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
                    donor_ready=True,
                )
                return output
            await self._log_cacheblend_barrier_event(
                request_id=str(request_id),
                agent_uid=agent_uid,
                agent_turn=agent_turn,
                rollout_idx=rollout_idx,
                global_step=warmup_global_step,
                warmup_key=warmup_key,
                barrier_role="recipient",
                wait_ms=wait_ms,
                donor_ready=True,
            )
        return await _call_server()


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
