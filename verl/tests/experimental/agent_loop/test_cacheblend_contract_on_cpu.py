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

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LLM_SERVER_SOURCE = REPO_ROOT / "verl/workers/rollout/llm_server.py"
SGLANG_SERVER_SOURCE = REPO_ROOT / "verl/workers/rollout/sglang_rollout/async_sglang_server.py"
REFOCUS_PROFILE_SCRIPT = REPO_ROOT / "examples/profile/workloads/geo3k/run_geo3k_refocus_profile.sh"
PROFILE_AGENT_LOOPS = [
    REPO_ROOT / "examples/profile/workloads/geo3k/geo3k_refocus_agent_loop.py",
    REPO_ROOT / "examples/profile/workloads/sokoban/sokoban_agent_loop.py",
    REPO_ROOT / "examples/profile/workloads/deepeyes/deepeyes_agent_loop.py",
    REPO_ROOT / "examples/profile/shared/agent/vtool_agent_loop.py",
]


def test_cacheblend_barrier_timeout_is_not_silent_fallback() -> None:
    source = LLM_SERVER_SOURCE.read_text(encoding="utf-8")
    assert "SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_TIMEOUT_ACTION" in source
    assert "raise TimeoutError(message)" in source
    assert "donor_ready=False" in source


def test_cacheblend_barrier_supports_bounded_wait_policy() -> None:
    source = LLM_SERVER_SOURCE.read_text(encoding="utf-8")
    assert "SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_WAIT_POLICY" in source
    assert "SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_MAX_WAIT_S" in source
    assert "_vlm_cacheblend_warmup_wait_timeout_s()" in source
    assert "continuing with bounded CacheBlend barrier" in source
    assert "if timeout_s <= 0:" in source


def test_cacheblend_barrier_log_splits_wait_and_server_time() -> None:
    source = LLM_SERVER_SOURCE.read_text(encoding="utf-8")
    assert '"barrier_wait_ms": f"{barrier_wait_ms:.3f}"' in source
    assert '"server_call_ms": f"{server_call_ms:.3f}"' in source
    assert '"wait_policy": wait_policy' in source


def test_cacheblend_metadata_registration_failure_is_logged() -> None:
    source = SGLANG_SERVER_SOURCE.read_text(encoding="utf-8")
    assert "Failed to register SGLang request metadata" in source
    assert "except Exception:\n            pass" not in source


def test_refocus_profile_disables_chunked_prefill_when_cacheblend_is_enabled() -> None:
    source = REFOCUS_PROFILE_SCRIPT.read_text(encoding="utf-8")
    assert "SGLANG_VLM_CACHEBLEND" in source
    assert "actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size" in source
    assert "CACHEBLEND_CHUNKED_PREFILL_SIZE:--1" in source


def test_refocus_profile_defaults_cacheblend_to_bounded_barrier() -> None:
    source = REFOCUS_PROFILE_SCRIPT.read_text(encoding="utf-8")
    assert "SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_WAIT_POLICY" in source
    assert "SGLANG_VLM_CACHEBLEND_WARMUP_BARRIER_MAX_WAIT_S" in source
    assert "cacheblend_barrier_log_${suffix}.csv" in source


def test_profile_agent_loops_forward_training_global_step_with_agent_uid() -> None:
    for path in PROFILE_AGENT_LOOPS:
        source = path.read_text(encoding="utf-8")
        assert 'global_step = kwargs.get("global_step")' in source, path
        assert "agent_uid=uid" in source, path
        assert "training_global_step=global_step" in source, path
