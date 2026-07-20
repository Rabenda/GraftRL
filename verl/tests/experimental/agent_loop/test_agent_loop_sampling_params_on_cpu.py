# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace

import numpy as np
import pytest

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.protocol import DataProto


@pytest.mark.asyncio
async def test_generate_sequences_forwards_ignore_eos_on_cpu() -> None:
    captured_sampling_params = []

    class _DummyWorker:
        rollout_config = SimpleNamespace(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            calculate_log_probs=False,
            ignore_eos=True,
            val_kwargs=SimpleNamespace(top_p=1.0, top_k=-1, temperature=0.0),
            agent=SimpleNamespace(default_agent_loop="test"),
        )

        async def _run_agent_loop(self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
            del trajectory, agent_name, trace, kwargs
            captured_sampling_params.append(sampling_params)
            return object()

        def _postprocess(self, outputs, *, input_non_tensor_batch, validate=False):
            del input_non_tensor_batch, validate
            return outputs

    batch = DataProto(
        non_tensor_batch={"agent_name": np.array(["test"], dtype=object)},
        meta_info={"global_steps": 0},
    )

    await AgentLoopWorker.generate_sequences(_DummyWorker(), batch)

    assert captured_sampling_params == [
        {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "logprobs": False,
            "ignore_eos": True,
        }
    ]
