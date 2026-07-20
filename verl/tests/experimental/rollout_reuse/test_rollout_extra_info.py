import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import _apply_rollout_extra_info


def test_rollout_extra_info_replaces_dataset_value_before_union():
    batch = DataProto(
        batch=TensorDict({"input": torch.tensor([1, 2])}, batch_size=[2]),
        non_tensor_batch={
            "extra_info": np.array([{"index": 0}, {"index": 0}], dtype=object),
            "uid": np.array(["g0", "g0"], dtype=object),
        },
    )
    branch_values = np.array(
        [
            {"index": 0, "assistant_turns": ["branch 0"]},
            {"index": 0, "assistant_turns": ["branch 1"]},
        ],
        dtype=object,
    )
    rollout = DataProto(
        batch=TensorDict({"output": torch.tensor([3, 4])}, batch_size=[2]),
        non_tensor_batch={
            "extra_info": np.array([{"index": 0}, {"index": 0}], dtype=object),
            "uid": np.array(["g0", "g0"], dtype=object),
            "rollout_extra_info": branch_values,
        },
    )

    _apply_rollout_extra_info(batch, rollout)

    assert "rollout_extra_info" not in rollout.non_tensor_batch
    assert batch.non_tensor_batch["extra_info"] is branch_values
    assert rollout.non_tensor_batch["extra_info"] is branch_values
    batch.union(rollout)
    assert batch.non_tensor_batch["extra_info"][1]["assistant_turns"] == ["branch 1"]


def test_rollout_extra_info_rejects_wrong_branch_count():
    batch = DataProto(
        batch=None,
        non_tensor_batch={"extra_info": np.array([{}, {}], dtype=object)},
    )
    rollout = DataProto(
        batch=None,
        non_tensor_batch={"rollout_extra_info": np.array([{}], dtype=object)},
    )

    with pytest.raises(ValueError, match="one value per rollout branch"):
        _apply_rollout_extra_info(batch, rollout)
