from examples.profile.workloads.mmdu.mmdu_multiturn_agent_loop import (
    MMDUMultiturnAgentLoop,
    paired_sampling_seed,
)


def test_mmdu_paired_sampling_uses_sglang_internal_seed_field() -> None:
    params = MMDUMultiturnAgentLoop._sampling_params(
        None,
        {"max_tokens": 99, "temperature": 1.0},
        128,
        seed=1234,
    )

    assert params == {
        "max_new_tokens": 128,
        "temperature": 1.0,
        "sampling_seed": 1234,
    }


def test_mmdu_paired_sampling_seed_is_stable_and_branch_local() -> None:
    first = paired_sampling_seed(
        42,
        global_step=1,
        sample_index=7,
        rollout_index=2,
        turn_index=3,
    )
    repeated = paired_sampling_seed(
        42,
        global_step=1,
        sample_index=7,
        rollout_index=2,
        turn_index=3,
    )
    other_branch = paired_sampling_seed(
        42,
        global_step=1,
        sample_index=7,
        rollout_index=3,
        turn_index=3,
    )

    assert first == repeated
    assert first != other_branch
