"""Reward helper for OSWorld GUI profiling datasets.

The OSWorld profiling workloads replay offline trajectories. Their rows already
carry the scalar score in ``reward_model.ground_truth``; synthetic/task-backed
rows use ``1.0`` so profiling can complete without a task-specific evaluator.
"""


def compute_score(solution_str: str, ground_truth: str | int | float, **kwargs) -> float:
    del solution_str, kwargs
    try:
        return float(ground_truth)
    except (TypeError, ValueError):
        return 0.0
