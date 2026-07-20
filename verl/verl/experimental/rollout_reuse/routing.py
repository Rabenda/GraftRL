"""Routing helpers that keep one GRPO group inside one artifact owner."""

from __future__ import annotations

from collections.abc import Sequence


def group_preserving_slices(group_ids: Sequence[object], max_partitions: int) -> list[slice]:
    """Split contiguous GRPO groups without placing one group in two workers."""
    if max_partitions <= 0:
        raise ValueError("max_partitions must be positive")
    if not group_ids:
        return []

    spans: list[tuple[int, int]] = []
    seen: set[object] = set()
    start = 0
    for index in range(1, len(group_ids) + 1):
        boundary = index == len(group_ids) or group_ids[index] != group_ids[start]
        if not boundary:
            continue
        group_id = group_ids[start]
        if group_id in seen:
            raise ValueError("GRPO group ids must be contiguous before group-preserving routing")
        seen.add(group_id)
        spans.append((start, index))
        start = index

    partition_count = min(max_partitions, len(spans))
    partitions: list[slice] = []
    span_index = 0
    for partition_index in range(partition_count):
        first = spans[span_index][0]
        partitions_left = partition_count - partition_index
        groups_left = len(spans) - span_index
        if partitions_left == 1:
            partitions.append(slice(first, spans[-1][1]))
            break

        rows_left = len(group_ids) - first
        target = rows_left / partitions_left
        end = first
        max_groups = groups_left - (partitions_left - 1)
        used_groups = 0
        while used_groups < max_groups:
            candidate_end = spans[span_index + used_groups][1]
            if used_groups > 0 and abs((end - first) - target) <= abs((candidate_end - first) - target):
                break
            end = candidate_end
            used_groups += 1
        if used_groups == 0:
            end = spans[span_index][1]
            used_groups = 1
        partitions.append(slice(first, end))
        span_index += used_groups

    return partitions
