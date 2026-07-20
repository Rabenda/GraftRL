from examples.profile.workloads.mmsearch_r1.context_cache import (
    ContentArtifact,
    ContentSelectionCache,
    ContentValueCache,
)
from verl.utils.reward_score.mmsearch_r1 import compute_score


def test_content_selection_deduplicates_compacts_and_reuses_by_content():
    cache = ContentSelectionCache(max_entries=4)
    artifacts = [
        ContentArtifact("document", "a", 0, "A", "A"),
        ContentArtifact("document", "a", 1, "duplicate", "duplicate"),
        ContentArtifact("document", "b", 2, "B", "B"),
        ContentArtifact("document", "c", 3, "C", "C"),
    ]

    first = cache.select(artifacts, query_hash="query", topk=2)
    second = cache.select(list(reversed(artifacts)), query_hash="query", topk=2)

    assert [artifact.content_hash for artifact in first.artifacts] == ["a", "b"]
    assert first.candidate_count == 4
    assert first.unique_count == 3
    assert first.reduction_ratio == 0.5
    assert not first.cache_hit
    assert second.cache_hit


def test_content_value_cache_reuses_position_independent_artifacts():
    cache = ContentValueCache(max_entries=1)
    assert cache.get(("image", "a")) is None
    assert cache.set(("image", "a"), ("encoded",)) == ("encoded",)
    assert cache.get(("image", "a")) == ("encoded",)
    cache.set(("image", "b"), ("new",))
    assert cache.get(("image", "a")) is None


def test_mmsearch_reward_uses_real_turns_and_candidate_answers():
    result = compute_score(
        solution_str="ignored combined transcript",
        ground_truth="Spain",
        extra_info={
            "mmsearch_assistant_turns": [
                "<reason>need evidence</reason><search><img></search>",
                "<reason>identified it</reason><answer>Kingdom of Spain</answer>",
            ],
            "mmsearch_candidate_answers": ["Kingdom of Spain"],
        },
    )

    assert result == {
        "score": 0.91,
        "accuracy": 1.0,
        "format_score": 1.0,
        "search_count": 1,
    }


def test_mmsearch_reward_exposes_quality_drop_even_when_format_is_valid():
    result = compute_score(
        solution_str="<reason>guess</reason><answer>France</answer>",
        ground_truth="Spain",
    )

    assert result["accuracy"] == 0.0
    assert result["format_score"] == 1.0
    assert result["score"] == 0.1
