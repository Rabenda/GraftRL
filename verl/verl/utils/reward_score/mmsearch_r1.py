"""Rule reward for MMSearch-R1 multi-turn search trajectories."""

from __future__ import annotations

import json
import re
import string
from typing import Any


def normalize_answer(value: str) -> str:
    value = str(value).lower()
    value = "".join(character for character in value if character not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def _answers(ground_truth: Any, extra_info: dict[str, Any]) -> list[str]:
    values = list(ground_truth) if isinstance(ground_truth, (list, tuple)) else [ground_truth]
    candidates = extra_info.get("mmsearch_candidate_answers", [])
    if isinstance(candidates, str):
        try:
            candidates = json.loads(candidates)
        except json.JSONDecodeError:
            candidates = [candidates]
    if isinstance(candidates, (list, tuple)):
        values.extend(candidates)
    return [str(value) for value in values if value is not None]


def _extract_answer(response: str) -> str | None:
    matches = re.findall(r"<answer>(.*?)</answer>", response, flags=re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else None


def _is_direct_answer(response: str) -> bool:
    return bool(re.fullmatch(r"\s*<reason>.*</reason>.*<answer>.*</answer>\s*", response, re.DOTALL)) and all(
        response.count(tag) == 1 for tag in ("<reason>", "</reason>", "<answer>", "</answer>")
    ) and not any(tag in response for tag in ("<search><img></search>", "<text_search>", "</text_search>"))


def _is_image_search(response: str) -> bool:
    return bool(
        re.fullmatch(r"\s*<reason>.*</reason>.*<search><img></search>\s*", response, re.DOTALL)
    ) and all(
        response.count(tag) == 1 for tag in ("<reason>", "</reason>", "<search><img></search>")
    ) and not any(tag in response for tag in ("<answer>", "</answer>", "<text_search>", "</text_search>"))


def _is_text_search(response: str) -> bool:
    return bool(
        re.fullmatch(r"\s*<reason>.*</reason>.*<text_search>.*</text_search>\s*", response, re.DOTALL)
    ) and all(
        response.count(tag) == 1 for tag in ("<reason>", "</reason>", "<text_search>", "</text_search>")
    ) and not any(tag in response for tag in ("<answer>", "</answer>", "<search><img></search>"))


def _format_reward(responses: list[str]) -> tuple[float, int]:
    search_responses = responses if len(responses) == 1 else responses[:-1]
    search_count = sum(
        "<search><img></search>" in response
        or ("<text_search>" in response and "</text_search>" in response)
        for response in search_responses
    )
    valid = False
    if len(responses) == 1:
        valid = _is_direct_answer(responses[0])
    elif len(responses) == 2:
        valid = (_is_image_search(responses[0]) or _is_text_search(responses[0])) and _is_direct_answer(
            responses[1]
        )
    elif len(responses) == 3:
        valid = _is_image_search(responses[0]) and _is_text_search(responses[1]) and _is_direct_answer(
            responses[2]
        )
    return float(valid), int(search_count)


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float | int]:
    extra_info = extra_info or {}
    responses = extra_info.get("mmsearch_assistant_turns")
    if not isinstance(responses, (list, tuple)) or not responses:
        responses = [solution_str]
    responses = [str(response) for response in responses]

    answer = _extract_answer(responses[-1])
    normalized_answers = {normalize_answer(value) for value in _answers(ground_truth, extra_info)}
    normalized_prediction = normalize_answer(answer) if answer is not None else ""
    reward_mode = str(extra_info.get("reward_mode", "EM"))
    if reward_mode == "SubEM":
        accuracy = float(any(value and value in normalized_prediction for value in normalized_answers))
    else:
        accuracy = float(normalized_prediction in normalized_answers and bool(normalized_prediction))

    format_score, search_count = _format_reward(responses)
    search_penalty = float(extra_info.get("search_penalty", 0.1))
    format_weight = float(extra_info.get("format_penalty", 0.1))
    penalized_accuracy = accuracy
    if search_count and accuracy:
        penalty_count = search_count if extra_info.get("use_search_count_penalty", False) else 1
        penalized_accuracy *= (1.0 - search_penalty) ** penalty_count
    score = (1.0 - format_weight) * penalized_accuracy + format_weight * format_score
    return {
        "score": float(score),
        "accuracy": accuracy,
        "format_score": format_score,
        "search_count": search_count,
    }
