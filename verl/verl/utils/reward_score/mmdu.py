"""Lightweight reward for MMDU long-form dialogue profiling."""

from __future__ import annotations

import re
from collections import Counter


_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text or "")]


def _f1(predict_str: str, ground_truth: str) -> float:
    pred = _tokens(predict_str)
    gold = _tokens(ground_truth)
    if not pred or not gold:
        return 0.0
    common = Counter(pred) & Counter(gold)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def _length_reward(predict_str: str) -> float:
    token_count = len(_tokens(predict_str))
    if token_count < 16:
        return 0.0
    if token_count >= 64:
        return 1.0
    return (token_count - 16) / 48


def compute_score(predict_str: str, ground_truth: str, format_score: float = 0.1) -> float:
    return (1.0 - format_score) * _f1(predict_str, ground_truth) + format_score * _length_reward(predict_str)
