# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""Rule-based reward for VTOOL/Refocus_Chart single-turn profiling."""

from __future__ import annotations

import re

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_answer_tag(text: str) -> str | None:
    matches = _ANSWER_RE.findall(text or "")
    if not matches:
        return None
    return matches[-1].strip()


def format_reward(predict_str: str) -> float:
    boxed_pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    answer_pattern = re.compile(r"<think>.*</think>.*<answer>.*</answer>.*", re.DOTALL | re.IGNORECASE)
    return 1.0 if boxed_pattern.fullmatch(predict_str) or answer_pattern.fullmatch(predict_str) else 0.0


def acc_reward_chart(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    if use_boxed:
        from mathruler.grader import extract_boxed_content

        answer = extract_boxed_content(predict_str)
        if not answer or str(answer).strip().lower() == "none":
            answer = _extract_answer_tag(predict_str)
    else:
        answer = predict_str
    if answer is None:
        return 0.0

    pred = _normalize_answer(answer)
    gt = _normalize_answer(ground_truth)
    if pred == gt:
        return 1.0

    try:
        pred_num = float(pred.rstrip("%"))
        gt_num = float(gt.rstrip("%"))
        if abs(pred_num - gt_num) < 1e-3:
            return 1.0
        if gt_num != 0 and abs(pred_num / gt_num - 1.0) < 0.01:
            return 1.0
    except ValueError:
        pass

    return 0.0


def compute_score(
    predict_str: str,
    ground_truth: str,
    use_boxed: bool = True,
    format_score: float = 0.1,
) -> float:
    return (1.0 - format_score) * acc_reward_chart(predict_str, ground_truth, use_boxed) + format_score * format_reward(
        predict_str
    )
