#!/usr/bin/env python3
"""Discover positive SGLang rollout branches for partial-window parity.

Reads trainer rollout JSONL produced by ``trainer.rollout_data_dir`` plus the
image-dump manifest. Unlike HF offline discovery, this uses the actual SGLang
rollout response and score, so positives map back to real GRPO branches.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", re.DOTALL)
_FINAL_RE = re.compile(
    r"(?:\*\*Final Answer:\*\*|Final Answer:|FINAL ANSWER:|ANSWER:)\s*(.+?)(?:\n\n|\n*$|\.?\s*TERMINATE)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class PositiveRolloutPair:
    source_label: str
    dump_dir: str
    use_diversified_oracle: bool
    group_uid: str
    dataset_row: int
    target_request_id: str
    donor_request_id: str
    ground_truth: str
    baseline_reward: float
    baseline_answer: str
    extracted_answer: str
    answer_source: str
    logged_score_source: str
    response_text_reward: float
    output_reward: float
    response_text_score_delta: float
    output_score_delta: float
    rollout_step: int
    num_turns: int
    refocus_source: str


def _iter_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _rollout_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(path.glob("*.jsonl"))


def _load_refocus_branches(
    manifest_path: Path,
    *,
    turn: int = 1,
    roles: tuple[str, ...] = ("refocus_output", "zoom_output"),
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("turn", -1)) != turn or row.get("role") not in roles:
                continue
            uid = str(row.get("uid") or "")
            rid = str(row.get("request_id") or "")
            if uid and rid:
                out.setdefault(uid, set()).add(rid)
    return out


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_answer(text: str) -> str:
    boxed = _BOXED_RE.findall(text or "")
    if boxed:
        return boxed[-1].strip()
    matches = _FINAL_RE.findall(text or "")
    if not matches:
        return ""
    ans = re.sub(r"\s+", " ", matches[-1]).strip("`*\"' ")
    ans = re.sub(r"\s*TERMINATE\s*$", "", ans, flags=re.IGNORECASE)
    return ans.rstrip(".").strip()


def _first_present(row: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    for key in keys:
        val = row.get(key)
        if val not in (None, ""):
            return val
    return default


def _first_present_with_key(row: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> tuple[str, Any]:
    for key in keys:
        val = row.get(key)
        if val not in (None, ""):
            return key, val
    return "", default


def _maybe_compute_refocus_score(answer: str, ground_truth: str, enabled: bool) -> float:
    if not enabled or not ground_truth:
        return -1.0
    from verl.utils.reward_score.refocus_chart import compute_score

    return float(compute_score(answer, ground_truth))


def _score_delta(score: float, logged_score: float) -> float:
    return score - logged_score if score >= 0 else 0.0


def discover_pairs(
    *,
    rollout_rows: list[dict[str, Any]],
    refocus_by_uid: dict[str, set[str]],
    dump_dir: Path,
    source_label: str,
    use_diversified_oracle: bool,
    min_score: float,
    max_positive: int,
    verify_refocus_score: bool,
    answer_field: str,
) -> list[PositiveRolloutPair]:
    positives: list[PositiveRolloutPair] = []
    seen: set[tuple[str, str]] = set()
    for row in rollout_rows:
        score_key, raw_score = _first_present_with_key(row, ("compute_score", "reward_score", "score"), 0.0)
        score = _as_float(raw_score)
        if score <= min_score:
            continue
        uid = str(_first_present(row, ("agent_uid", "uid"), ""))
        target_rid = str(_first_present(row, ("request_id", "agent_request_id"), ""))
        if not uid or not target_rid or (uid, target_rid) in seen:
            continue
        refocus_ids = sorted(refocus_by_uid.get(uid, set()))
        if target_rid not in refocus_ids:
            continue
        donor_rid = next((rid for rid in refocus_ids if rid != target_rid), None)
        if donor_rid is None:
            continue
        dataset_row = _as_int(_first_present(row, ("rollout_idx", "index"), -1))
        if dataset_row < 0:
            continue
        response_text = str(_first_present(row, ("final_response_text", "response_text"), ""))
        output_text = str(row.get("output") or "")
        if answer_field == "output":
            answer = output_text
            answer_source = "output"
        elif answer_field == "response_text":
            answer = response_text
            answer_source = "response_text"
        else:
            answer = response_text or output_text
            answer_source = "response_text" if response_text else "output"
        ground_truth = str(row.get("gts") or "")
        response_text_reward = _maybe_compute_refocus_score(response_text, ground_truth, verify_refocus_score)
        output_reward = _maybe_compute_refocus_score(output_text, ground_truth, verify_refocus_score)
        positives.append(
            PositiveRolloutPair(
                source_label=source_label,
                dump_dir=str(dump_dir),
                use_diversified_oracle=use_diversified_oracle,
                group_uid=uid,
                dataset_row=dataset_row,
                target_request_id=target_rid,
                donor_request_id=donor_rid,
                ground_truth=ground_truth,
                baseline_reward=score,
                baseline_answer=answer,
                extracted_answer=_extract_answer(answer),
                answer_source=answer_source,
                logged_score_source=score_key,
                response_text_reward=response_text_reward,
                output_reward=output_reward,
                response_text_score_delta=_score_delta(response_text_reward, score),
                output_score_delta=_score_delta(output_reward, score),
                rollout_step=_as_int(row.get("step"), -1),
                num_turns=_as_int(row.get("num_turns") or row.get("__num_turns__"), -1),
                refocus_source=str(row.get("vtool_refocus_source") or ""),
            )
        )
        seen.add((uid, target_rid))
        if max_positive and len(positives) >= max_positive:
            break
    return positives


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout-data", required=True, type=Path, help="rollout_data dir or a single step JSONL")
    ap.add_argument("--image-dump-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default="", type=Path)
    ap.add_argument("--source-label", default="")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--max-positive", type=int, default=0)
    ap.add_argument("--use-diversified-oracle", action="store_true")
    ap.add_argument(
        "--answer-field",
        choices=["auto", "response_text", "output"],
        default="auto",
        help="which dumped text to store as baseline_answer in pairs JSON",
    )
    ap.add_argument(
        "--verify-refocus-score",
        action="store_true",
        help="also recompute refocus_chart reward for response_text and output",
    )
    ap.add_argument(
        "--manifest-turn1-roles",
        default="refocus_output,zoom_output",
        help="comma-separated manifest roles at turn=1 that mark a usable branch (DeepEyes: zoom_output)",
    )
    args = ap.parse_args()

    rollout_paths = _rollout_paths(args.rollout_data)
    if not rollout_paths:
        raise SystemExit(f"no rollout JSONL files under {args.rollout_data}")
    manifest_path = args.image_dump_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest: {manifest_path}")

    turn1_roles = tuple(r.strip() for r in args.manifest_turn1_roles.split(",") if r.strip())
    rollout_rows = _iter_jsonl(rollout_paths)
    refocus_by_uid = _load_refocus_branches(manifest_path, roles=turn1_roles)
    source_label = args.source_label or args.image_dump_dir.name
    pairs = discover_pairs(
        rollout_rows=rollout_rows,
        refocus_by_uid=refocus_by_uid,
        dump_dir=args.image_dump_dir,
        source_label=source_label,
        use_diversified_oracle=args.use_diversified_oracle,
        min_score=args.min_score,
        max_positive=args.max_positive,
        verify_refocus_score=args.verify_refocus_score,
        answer_field=args.answer_field,
    )

    out_dir = args.out_dir or (args.rollout_data if args.rollout_data.is_dir() else args.rollout_data.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"positive_rollout_pairs_{stamp}.json"
    payload = {
        "config": {
            "rollout_data": str(args.rollout_data),
            "image_dump_dir": str(args.image_dump_dir),
            "source_label": source_label,
            "min_score": args.min_score,
            "use_diversified_oracle": args.use_diversified_oracle,
            "answer_field": args.answer_field,
            "verify_refocus_score": args.verify_refocus_score,
        },
        "scanned_rollout_rows": len(rollout_rows),
        "groups_with_refocus": len(refocus_by_uid),
        "n_positive": len(pairs),
        "pairs": [asdict(p) for p in pairs],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=== positive rollout discovery ===")
    print(f"rollout_files={len(rollout_paths)} scanned_rows={len(rollout_rows)}")
    print(f"groups_with_refocus={len(refocus_by_uid)} n_positive={len(pairs)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
