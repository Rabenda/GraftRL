#!/usr/bin/env python3
"""Answer / reward parity for partial-window ViT reuse.

Both baseline and partial paths use the same ``model.generate(pixel_values=...)`` route.
Partial reuse is injected by temporarily patching ``visual.forward``: chart image stays
full ViT; refocus image uses partial-window reuse against the donor cache.

Reports:
  - raw answer / token parity vs baseline
  - extracted final/boxed answer parity vs baseline and vs ground truth
  - reward parity vs baseline
  - aggregate over all pairs and over baseline-reward>0 subset
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image

# Allow sibling imports when run as a script.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_partial_window_reuse_sweep import (  # noqa: E402
    _image_embeds_from_image,
    discover_valid_groups,
    visual_partial_window,
)
from run_phase2_token_replacement import (  # noqa: E402
    build_pairs,
    build_processor_inputs,
    build_turn1_messages,
    greedy_generate,
    image_dump_paths,
    load_model_and_processor,
    load_parquet_row,
)

THRESHOLDS_DEFAULT = (0.95, 0.98, 0.99)

_FINAL_ANSWER_PATTERNS = (
    re.compile(r"\\boxed\{([^}]*)\}", re.DOTALL),
    re.compile(r"\*\*Final Answer:\*\*\s*(.+?)(?:\n\n|\n*$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"Final Answer:\s*(.+?)(?:\n\n|\n*$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"FINAL ANSWER:\s*(.+?)(?:\.\s*TERMINATE|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"ANSWER:\s*(.+?)(?:\.\s*FINAL ANSWER|\.?\s*TERMINATE|$)", re.IGNORECASE | re.DOTALL),
)


def split_images_by_grid(
    pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Split concatenated patch rows into per-image (patches, grid_row) segments."""
    if image_grid_thw.numel() == 0:
        return [(pixel_values, image_grid_thw)]
    grid = image_grid_thw.reshape(-1, 3)
    if grid.shape[0] <= 1:
        return [(pixel_values, grid if grid.dim() == 2 else image_grid_thw)]
    patch_counts = (grid[:, 0] * grid[:, 1] * grid[:, 2]).tolist()
    segments: list[tuple[torch.Tensor, torch.Tensor]] = []
    start = 0
    for cnt in patch_counts:
        n = int(cnt)
        end = start + n
        segments.append((pixel_values[start:end], grid[len(segments) : len(segments) + 1]))
        start = end
    if start != pixel_values.shape[0]:
        return [(pixel_values, grid)]
    return segments


def _replace_visual_pooler_output(full_out: Any, merged: torch.Tensor) -> Any:
    if hasattr(full_out, "pooler_output"):
        full_out.pooler_output = merged
        return full_out
    if isinstance(full_out, tuple):
        extra = full_out[1:] if len(full_out) > 1 else ()
        return (merged, *extra)
    return merged


@contextmanager
def partial_visual_forward_hook(
    visual,
    *,
    donor_cache: dict[str, Any],
    threshold: float,
):
    """Patch visual.forward so the 2nd image uses partial-window reuse."""
    from verl.utils.transformers_compat import unpack_visual_output

    orig_forward = visual.forward

    def patched_forward(pixel_values, grid_thw=None, **kwargs):
        segments = split_images_by_grid(pixel_values, grid_thw)
        if len(segments) != 2:
            return orig_forward(pixel_values, grid_thw=grid_thw, **kwargs)

        (pv0, g0), (pv1, g1) = segments
        emb0, _ = unpack_visual_output(orig_forward(pv0, grid_thw=g0, **kwargs))
        partial_emb, _ = visual_partial_window(
            visual,
            pv1,
            g1,
            donor_cache=donor_cache,
            threshold=threshold,
            profile=False,
        )
        merged = torch.cat(
            [emb0, partial_emb.to(dtype=emb0.dtype, device=emb0.device)],
            dim=0,
        )
        full_out = orig_forward(pixel_values, grid_thw=grid_thw, **kwargs)
        return _replace_visual_pooler_output(full_out, merged)

    visual.forward = patched_forward
    try:
        yield
    finally:
        visual.forward = orig_forward


def extract_final_answer(text: str) -> str:
    """Extract final/boxed answer text for parity (boxed first, then FINAL ANSWER heuristics)."""
    from mathruler.grader import extract_boxed_content

    boxed = extract_boxed_content(text)
    if boxed is not None and str(boxed).strip() and str(boxed).lower() != "none":
        return str(boxed).strip()

    for pat in _FINAL_ANSWER_PATTERNS:
        matches = pat.findall(text)
        if matches:
            ans = matches[-1].strip()
            ans = re.sub(r"\.\s*TERMINATE.*$", "", ans, flags=re.IGNORECASE)
            ans = re.sub(r"\s*TERMINATE\s*$", "", ans, flags=re.IGNORECASE)
            ans = re.sub(r"\s+", " ", ans)
            ans = ans.strip("`*\"' ")
            ans = ans.rstrip(".")
            if ans:
                return ans
    return ""


def normalize_extracted_answer(text: str) -> str:
    from verl.utils.reward_score.refocus_chart import _normalize_answer

    return _normalize_answer(text)


def acc_on_extracted(extracted: str, ground_truth: str) -> float:
    from verl.utils.reward_score.refocus_chart import acc_reward_chart

    if not extracted:
        return 0.0
    return float(acc_reward_chart(extracted, ground_truth, use_boxed=False))


def compute_reward(answer: str, ground_truth: str) -> float:
    from verl.utils.reward_score.refocus_chart import compute_score

    return float(compute_score(answer, ground_truth))


def pair_key(row: "ParityRow") -> tuple[str, str, str]:
    return (row.group_uid, row.target_request_id, row.donor_request_id)


@dataclass
class ParityRow:
    source_label: str
    dump_dir: str
    group_uid: str
    dataset_row: int
    target_request_id: str
    donor_request_id: str
    threshold: str
    ground_truth: str
    answer: str
    extracted_answer: str
    reward: float
    rollout_baseline_answer: str
    rollout_baseline_extracted: str
    rollout_baseline_reward: float
    pair_baseline_reward: float
    answer_match_baseline: bool
    token_match_baseline: bool
    extracted_match_baseline: bool
    answer_match_rollout_baseline: bool
    extracted_match_rollout_baseline: bool
    boxed_acc_vs_gt: float
    reward_match_baseline: bool
    reward_match_rollout_baseline: bool
    reward_delta_vs_baseline: float
    reward_delta_vs_rollout_baseline: float
    acc_vs_gt: float


def _normalize_optional_rollout_answer(text: str | None) -> str:
    return str(text or "").strip()


def _optional_float(value: Any, default: float = -1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rollout_comparison_fields(
    *,
    answer: str,
    extracted_answer: str,
    reward: float,
    rollout_answer: str,
    rollout_extracted: str,
    rollout_reward: float,
) -> dict[str, bool | float]:
    has_rollout_answer = bool(rollout_answer)
    has_rollout_reward = rollout_reward >= 0.0
    return {
        "answer_match_rollout_baseline": has_rollout_answer and answer.strip() == rollout_answer.strip(),
        "extracted_match_rollout_baseline": has_rollout_answer
        and normalize_extracted_answer(extracted_answer) == normalize_extracted_answer(rollout_extracted),
        "reward_match_rollout_baseline": has_rollout_reward and abs(reward - rollout_reward) < 1e-9,
        "reward_delta_vs_rollout_baseline": reward - rollout_reward if has_rollout_reward else 0.0,
    }



def run_pair(
    model,
    processor,
    device,
    *,
    source_label: str,
    row: dict,
    dump_dir: Path,
    group_uid: str,
    dataset_row: int,
    target_rid: str,
    donor_rid: str,
    thresholds: tuple[float, ...],
    max_new_tokens: int,
    use_diversified_oracle: bool = True,
    rollout_baseline_answer: str = "",
    rollout_baseline_reward: float = -1.0,
    rollout_baseline_extracted: str = "",
) -> list[ParityRow]:
    ground_truth = str((row.get("reward_model") or {}).get("ground_truth", ""))
    if not ground_truth:
        raise ValueError("missing reward_model.ground_truth")

    chart_path, refocus_t_path = image_dump_paths(dump_dir, group_uid, target_rid)
    _, refocus_d_path = image_dump_paths(dump_dir, group_uid, donor_rid)
    chart = Image.open(chart_path).convert("RGB")
    target_refocus = Image.open(refocus_t_path).convert("RGB")
    donor_refocus = Image.open(refocus_d_path).convert("RGB")

    messages, _ = build_turn1_messages(
        row,
        chart_image=chart,
        refocus_image=target_refocus,
        request_id=target_rid,
        use_diversified_oracle=use_diversified_oracle,
    )
    prompt_inputs = build_processor_inputs(processor, messages, [chart, target_refocus])
    input_ids = prompt_inputs["input_ids"]
    attention_mask = prompt_inputs["attention_mask"]

    baseline_ids = greedy_generate(
        model,
        processor,
        input_ids=input_ids,
        attention_mask=attention_mask,
        mm_inputs=prompt_inputs,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    baseline_text = processor.tokenizer.decode(baseline_ids, skip_special_tokens=True)
    baseline_reward = compute_reward(baseline_text, ground_truth)
    baseline_extracted = extract_final_answer(baseline_text)
    rollout_baseline_answer = _normalize_optional_rollout_answer(rollout_baseline_answer)
    rollout_baseline_extracted = (
        _normalize_optional_rollout_answer(rollout_baseline_extracted)
        or extract_final_answer(rollout_baseline_answer)
    )
    rollout_baseline_reward = float(rollout_baseline_reward)
    baseline_rollout_cmp = _rollout_comparison_fields(
        answer=baseline_text,
        extracted_answer=baseline_extracted,
        reward=baseline_reward,
        rollout_answer=rollout_baseline_answer,
        rollout_extracted=rollout_baseline_extracted,
        rollout_reward=rollout_baseline_reward,
    )

    rows: list[ParityRow] = [
        ParityRow(
            source_label=source_label,
            dump_dir=str(dump_dir),
            group_uid=group_uid,
            dataset_row=dataset_row,
            target_request_id=target_rid,
            donor_request_id=donor_rid,
            threshold="baseline",
            ground_truth=ground_truth,
            answer=baseline_text,
            extracted_answer=baseline_extracted,
            reward=baseline_reward,
            rollout_baseline_answer=rollout_baseline_answer,
            rollout_baseline_extracted=rollout_baseline_extracted,
            rollout_baseline_reward=rollout_baseline_reward,
            pair_baseline_reward=baseline_reward,
            answer_match_baseline=True,
            token_match_baseline=True,
            extracted_match_baseline=True,
            answer_match_rollout_baseline=bool(baseline_rollout_cmp["answer_match_rollout_baseline"]),
            extracted_match_rollout_baseline=bool(baseline_rollout_cmp["extracted_match_rollout_baseline"]),
            boxed_acc_vs_gt=acc_on_extracted(baseline_extracted, ground_truth),
            reward_match_baseline=True,
            reward_match_rollout_baseline=bool(baseline_rollout_cmp["reward_match_rollout_baseline"]),
            reward_delta_vs_baseline=0.0,
            reward_delta_vs_rollout_baseline=float(baseline_rollout_cmp["reward_delta_vs_rollout_baseline"]),
            acc_vs_gt=1.0 if baseline_reward >= 0.99 else 0.0,
        )
    ]

    _, donor_cache, _ = _image_embeds_from_image(model, processor, donor_refocus, device)
    visual = model.model.visual

    for th in thresholds:
        with partial_visual_forward_hook(visual, donor_cache=donor_cache, threshold=th):
            partial_ids = greedy_generate(
                model,
                processor,
                input_ids=input_ids,
                attention_mask=attention_mask,
                mm_inputs=prompt_inputs,
                max_new_tokens=max_new_tokens,
                device=device,
            )
        partial_text = processor.tokenizer.decode(partial_ids, skip_special_tokens=True)
        partial_extracted = extract_final_answer(partial_text)
        partial_reward = compute_reward(partial_text, ground_truth)
        partial_rollout_cmp = _rollout_comparison_fields(
            answer=partial_text,
            extracted_answer=partial_extracted,
            reward=partial_reward,
            rollout_answer=rollout_baseline_answer,
            rollout_extracted=rollout_baseline_extracted,
            rollout_reward=rollout_baseline_reward,
        )
        rows.append(
            ParityRow(
                source_label=source_label,
                dump_dir=str(dump_dir),
                group_uid=group_uid,
                dataset_row=dataset_row,
                target_request_id=target_rid,
                donor_request_id=donor_rid,
                threshold=f"{th:.2f}",
                ground_truth=ground_truth,
                answer=partial_text,
                extracted_answer=partial_extracted,
                reward=partial_reward,
                rollout_baseline_answer=rollout_baseline_answer,
                rollout_baseline_extracted=rollout_baseline_extracted,
                rollout_baseline_reward=rollout_baseline_reward,
                pair_baseline_reward=baseline_reward,
                answer_match_baseline=partial_text.strip() == baseline_text.strip(),
                token_match_baseline=partial_ids == baseline_ids,
                extracted_match_baseline=normalize_extracted_answer(partial_extracted)
                == normalize_extracted_answer(baseline_extracted),
                answer_match_rollout_baseline=bool(partial_rollout_cmp["answer_match_rollout_baseline"]),
                extracted_match_rollout_baseline=bool(partial_rollout_cmp["extracted_match_rollout_baseline"]),
                boxed_acc_vs_gt=acc_on_extracted(partial_extracted, ground_truth),
                reward_match_baseline=abs(partial_reward - baseline_reward) < 1e-9,
                reward_match_rollout_baseline=bool(partial_rollout_cmp["reward_match_rollout_baseline"]),
                reward_delta_vs_baseline=partial_reward - baseline_reward,
                reward_delta_vs_rollout_baseline=float(partial_rollout_cmp["reward_delta_vs_rollout_baseline"]),
                acc_vs_gt=1.0 if partial_reward >= 0.99 else 0.0,
            )
        )
    return rows


def _positive_pair_keys(rows: list[ParityRow], *, min_baseline_reward: float) -> set[tuple[str, str, str]]:
    return {
        pair_key(r)
        for r in rows
        if r.threshold == "baseline"
        and (r.rollout_baseline_reward if r.rollout_baseline_reward >= 0 else r.reward) > min_baseline_reward
    }


def _filter_rows(rows: list[ParityRow], pair_keys: set[tuple[str, str, str]]) -> list[ParityRow]:
    return [r for r in rows if pair_key(r) in pair_keys]


def aggregate(
    rows: list[ParityRow],
    *,
    threshold_labels: list[str],
    subset_name: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label in threshold_labels:
        rs = [r for r in rows if r.threshold == label]
        if not rs:
            continue
        n = len(rs)
        is_baseline = label == "baseline"
        rollout_answer_rows = [r for r in rs if r.rollout_baseline_answer]
        rollout_reward_rows = [r for r in rs if r.rollout_baseline_reward >= 0]
        out.append(
            {
                "subset": subset_name,
                "threshold": label,
                "n_pairs": n,
                "answer_match_baseline_rate": (
                    sum(1 for r in rs if r.answer_match_baseline) / n if not is_baseline else 1.0
                ),
                "token_match_baseline_rate": (
                    sum(1 for r in rs if r.token_match_baseline) / n if not is_baseline else 1.0
                ),
                "extracted_match_baseline_rate": (
                    sum(1 for r in rs if r.extracted_match_baseline) / n if not is_baseline else 1.0
                ),
                "answer_match_rollout_baseline_rate": (
                    sum(1 for r in rollout_answer_rows if r.answer_match_rollout_baseline)
                    / len(rollout_answer_rows)
                    if rollout_answer_rows
                    else ""
                ),
                "extracted_match_rollout_baseline_rate": (
                    sum(1 for r in rollout_answer_rows if r.extracted_match_rollout_baseline)
                    / len(rollout_answer_rows)
                    if rollout_answer_rows
                    else ""
                ),
                "reward_match_baseline_rate": (
                    sum(1 for r in rs if r.reward_match_baseline) / n if not is_baseline else 1.0
                ),
                "reward_match_rollout_baseline_rate": (
                    sum(1 for r in rollout_reward_rows if r.reward_match_rollout_baseline)
                    / len(rollout_reward_rows)
                    if rollout_reward_rows
                    else ""
                ),
                "reward_delta_vs_baseline_mean": sum(r.reward_delta_vs_baseline for r in rs) / n,
                "reward_delta_vs_rollout_baseline_mean": (
                    sum(r.reward_delta_vs_rollout_baseline for r in rollout_reward_rows) / len(rollout_reward_rows)
                    if rollout_reward_rows
                    else ""
                ),
                "reward_mean": sum(r.reward for r in rs) / n,
                "rollout_baseline_reward_mean": (
                    sum(r.rollout_baseline_reward for r in rollout_reward_rows) / len(rollout_reward_rows)
                    if rollout_reward_rows
                    else ""
                ),
                "acc_vs_gt_rate": sum(r.acc_vs_gt for r in rs) / n,
                "boxed_acc_vs_gt_rate": sum(r.boxed_acc_vs_gt for r in rs) / n,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default="/data/refocus_chart_multiturn_oracle_changed/train.parquet")
    ap.add_argument(
        "--image-dump-dir",
        default="profile_logs_vtool_chart_diversified/image_dump_vtool_chart_bs64_n4_diversified_baseline",
    )
    ap.add_argument("--out-dir", default="profile_logs_vtool_chart_diversified/partial_window_parity")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--thresholds", default="0.95,0.98,0.99")
    ap.add_argument("--max-pairs", type=int, default=50)
    ap.add_argument("--pairs-json", default="", help="JSON from run_discover_positive_baseline_pairs.py")
    ap.add_argument("--pairs-per-group", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--min-baseline-reward",
        type=float,
        default=0.0,
        help="baseline reward must be > this value for the positive subset aggregate",
    )
    args = ap.parse_args()

    thresholds = tuple(float(x) for x in args.thresholds.split(",") if x.strip())
    threshold_labels = ["baseline"] + [f"{th:.2f}" for th in thresholds]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor, device = load_model_and_processor(args)
    all_rows: list[ParityRow] = []
    skipped: list[str] = []
    pair_count = 0

    if args.pairs_json:
        payload = json.loads(Path(args.pairs_json).read_text(encoding="utf-8"))
        pair_specs = payload.get("pairs") or []
        if not pair_specs:
            raise SystemExit(f"no pairs in {args.pairs_json}")
        print(f"Loaded {len(pair_specs)} positive pairs from {args.pairs_json}")
        for i, spec in enumerate(pair_specs):
            try:
                dump_dir = Path(spec["dump_dir"])
                row = load_parquet_row(args.parquet, int(spec["dataset_row"]))
                print(
                    f"[pair {i + 1}/{len(pair_specs)}] src={spec.get('source_label','?')} "
                    f"uid={spec['group_uid'][:8]} reward={spec.get('baseline_reward', '?')}"
                )
                all_rows.extend(
                    run_pair(
                        model,
                        processor,
                        device,
                        source_label=str(spec.get("source_label", "unknown")),
                        row=row,
                        dump_dir=dump_dir,
                        group_uid=spec["group_uid"],
                        dataset_row=int(spec["dataset_row"]),
                        target_rid=spec["target_request_id"],
                        donor_rid=spec["donor_request_id"],
                        thresholds=thresholds,
                        max_new_tokens=args.max_new_tokens,
                        use_diversified_oracle=bool(spec.get("use_diversified_oracle", True)),
                        rollout_baseline_answer=str(spec.get("baseline_answer") or ""),
                        rollout_baseline_reward=_optional_float(spec.get("baseline_reward"), -1.0),
                        rollout_baseline_extracted=str(spec.get("extracted_answer") or ""),
                    )
                )
                pair_count += 1
            except Exception as exc:  # noqa: BLE001
                skipped.append(f"{spec.get('group_uid', '?')[:8]}: {type(exc).__name__}: {exc}")
                print(f"[skip] {skipped[-1]}")
    else:
        dump_dir = Path(args.image_dump_dir)
        specs = discover_valid_groups(dump_dir, args.parquet, min_branches=2)
        for spec in specs:
            if pair_count >= args.max_pairs:
                break
            try:
                row = load_parquet_row(args.parquet, spec.dataset_row)
                pairs = build_pairs(spec.refocus_request_ids, pairs_per_group=args.pairs_per_group, seed=args.seed)
                for target_rid, donor_rid in pairs:
                    if pair_count >= args.max_pairs:
                        break
                    print(f"[pair {pair_count + 1}] uid={spec.group_uid[:8]} target={target_rid[:8]} donor={donor_rid[:8]}")
                    all_rows.extend(
                        run_pair(
                            model,
                            processor,
                            device,
                            source_label="default",
                            row=row,
                            dump_dir=dump_dir,
                            group_uid=spec.group_uid,
                            dataset_row=spec.dataset_row,
                            target_rid=target_rid,
                            donor_rid=donor_rid,
                            thresholds=thresholds,
                            max_new_tokens=args.max_new_tokens,
                            use_diversified_oracle=True,
                        )
                    )
                    pair_count += 1
            except Exception as exc:  # noqa: BLE001
                skipped.append(f"{spec.group_uid[:8]}: {type(exc).__name__}: {exc}")
                print(f"[skip] {skipped[-1]}")

    if pair_count == 0:
        raise SystemExit("no parity rows produced")

    positive_keys = _positive_pair_keys(all_rows, min_baseline_reward=args.min_baseline_reward)
    positive_rows = _filter_rows(all_rows, positive_keys)

    agg_all = aggregate(all_rows, threshold_labels=threshold_labels, subset_name="all")
    agg_positive = aggregate(positive_rows, threshold_labels=threshold_labels, subset_name="baseline_reward_gt")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"partial_window_parity_{stamp}.csv"
    agg_path = out_dir / f"partial_window_parity_{stamp}_agg.csv"
    agg_pos_path = out_dir / f"partial_window_parity_{stamp}_agg_positive.csv"
    json_path = out_dir / f"partial_window_parity_{stamp}.json"

    fields = list(asdict(all_rows[0]).keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in all_rows:
            w.writerow(asdict(row))

    agg_fields = list(agg_all[0].keys()) if agg_all else list(agg_positive[0].keys())
    with agg_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        w.writerows(agg_all)

    with agg_pos_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        w.writerows(agg_positive)

    json_path.write_text(
        json.dumps(
            {
                "config": vars(args),
                "skipped": skipped,
                "n_pairs_total": pair_count,
                "n_pairs_baseline_reward_gt": len(positive_keys),
                "aggregate_all": agg_all,
                "aggregate_baseline_reward_gt": agg_positive,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    def _print_agg(title: str, agg: list[dict[str, Any]]) -> None:
        def _fmt(value: Any) -> str:
            if value == "" or value is None:
                return "n/a"
            return f"{float(value):.3f}"

        def _fmt_delta(value: Any) -> str:
            if value == "" or value is None:
                return "n/a"
            return f"{float(value):+.4f}"

        print(f"\n=== {title} ===")
        if not agg:
            print("(empty)")
            return
        for r in agg:
            print(
                f"{r['threshold']:>8s} n={r['n_pairs']} "
                f"ans={r['answer_match_baseline_rate']:.3f} "
                f"tok={r['token_match_baseline_rate']:.3f} "
                f"ext={r['extracted_match_baseline_rate']:.3f} "
                f"roll_ext={_fmt(r.get('extracted_match_rollout_baseline_rate'))} "
                f"rew={r['reward_match_baseline_rate']:.3f} "
                f"roll_rew={_fmt(r.get('reward_match_rollout_baseline_rate'))} "
                f"rew_delta={r['reward_delta_vs_baseline_mean']:+.4f} "
                f"roll_rew_delta={_fmt_delta(r.get('reward_delta_vs_rollout_baseline_mean'))} "
                f"acc_gt={r['acc_vs_gt_rate']:.3f} "
                f"boxed_acc={r['boxed_acc_vs_gt_rate']:.3f} "
                f"reward_mean={r['reward_mean']:.4f}"
            )

    _print_agg("parity aggregate (all)", agg_all)
    _print_agg(f"parity aggregate (baseline reward > {args.min_baseline_reward})", agg_positive)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {agg_path}")
    print(f"Wrote {agg_pos_path}")


if __name__ == "__main__":
    main()
