#!/usr/bin/env python3
"""Discover turn1 refocus pairs whose offline baseline compute_score > 0.

Scans multiple image-dump sources (not the fixed 51 diversified pairs).
Uses greedy decode first, then rollout-like sampling (temperature + num_samples).
Outputs JSON for run_partial_window_parity.py --pairs-json.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_partial_window_parity import compute_reward, extract_final_answer  # noqa: E402
from run_partial_window_reuse_sweep import discover_valid_groups  # noqa: E402
from run_phase2_token_replacement import (  # noqa: E402
    build_processor_inputs,
    build_turn1_messages,
    image_dump_paths,
    load_model_and_processor,
    load_parquet_row,
)

DEFAULT_DUMP_SOURCES = (
    {
        "label": "diversified_grpo_cache_sim0985",
        "dump_dir": "profile_logs_vtool_chart_diversified/image_dump_vtool_chart_bs64_n4_diversified_grpo_cache_sim0985",
        "use_diversified_oracle": True,
    },
    {
        "label": "vtool_chart_clean",
        "dump_dir": "profile_logs_vtool_chart_clean/image_dump_vtool_chart_bs64_n4",
        "use_diversified_oracle": False,
    },
    {
        "label": "vtool_chart",
        "dump_dir": "profile_logs_vtool_chart/image_dump_vtool_chart_bs64_n4",
        "use_diversified_oracle": False,
    },
    {
        "label": "model_refocus",
        "dump_dir": "profile_logs_vtool_chart_model_refocus/image_dump_vtool_chart_bs64_n4_model_refocus",
        "use_diversified_oracle": False,
    },
    {
        "label": "diversified_baseline",
        "dump_dir": "profile_logs_vtool_chart_diversified/image_dump_vtool_chart_bs64_n4_diversified_baseline",
        "use_diversified_oracle": True,
    },
    {
        "label": "refocus_chart_diversified",
        "dump_dir": "profile_logs_refocus_chart/image_dump_refocus_chart_multiturn_bs64_n4_diversified",
        "use_diversified_oracle": True,
    },
)


@dataclass
class PositivePairSpec:
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


def _pick_donor(refocus_ids: list[str], target_rid: str) -> str | None:
    for rid in refocus_ids:
        if rid != target_rid:
            return rid
    return None


@torch.inference_mode()
def generate_answers(
    model,
    processor,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mm_inputs: dict,
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 0.0,
    num_samples: int = 1,
    seed: int = 0,
) -> list[tuple[list[int], str, float, str]]:
    """Return up to num_samples (ids, text, reward, extracted) sorted by reward desc."""
    prompt_len = input_ids.shape[1]
    pixel_values = mm_inputs["pixel_values"].to(device=device, dtype=model.dtype)
    image_grid_thw = mm_inputs["image_grid_thw"].to(device)
    common = dict(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
    outs: list[tuple[list[int], str, float, str]] = []

    if temperature <= 0.0 or num_samples <= 1:
        gen = model.generate(**common, do_sample=False, temperature=None, top_p=None)
        ids = gen[0, prompt_len:].tolist()
        text = processor.tokenizer.decode(ids, skip_special_tokens=True)
        return [(ids, text, -1.0, extract_final_answer(text))]

    gen = model.generate(
        **common,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        num_return_sequences=num_samples,
    )
    for row in gen:
        ids = row[prompt_len:].tolist()
        text = processor.tokenizer.decode(ids, skip_special_tokens=True)
        outs.append((ids, text, -1.0, extract_final_answer(text)))
    return outs


@torch.inference_mode()
def score_target_branch(
    model,
    processor,
    device,
    *,
    row: dict,
    dump_dir: Path,
    group_uid: str,
    target_rid: str,
    use_diversified_oracle: bool,
    max_new_tokens: int,
    temperature: float,
    num_samples: int,
    seed: int,
    ground_truth: str,
    min_baseline_reward: float,
) -> tuple[float, str, str, list[int]] | None:
    chart_path, refocus_path = image_dump_paths(dump_dir, group_uid, target_rid)
    chart = Image.open(chart_path).convert("RGB")
    refocus = Image.open(refocus_path).convert("RGB")
    messages, _ = build_turn1_messages(
        row,
        chart_image=chart,
        refocus_image=refocus,
        request_id=target_rid,
        use_diversified_oracle=use_diversified_oracle,
    )
    prompt_inputs = build_processor_inputs(processor, messages, [chart, refocus])
    candidates = generate_answers(
        model,
        processor,
        input_ids=prompt_inputs["input_ids"],
        attention_mask=prompt_inputs["attention_mask"],
        mm_inputs=prompt_inputs,
        max_new_tokens=max_new_tokens,
        device=device,
        temperature=temperature,
        num_samples=num_samples,
        seed=seed,
    )
    best: tuple[float, str, str, list[int]] | None = None
    for ids, text, _, extracted in candidates:
        reward = compute_reward(text, ground_truth)
        if reward > min_baseline_reward and (best is None or reward > best[0]):
            best = (reward, text, extracted, ids)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default="/data/refocus_chart_multiturn_oracle_changed/train.parquet")
    ap.add_argument("--out-dir", default="profile_logs_vtool_chart_diversified/partial_window_parity")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0, help="rollout-like sampling temperature")
    ap.add_argument("--num-samples", type=int, default=8, help="sampled candidates per target after greedy miss")
    ap.add_argument("--skip-greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-baseline-reward", type=float, default=0.0)
    ap.add_argument("--max-positive", type=int, default=0, help="stop after N positives (0=all)")
    ap.add_argument(
        "--dump-sources-json",
        default="",
        help="optional JSON list overriding DEFAULT_DUMP_SOURCES",
    )
    args = ap.parse_args()

    if args.dump_sources_json:
        sources = json.loads(Path(args.dump_sources_json).read_text(encoding="utf-8"))
    else:
        sources = [dict(s) for s in DEFAULT_DUMP_SOURCES]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor, device = load_model_and_processor(args)
    repo_root = Path.cwd()
    positives: list[PositivePairSpec] = []
    scanned = 0
    skipped: list[str] = []

    for src in sources:
        dump_dir = repo_root / src["dump_dir"]
        if not (dump_dir / "manifest.jsonl").is_file():
            skipped.append(f"{src['label']}: missing manifest at {dump_dir}")
            continue
        label = src["label"]
        use_div = bool(src.get("use_diversified_oracle", True))
        print(f"\n[source] {label} dump={dump_dir}", flush=True)
        try:
            specs = discover_valid_groups(dump_dir, args.parquet, min_branches=2)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{src['label']}: discover failed: {exc}")
            continue
        print(f"  groups={len(specs)}")
        for spec in specs:
            try:
                row = load_parquet_row(args.parquet, spec.dataset_row)
                ground_truth = str((row.get("reward_model") or {}).get("ground_truth", ""))
                if not ground_truth:
                    continue
                for target_rid in spec.refocus_request_ids:
                    donor_rid = _pick_donor(spec.refocus_request_ids, target_rid)
                    if donor_rid is None:
                        continue
                    scanned += 1
                    if scanned == 1 or scanned % 10 == 0:
                        print(f"  scanned={scanned} positives={len(positives)}", flush=True)
                    hit = None
                    if not args.skip_greedy:
                        hit = score_target_branch(
                            model,
                            processor,
                            device,
                            row=row,
                            dump_dir=dump_dir,
                            group_uid=spec.group_uid,
                            target_rid=target_rid,
                            use_diversified_oracle=use_div,
                            max_new_tokens=args.max_new_tokens,
                            temperature=0.0,
                            num_samples=1,
                            seed=args.seed,
                            ground_truth=ground_truth,
                            min_baseline_reward=args.min_baseline_reward,
                        )
                    if hit is None and args.num_samples > 1 and args.temperature > 0:
                        hit = score_target_branch(
                            model,
                            processor,
                            device,
                            row=row,
                            dump_dir=dump_dir,
                            group_uid=spec.group_uid,
                            target_rid=target_rid,
                            use_diversified_oracle=use_div,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            num_samples=args.num_samples,
                            seed=0,
                            ground_truth=ground_truth,
                            min_baseline_reward=args.min_baseline_reward,
                        )
                    if hit is None:
                        continue
                    reward, answer, extracted, _ = hit
                    positives.append(
                        PositivePairSpec(
                            source_label=label,
                            dump_dir=str(dump_dir),
                            use_diversified_oracle=use_div,
                            group_uid=spec.group_uid,
                            dataset_row=spec.dataset_row,
                            target_request_id=target_rid,
                            donor_request_id=donor_rid,
                            ground_truth=ground_truth,
                            baseline_reward=reward,
                            baseline_answer=answer,
                            extracted_answer=extracted,
                        )
                    )
                    print(
                        f"  + reward={reward:.4f} uid={spec.group_uid[:8]} "
                        f"target={target_rid[:8]} gt={ground_truth!r} ext={extracted!r}",
                        flush=True,
                    )
                    if args.max_positive and len(positives) >= args.max_positive:
                        break
            except Exception as exc:  # noqa: BLE001
                skipped.append(f"{label}/{spec.group_uid[:8]}: {type(exc).__name__}: {exc}")
            if args.max_positive and len(positives) >= args.max_positive:
                break
        if args.max_positive and len(positives) >= args.max_positive:
            break

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"positive_baseline_pairs_{stamp}.json"
    payload = {
        "config": vars(args),
        "scanned_targets": scanned,
        "n_positive": len(positives),
        "skipped": skipped,
        "pairs": [asdict(p) for p in positives],
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== discovery summary ===")
    print(f"scanned_targets={scanned} n_positive={len(positives)}")
    if positives:
        by_src: dict[str, int] = {}
        for p in positives:
            by_src[p.source_label] = by_src.get(p.source_label, 0) + 1
        for k, v in sorted(by_src.items()):
            print(f"  {k}: {v}")
    else:
        print("  no positive baseline pairs found")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
