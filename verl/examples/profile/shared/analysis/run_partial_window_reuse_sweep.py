#!/usr/bin/env python3
"""Offline threshold sweep for Qwen2.5-VL partial-window ViT reuse.

This validates semantics for the partial-window prototype without running a full
rollout for every threshold. For each target/donor refocus pair:

1. compute target full ViT image tokens with real HF Qwen2.5-VL weights;
2. compute donor cache and target partial-window reuse tokens;
3. inject partial target tokens into the second image span;
4. teacher-force the original greedy answer and report ΔNLL/top1 drift.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from run_phase2_token_replacement import (
    build_inputs_embeds,
    build_pairs,
    build_processor_inputs,
    build_turn1_messages,
    discover_groups,
    forward_logits,
    greedy_generate,
    image_dump_paths,
    image_pad_spans,
    load_model_and_processor,
    load_parquet_row,
    teacher_force_metrics,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _window_token_indices(cu_window_seqlens: torch.Tensor, window_ids: torch.Tensor) -> torch.Tensor:
    parts = []
    cu = cu_window_seqlens.detach().to(device=window_ids.device)
    for wid in window_ids.tolist():
        start = int(cu[wid].item())
        end = int(cu[wid + 1].item())
        if end > start:
            parts.append(torch.arange(start, end, device=window_ids.device, dtype=torch.long))
    if not parts:
        return torch.empty(0, device=window_ids.device, dtype=torch.long)
    return torch.cat(parts, dim=0)


def _subset_cu(cu_window_seqlens: torch.Tensor, window_ids: torch.Tensor) -> torch.Tensor:
    out = [0]
    cu = cu_window_seqlens.detach().to(device=window_ids.device)
    for wid in window_ids.tolist():
        out.append(out[-1] + int(cu[wid + 1].item() - cu[wid].item()))
    return torch.tensor(out, device=window_ids.device, dtype=torch.int32)


def _prepare_visual(visual, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
    hidden = visual.patch_embed(pixel_values)
    rotary_pos_emb = visual.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = visual.get_window_index(grid_thw)
    window_index = window_index.to(device=hidden.device)
    cu_window_seqlens = torch.tensor(cu_window_seqlens, device=hidden.device, dtype=torch.int32)
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden.size()
    hidden = hidden.reshape(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
    hidden = hidden[window_index, :, :].reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.to(device=hidden.device, dtype=hidden.dtype)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :].reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    return hidden, position_embeddings, cu_seqlens, cu_window_seqlens, window_index


@torch.inference_mode()
def visual_full_with_cache(visual, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
    hidden, position_embeddings, cu_seqlens, cu_window_seqlens, window_index = _prepare_visual(
        visual, pixel_values, grid_thw
    )
    fullatt = set(int(i) for i in visual.fullatt_block_indexes)
    first_fullatt = min(fullatt) if fullatt else len(visual.blocks)
    cache = {
        "patch_hidden": hidden.detach(),
        "window_index": window_index.detach(),
        "cu_window_seqlens": cu_window_seqlens.detach(),
        "first_fullatt": int(first_fullatt),
        "prefull_layer_outputs": [],
    }
    total_windows = max(int(cu_window_seqlens.numel()) - 1, 0)
    stats = {"total_windows": total_windows, "first_fullatt": first_fullatt}
    for layer_num, blk in enumerate(visual.blocks):
        cu_now = cu_seqlens if layer_num in fullatt else cu_window_seqlens
        hidden = blk(hidden, cu_seqlens=cu_now, position_embeddings=position_embeddings)
        if layer_num < first_fullatt:
            cache["prefull_layer_outputs"].append(hidden.detach())
    hidden = visual.merger(hidden)
    reverse_indices = torch.argsort(window_index)
    return hidden[reverse_indices, :], cache, stats


@torch.inference_mode()
def visual_partial_window(
    visual,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    donor_cache: dict[str, Any],
    threshold: float,
    profile: bool,
):
    timings = {
        "prepare_ms": 0.0,
        "window_similarity_ms": 0.0,
        "index_scatter_gather_ms": 0.0,
        "target_block_compute_ms": 0.0,
        "full_attention_ms": 0.0,
        "patch_merger_ms": 0.0,
    }
    device = pixel_values.device
    _sync(device)
    t0 = time.perf_counter()
    hidden, position_embeddings, cu_seqlens, cu_window_seqlens, window_index = _prepare_visual(
        visual, pixel_values, grid_thw
    )
    _sync(device)
    timings["prepare_ms"] = (time.perf_counter() - t0) * 1000

    fullatt = set(int(i) for i in visual.fullatt_block_indexes)
    first_fullatt = min(fullatt) if fullatt else len(visual.blocks)
    total_windows = max(int(cu_window_seqlens.numel()) - 1, 0)
    if int(donor_cache.get("first_fullatt", -1)) != int(first_fullatt):
        raise ValueError("first_fullatt_mismatch")
    if not torch.equal(donor_cache["window_index"].to(window_index.device), window_index):
        raise ValueError("window_index_mismatch")
    if not torch.equal(donor_cache["cu_window_seqlens"].to(cu_window_seqlens.device), cu_window_seqlens):
        raise ValueError("cu_window_seqlens_mismatch")
    if len(donor_cache["prefull_layer_outputs"]) < first_fullatt:
        raise ValueError("donor_cache_missing_layers")

    _sync(device)
    t0 = time.perf_counter()
    donor_patch = donor_cache["patch_hidden"].to(device=hidden.device, dtype=hidden.dtype)
    patch_cos = F.cosine_similarity(hidden.float(), donor_patch.float(), dim=-1, eps=1e-6)
    vals = []
    for wid in range(total_windows):
        start = int(cu_window_seqlens[wid].item())
        end = int(cu_window_seqlens[wid + 1].item())
        vals.append(patch_cos[start:end].mean() if end > start else torch.tensor(-1.0, device=device))
    sims = torch.stack(vals).float() if vals else torch.empty(0, device=device)
    reuse_mask = sims >= threshold
    all_window_ids = torch.arange(total_windows, device=device, dtype=torch.long)
    reused_window_ids = all_window_ids[reuse_mask]
    computed_window_ids = all_window_ids[~reuse_mask]
    _sync(device)
    timings["window_similarity_ms"] = (time.perf_counter() - t0) * 1000

    hidden = hidden
    for layer_num, blk in enumerate(visual.blocks):
        if layer_num < first_fullatt:
            _sync(device)
            t0 = time.perf_counter()
            next_hidden = hidden.clone()
            if reused_window_ids.numel() > 0:
                reused_token_ids = _window_token_indices(cu_window_seqlens, reused_window_ids)
                donor_layer = donor_cache["prefull_layer_outputs"][layer_num].to(
                    device=hidden.device, dtype=hidden.dtype
                )
                next_hidden[reused_token_ids] = donor_layer[reused_token_ids]
            _sync(device)
            timings["index_scatter_gather_ms"] += (time.perf_counter() - t0) * 1000

            if computed_window_ids.numel() > 0:
                _sync(device)
                t0 = time.perf_counter()
                computed_token_ids = _window_token_indices(cu_window_seqlens, computed_window_ids)
                sub_cu = _subset_cu(cu_window_seqlens, computed_window_ids)
                sub_pos = (
                    position_embeddings[0][computed_token_ids],
                    position_embeddings[1][computed_token_ids],
                )
                sub_out = blk(hidden[computed_token_ids], cu_seqlens=sub_cu, position_embeddings=sub_pos)
                _sync(device)
                timings["target_block_compute_ms"] += (time.perf_counter() - t0) * 1000

                _sync(device)
                t0 = time.perf_counter()
                next_hidden[computed_token_ids] = sub_out
                _sync(device)
                timings["index_scatter_gather_ms"] += (time.perf_counter() - t0) * 1000
            hidden = next_hidden
            continue

        _sync(device)
        t0 = time.perf_counter()
        cu_now = cu_seqlens if layer_num in fullatt else cu_window_seqlens
        hidden = blk(hidden, cu_seqlens=cu_now, position_embeddings=position_embeddings)
        _sync(device)
        timings["full_attention_ms"] += (time.perf_counter() - t0) * 1000

    _sync(device)
    t0 = time.perf_counter()
    hidden = visual.merger(hidden)
    reverse_indices = torch.argsort(window_index)
    hidden = hidden[reverse_indices, :]
    _sync(device)
    timings["patch_merger_ms"] = (time.perf_counter() - t0) * 1000

    stats = {
        "threshold": threshold,
        "total_windows": total_windows,
        "reused_windows": int(reused_window_ids.numel()),
        "computed_windows": int(computed_window_ids.numel()),
        "reuse_ratio": float(reused_window_ids.numel() / total_windows) if total_windows else 0.0,
        "prefull_window_layers": int(first_fullatt),
        "reused_window_layer_windows": int(reused_window_ids.numel() * first_fullatt),
        "computed_window_layer_windows": int(computed_window_ids.numel() * first_fullatt),
        "window_cosine_min": float(sims.min().item()) if sims.numel() else -1.0,
        "window_cosine_mean": float(sims.mean().item()) if sims.numel() else -1.0,
        "window_cosine_max": float(sims.max().item()) if sims.numel() else -1.0,
    }
    stats.update(timings)
    stats["partial_visual_ms"] = sum(timings.values())
    return hidden, stats


def _refocus_visual_inputs(processor, image: Image.Image, device: torch.device, dtype: torch.dtype):
    inputs = processor(images=image, text="", return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    grid = inputs["image_grid_thw"].to(device)
    return pixel_values, grid


def _image_embeds_from_image(model, processor, image: Image.Image, device: torch.device):
    pixel_values, grid = _refocus_visual_inputs(processor, image, device, model.dtype)
    return visual_full_with_cache(model.model.visual, pixel_values, grid)


def _metrics_from_embeds(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | bool]:
    if a.shape != b.shape:
        return {"shape_match": False}
    cos = F.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-6)
    top1 = torch.argmax(a.float(), dim=-1).eq(torch.argmax(b.float(), dim=-1)).float()
    return {
        "shape_match": True,
        "image_token_mean_cos": float(cos.mean().item()),
        "image_token_min_cos": float(cos.min().item()),
        "image_token_max_abs_diff": float((a.float() - b.float()).abs().max().item()),
        "image_token_top1_match": float(top1.mean().item()),
    }


def _replace_second_image(base_embeds: torch.Tensor, second_span: tuple[int, int], new_embeds: torch.Tensor) -> torch.Tensor:
    start, end = second_span
    if end - start != new_embeds.shape[0]:
        raise ValueError(f"image span {end-start} != embeds {new_embeds.shape[0]}")
    out = base_embeds.clone()
    out[0, start:end] = new_embeds.to(device=out.device, dtype=out.dtype)
    return out


@dataclass
class SweepRow:
    group_uid: str
    dataset_row: int
    target_request_id: str
    donor_request_id: str
    threshold: float
    total_windows: int
    reused_windows: int
    computed_windows: int
    reuse_ratio: float
    prefull_window_layers: int
    reused_window_layer_windows: int
    computed_window_layer_windows: int
    vision_ms_full_target: float
    vision_ms_partial: float
    window_similarity_ms: float
    index_scatter_gather_ms: float
    target_block_compute_ms: float
    full_attention_ms: float
    patch_merger_ms: float
    window_cosine_mean: float
    window_cosine_max: float
    image_token_mean_cos: float
    image_token_min_cos: float
    image_token_max_abs_diff: float
    image_token_top1_match: float
    delta_nll_per_token: float
    delta_nll: float
    top1_match_rate: float
    response_len: int


def run_pair(model, processor, device, *, row: dict, dump_dir: Path, group_uid: str, dataset_row: int,
             target_rid: str, donor_rid: str, thresholds: list[float], max_new_tokens: int,
             save_embeds_dir: Path | None) -> list[SweepRow]:
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
        use_diversified_oracle=True,
    )
    prompt_inputs = build_processor_inputs(processor, messages, [chart, target_refocus])
    input_ids = prompt_inputs["input_ids"]
    attention_mask = prompt_inputs["attention_mask"]
    prompt_len = input_ids.shape[1]

    response_ids = greedy_generate(
        model, processor, input_ids=input_ids, attention_mask=attention_mask,
        mm_inputs=prompt_inputs, max_new_tokens=max_new_tokens, device=device,
    )
    resp_tensor = torch.tensor([response_ids], dtype=input_ids.dtype)
    full_ids = torch.cat([input_ids, resp_tensor], dim=1)
    full_mask = torch.ones_like(full_ids)

    logits_orig = forward_logits(
        model, processor, input_ids=full_ids, attention_mask=full_mask, mm_inputs=prompt_inputs, device=device
    )
    base_embeds = build_inputs_embeds(model, full_ids, full_mask, prompt_inputs, device)
    spans = image_pad_spans(input_ids[0], processor.image_token_id)
    if len(spans) < 2:
        raise ValueError(f"expected >=2 image spans, got {spans}")
    second_span = spans[1]

    _sync(device)
    t0 = time.perf_counter()
    target_full, _, _ = _image_embeds_from_image(model, processor, target_refocus, device)
    _sync(device)
    full_target_ms = (time.perf_counter() - t0) * 1000
    donor_full, donor_cache, _ = _image_embeds_from_image(model, processor, donor_refocus, device)

    if save_embeds_dir:
        save_embeds_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{group_uid[:8]}_{target_rid[:8]}_{donor_rid[:8]}"
        torch.save({"embedding": target_full.detach().cpu()}, save_embeds_dir / f"{prefix}_target_full.pt")
        torch.save({"embedding": donor_full.detach().cpu()}, save_embeds_dir / f"{prefix}_donor_full.pt")

    out = []
    target_pv, target_grid = _refocus_visual_inputs(processor, target_refocus, device, model.dtype)
    for th in thresholds:
        partial_embeds, pst = visual_partial_window(
            model.model.visual,
            target_pv,
            target_grid,
            donor_cache=donor_cache,
            threshold=th,
            profile=True,
        )
        if save_embeds_dir:
            prefix = f"{group_uid[:8]}_{target_rid[:8]}_{donor_rid[:8]}"
            torch.save({"embedding": partial_embeds.detach().cpu(), "stats": pst}, save_embeds_dir / f"{prefix}_partial_{th:.2f}.pt")
        patched = _replace_second_image(base_embeds, second_span, partial_embeds)
        logits_partial = forward_logits(
            model,
            processor,
            input_ids=full_ids,
            attention_mask=full_mask,
            mm_inputs=prompt_inputs,
            inputs_embeds=patched,
            device=device,
        )
        nll = teacher_force_metrics(logits_orig, logits_partial, prompt_len=prompt_len, response_ids=response_ids)
        em = _metrics_from_embeds(target_full, partial_embeds)
        out.append(
            SweepRow(
                group_uid=group_uid,
                dataset_row=dataset_row,
                target_request_id=target_rid,
                donor_request_id=donor_rid,
                threshold=th,
                total_windows=int(pst["total_windows"]),
                reused_windows=int(pst["reused_windows"]),
                computed_windows=int(pst["computed_windows"]),
                reuse_ratio=float(pst["reuse_ratio"]),
                prefull_window_layers=int(pst["prefull_window_layers"]),
                reused_window_layer_windows=int(pst["reused_window_layer_windows"]),
                computed_window_layer_windows=int(pst["computed_window_layer_windows"]),
                vision_ms_full_target=float(full_target_ms),
                vision_ms_partial=float(pst["partial_visual_ms"]),
                window_similarity_ms=float(pst["window_similarity_ms"]),
                index_scatter_gather_ms=float(pst["index_scatter_gather_ms"]),
                target_block_compute_ms=float(pst["target_block_compute_ms"]),
                full_attention_ms=float(pst["full_attention_ms"]),
                patch_merger_ms=float(pst["patch_merger_ms"]),
                window_cosine_mean=float(pst["window_cosine_mean"]),
                window_cosine_max=float(pst["window_cosine_max"]),
                image_token_mean_cos=float(em.get("image_token_mean_cos", -1.0)),
                image_token_min_cos=float(em.get("image_token_min_cos", -1.0)),
                image_token_max_abs_diff=float(em.get("image_token_max_abs_diff", -1.0)),
                image_token_top1_match=float(em.get("image_token_top1_match", -1.0)),
                delta_nll_per_token=float(nll["delta_nll_per_token"]),
                delta_nll=float(nll["delta_nll"]),
                top1_match_rate=float(nll["top1_match_rate"]),
                response_len=int(nll["response_len"]),
            )
        )
    return out


def discover_valid_groups(
    dump_dir: Path,
    parquet_path: str | Path,
    *,
    min_branches: int = 2,
) -> list:
    import pyarrow.parquet as pq

    n_rows = pq.read_table(str(parquet_path)).num_rows
    specs = discover_groups(dump_dir, min_branches=min_branches)
    return [s for s in specs if 0 <= int(s.dataset_row) < n_rows]


def aggregate(rows: list[SweepRow]) -> list[dict[str, Any]]:
    out = []
    for th in sorted({r.threshold for r in rows}):
        rs = [r for r in rows if r.threshold == th]
        def arr(name):
            return np.asarray([getattr(r, name) for r in rs], dtype=float)
        out.append({
            "threshold": th,
            "n_pairs": len(rs),
            "reuse_ratio_mean": float(arr("reuse_ratio").mean()),
            "computed_windows_mean": float(arr("computed_windows").mean()),
            "vision_ms_partial_mean": float(arr("vision_ms_partial").mean()),
            "vision_ms_full_target_mean": float(arr("vision_ms_full_target").mean()),
            "delta_nll_per_token_mean": float(arr("delta_nll_per_token").mean()),
            "delta_nll_per_token_p90": float(np.percentile(arr("delta_nll_per_token"), 90)),
            "delta_nll_per_token_max": float(arr("delta_nll_per_token").max()),
            "top1_match_mean": float(arr("top1_match_rate").mean()),
            "top1_match_min": float(arr("top1_match_rate").min()),
            "image_token_cos_mean": float(arr("image_token_mean_cos").mean()),
            "image_token_cos_min": float(arr("image_token_mean_cos").min()),
            "image_token_max_abs_diff_mean": float(arr("image_token_max_abs_diff").mean()),
            "image_token_max_abs_diff_p90": float(np.percentile(arr("image_token_max_abs_diff"), 90)),
            "image_token_max_abs_diff_max": float(arr("image_token_max_abs_diff").max()),
            "window_cosine_max_mean": float(arr("window_cosine_max").mean()),
            "window_similarity_ms_mean": float(arr("window_similarity_ms").mean()),
            "index_scatter_gather_ms_mean": float(arr("index_scatter_gather_ms").mean()),
            "target_block_compute_ms_mean": float(arr("target_block_compute_ms").mean()),
            "full_attention_ms_mean": float(arr("full_attention_ms").mean()),
            "patch_merger_ms_mean": float(arr("patch_merger_ms").mean()),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/data/refocus_chart_multiturn_oracle_changed/train.parquet")
    ap.add_argument("--image-dump-dir", default="profile_logs_vtool_chart_diversified/image_dump_vtool_chart_bs64_n4_diversified_grpo_cache")
    ap.add_argument("--out-dir", default="profile_logs_vtool_chart_diversified/partial_window_sweep")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--thresholds", default="0.85,0.88,0.90,0.92,0.95")
    ap.add_argument("--max-groups", type=int, default=0, help="0 = all valid groups")
    ap.add_argument("--max-pairs", type=int, default=50)
    ap.add_argument("--pairs-per-group", type=int, default=1)
    ap.add_argument("--min-branches", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--save-embeds", action="store_true")
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dir = out_dir / "embeds" if args.save_embeds else None

    model, processor, device = load_model_and_processor(args)
    dump_dir = Path(args.image_dump_dir)
    specs = discover_valid_groups(dump_dir, args.parquet, min_branches=args.min_branches)
    if args.max_groups > 0:
        specs = specs[: args.max_groups]
    if not specs:
        raise SystemExit(f"no usable groups in {dump_dir}")

    all_rows: list[SweepRow] = []
    skipped = []
    pair_count = 0
    for spec in specs:
        if pair_count >= args.max_pairs:
            break
        try:
            row = load_parquet_row(args.parquet, spec.dataset_row)
            pairs = build_pairs(spec.refocus_request_ids, pairs_per_group=args.pairs_per_group, seed=args.seed)
            print(f"[group] {spec.group_uid[:8]} row={spec.dataset_row} pairs={len(pairs)}")
            for target_rid, donor_rid in pairs:
                if pair_count >= args.max_pairs:
                    break
                rows = run_pair(
                    model, processor, device, row=row, dump_dir=dump_dir, group_uid=spec.group_uid,
                    dataset_row=spec.dataset_row, target_rid=target_rid, donor_rid=donor_rid,
                    thresholds=thresholds, max_new_tokens=args.max_new_tokens, save_embeds_dir=save_dir,
                )
                all_rows.extend(rows)
                pair_count += 1
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{spec.group_uid[:8]}: {type(exc).__name__}: {exc}")
            print(f"[skip] {skipped[-1]}")

    if not all_rows:
        raise SystemExit("no sweep rows produced")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"partial_window_sweep_{stamp}.csv"
    agg_path = out_dir / f"partial_window_sweep_{stamp}_agg.csv"
    json_path = out_dir / f"partial_window_sweep_{stamp}.json"
    fields = list(asdict(all_rows[0]).keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in all_rows:
            w.writerow(asdict(row))
    agg = aggregate(all_rows)
    with agg_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)
    json_path.write_text(
        json.dumps({"config": vars(args), "skipped": skipped, "aggregate": agg}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n=== aggregate ===")
    for r in agg:
        print(
            f"th={r['threshold']:.2f} n={r['n_pairs']} reuse={r['reuse_ratio_mean']:.3f} "
            f"tok_cos={r['image_token_cos_mean']:.4f} max_diff={r['image_token_max_abs_diff_mean']:.2f} "
            f"dnll/tok={r['delta_nll_per_token_mean']:.6f} top1={r['top1_match_mean']:.3f}"
        )
    print(f"Wrote {csv_path}")
    print(f"Wrote {agg_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
