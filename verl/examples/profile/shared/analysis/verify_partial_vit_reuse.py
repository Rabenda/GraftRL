#!/usr/bin/env python3
"""Compare baseline, whole-slot cache, and partial-window reuse profile logs.

Default mode reads real rollout CSV logs and reports:
  - vision_encoder_time_ms
  - computed_windows / total_windows
  - reused_windows / total_windows
  - fallback_reason counts

Optional embedding mode compares saved image-token tensors for numerical drift:
  --baseline-emb target_full.pt --whole-emb donor_full.pt --partial-emb partial.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        return [
            r
            for r in csv.DictReader(f)
            if r.get("timestamp") != "timestamp" and r.get("request_id") != "request_id"
        ]


def _f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key) or 0)
    except Exception:
        return 0.0


def _i(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key) or 0))
    except Exception:
        return 0


def _summarize_vision(path: Path) -> dict[str, Any]:
    rows = _rows(path)
    turn1 = [r for r in rows if r.get("grpo_sim_agent_turn") == "1" or "_t1" in (r.get("request_ids") or "")]
    partial_rows = [r for r in turn1 if _i(r, "total_windows") > 0 or r.get("fallback_reason")]
    fallback = Counter((r.get("fallback_reason") or "") for r in partial_rows)
    return {
        "path": str(path),
        "rows": len(rows),
        "turn1_rows": len(turn1),
        "vision_encoder_time_ms": round(sum(_f(r, "vision_encoder_time_ms") for r in turn1), 3),
        "vit_calls": sum(_i(r, "grpo_sim_vit_calls") for r in rows),
        "vit_skipped": sum(_i(r, "grpo_sim_vit_skipped") for r in rows),
        "similarity_reuse": sum(_i(r, "grpo_sim_similarity_reuse") for r in rows),
        "partial_rows": len(partial_rows),
        "partial_vit_used": sum(_i(r, "partial_vit_used") for r in rows),
        "total_windows": sum(_i(r, "total_windows") for r in rows),
        "reused_windows": sum(_i(r, "reused_windows") for r in rows),
        "computed_windows": sum(_i(r, "computed_windows") for r in rows),
        "partial_vit_total_window_layers": sum(_i(r, "partial_vit_total_window_layers") for r in rows),
        "partial_vit_reused_window_layer_windows": sum(
            _i(r, "partial_vit_reused_window_layer_windows") for r in rows
        ),
        "partial_vit_computed_window_layer_windows": sum(
            _i(r, "partial_vit_computed_window_layer_windows") for r in rows
        ),
        "fallback_reason_counts": dict(fallback),
    }


def _load_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("emb", "embedding", "image_embeds", "image_tokens"):
            if key in obj:
                obj = obj[key]
                break
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"{path} did not contain a tensor")
    return obj


def _compare_embeds(base: Path, other: Path) -> dict[str, Any]:
    a = _load_tensor(base).float()
    b = _load_tensor(other).float()
    if a.shape != b.shape:
        return {"shape_match": False, "base_shape": list(a.shape), "other_shape": list(b.shape)}
    cos = F.cosine_similarity(a, b, dim=-1, eps=1e-6)
    top1 = torch.argmax(a, dim=-1).eq(torch.argmax(b, dim=-1)).float()
    return {
        "shape_match": True,
        "mean_cos": float(cos.mean().item()),
        "min_cos": float(cos.min().item()),
        "max_abs_diff": float((a - b).abs().max().item()),
        "top1_same": bool(top1.min().item() == 1.0),
        "top1_match_rate": float(top1.mean().item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-vision-log", required=True, type=Path)
    ap.add_argument("--whole-slot-vision-log", required=True, type=Path)
    ap.add_argument("--partial-vision-log", required=True, type=Path)
    ap.add_argument("--baseline-emb", type=Path)
    ap.add_argument("--whole-emb", type=Path)
    ap.add_argument("--partial-emb", type=Path)
    args = ap.parse_args()

    out: dict[str, Any] = {
        "baseline": _summarize_vision(args.baseline_vision_log),
        "whole_slot": _summarize_vision(args.whole_slot_vision_log),
        "partial_window": _summarize_vision(args.partial_vision_log),
    }
    if args.baseline_emb and args.whole_emb:
        out["whole_slot_vs_baseline_emb"] = _compare_embeds(args.baseline_emb, args.whole_emb)
    if args.baseline_emb and args.partial_emb:
        out["partial_vs_baseline_emb"] = _compare_embeds(args.baseline_emb, args.partial_emb)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
