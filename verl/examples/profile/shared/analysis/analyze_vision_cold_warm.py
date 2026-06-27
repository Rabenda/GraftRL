#!/usr/bin/env python3
"""
Classify vision_encoder_log rows into cold vs warm/batched paths and join with generate logs.

Cold path (true ViT encode for one request):
  pass_id == 1 and '|' not in request_ids

Warm/batched path:
  pass_id >= 2 or multiple request_ids in one row

Usage:
  python examples/profile/analyze_vision_cold_warm.py \
    --generate-log profile_logs_geo3k/verl_sglang_generate_log_qwen25vl_geo3k_2gpu_ab_req8.csv \
    --vision-log profile_logs_geo3k/vision_encoder_log_qwen25vl_geo3k_2gpu_ab_req8.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics as st


def load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def classify_vision_rows(vision_rows: list[dict]) -> dict:
    cold, batched, cache_hit = [], [], []
    for r in vision_rows:
        ids = r.get("request_ids") or ""
        if r.get("cached_image_features") == "1":
            cache_hit.append(r)
        if "|" in ids:
            batched.append(r)
        elif r.get("pass_id") == "1":
            cold.append(r)
        else:
            batched.append(r)
    return {"cold": cold, "batched": batched, "cache_hit": cache_hit}


def cold_vision_by_request(cold_rows: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in cold_rows:
        rid = (r.get("request_ids") or "").strip()
        if rid and "|" not in rid:
            out[rid] = float(r.get("vision_encoder_time_ms") or 0)
    return out


def batched_vision_allocated(batched_rows: list[dict]) -> dict[str, float]:
    """Sum vision time split evenly across request_ids in each batched row."""
    out: dict[str, float] = {}
    for r in batched_rows:
        ids = [x for x in (r.get("request_ids") or "").split("|") if x]
        if not ids:
            continue
        share = float(r.get("vision_encoder_time_ms") or 0) / len(ids)
        for rid in ids:
            out[rid] = out.get(rid, 0.0) + share
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-log", required=True)
    parser.add_argument("--vision-log", required=True)
    parser.add_argument("--label", default="run")
    args = parser.parse_args()

    gen = load_csv(args.generate_log)
    vis = load_csv(args.vision_log)
    groups = classify_vision_rows(vis)
    cold_map = cold_vision_by_request(groups["cold"])
    batched_map = batched_vision_allocated(groups["batched"])

    print(f"\n=== {args.label} ===")
    print(f"requests: {len(gen)}  vision_rows: {len(vis)}")
    print(
        f"cold_rows(pass1,single): {len(groups['cold'])}  "
        f"batched_rows: {len(groups['batched'])}  "
        f"cached_embed_rows: {len(groups['cache_hit'])}"
    )

    rows = []
    for g in gen:
        rid = g["request_id"]
        e2e = float(g.get("generate_e2e_ms") or 0)
        cold_ms = cold_map.get(rid)
        batched_ms = batched_map.get(rid, 0.0)
        path = "cold" if cold_ms is not None else "batched_only"
        cold_pct = (cold_ms / e2e * 100) if cold_ms and e2e else None
        alloc_pct = (batched_ms / e2e * 100) if e2e else 0.0
        rows.append(
            {
                "rid": rid[:8],
                "path": path,
                "prompt": int(float(g.get("prompt_tokens") or 0)),
                "img_tok": int(float(g.get("image_prompt_tokens") or 0)),
                "cold_ms": cold_ms,
                "batched_ms": batched_ms,
                "e2e": e2e,
                "cold_pct": cold_pct,
                "out_tok": int(float(g.get("output_tokens") or 0)),
                "finish": g.get("finish_reason"),
            }
        )

    cold_pcts = [r["cold_pct"] for r in rows if r["cold_pct"] is not None]
    batched_only = [r for r in rows if r["path"] == "batched_only"]
    if cold_pcts:
        print(
            f"cold vision/e2e: min={min(cold_pcts):.1f}% med={st.median(cold_pcts):.1f}% "
            f"max={max(cold_pcts):.1f}% mean={st.mean(cold_pcts):.1f}%"
        )
    print(f"batched_only requests: {len(batched_only)} / {len(rows)}")

    print("\n| req | path | img_tok | cold_vis(ms) | batched_alloc(ms) | e2e(ms) | cold% | out_tok |")
    for r in rows:
        cv = "-" if r["cold_ms"] is None else f"{r['cold_ms']:.0f}"
        cp = "-" if r["cold_pct"] is None else f"{r['cold_pct']:.1f}%"
        print(
            f"| {r['rid']} | {r['path']} | {r['img_tok']} | {cv} | {r['batched_ms']:.1f} | "
            f"{r['e2e']:.0f} | {cp} | {r['out_tok']} |"
        )


if __name__ == "__main__":
    main()
