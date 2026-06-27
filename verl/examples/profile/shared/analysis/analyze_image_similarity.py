#!/usr/bin/env python3
"""Compute image similarity within GRPO groups (same uid) and across turns."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image


def _load_rgb(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return arr


def _resize(arr: np.ndarray, size: int = 224) -> np.ndarray:
    im = Image.fromarray((arr * 255).astype(np.uint8))
    im = im.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def pixel_cosine(a: np.ndarray, b: np.ndarray, size: int = 224) -> float:
    va = _resize(a, size).reshape(-1)
    vb = _resize(b, size).reshape(-1)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom <= 1e-8:
        return 1.0 if np.allclose(va, vb) else 0.0
    return float(np.dot(va, vb) / denom)


def mse_similarity(a: np.ndarray, b: np.ndarray, size: int = 224) -> float:
    va = _resize(a, size)
    vb = _resize(b, size)
    mse = float(np.mean((va - vb) ** 2))
    return max(0.0, 1.0 - mse)


def histogram_correlation(a: np.ndarray, b: np.ndarray) -> float:
    ha = np.histogram(a.reshape(-1), bins=64, range=(0.0, 1.0))[0].astype(np.float64)
    hb = np.histogram(b.reshape(-1), bins=64, range=(0.0, 1.0))[0].astype(np.float64)
    if ha.sum() == 0 or hb.sum() == 0:
        return 0.0
    ha /= ha.sum()
    hb /= hb.sum()
    return float(np.corrcoef(ha, hb)[0, 1])


def combined_similarity(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    return {
        "pixel_cosine": pixel_cosine(a, b),
        "mse_sim": mse_similarity(a, b),
        "hist_corr": histogram_correlation(a, b),
        "combined": (
            0.5 * pixel_cosine(a, b)
            + 0.3 * mse_similarity(a, b)
            + 0.2 * max(0.0, histogram_correlation(a, b))
        ),
    }


def load_manifest(manifest_path: Path) -> list[dict]:
    rows = []
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_groups(manifest_rows: list[dict], dump_dir: Path) -> tuple[list[dict], list[dict]]:
    by_uid: dict[str, list[dict]] = defaultdict(list)
    for row in manifest_rows:
        by_uid[row["uid"]].append(row)

    pair_rows: list[dict] = []
    group_rows: list[dict] = []

    for uid, entries in sorted(by_uid.items()):
        by_turn: dict[int, list[dict]] = defaultdict(list)
        for e in entries:
            by_turn[int(e["turn"])].append(e)

        for turn, turn_entries in sorted(by_turn.items()):
            paths = []
            for e in turn_entries:
                p = dump_dir / e["path"]
                if p.exists():
                    paths.append((e.get("rollout_idx", e.get("request_id", "?")), p))
            if len(paths) < 2:
                continue

            sims = []
            for (id_a, pa), (id_b, pb) in combinations(paths, 2):
                metrics = combined_similarity(_load_rgb(pa), _load_rgb(pb))
                pair_rows.append(
                    {
                        "uid": uid,
                        "turn": turn,
                        "image_role": turn_entries[0].get("role", ""),
                        "id_a": id_a,
                        "id_b": id_b,
                        **metrics,
                    }
                )
                sims.append(metrics["combined"])

            if sims:
                group_rows.append(
                    {
                        "uid": uid,
                        "turn": turn,
                        "image_role": turn_entries[0].get("role", ""),
                        "n_images": len(paths),
                        "mean_combined_sim": float(np.mean(sims)),
                        "min_combined_sim": float(np.min(sims)),
                        "max_combined_sim": float(np.max(sims)),
                    }
                )

        turns = sorted(by_turn)
        for t_prev, t_next in zip(turns, turns[1:]):
            prev_entries = by_turn[t_prev]
            next_entries = by_turn[t_next]
            if not prev_entries or not next_entries:
                continue
            pa = dump_dir / prev_entries[0]["path"]
            pb = dump_dir / next_entries[0]["path"]
            if not pa.exists() or not pb.exists():
                continue
            metrics = combined_similarity(_load_rgb(pa), _load_rgb(pb))
            pair_rows.append(
                {
                    "uid": uid,
                    "turn": f"{t_prev}->{t_next}",
                    "image_role": f"{prev_entries[0].get('role','')}->{next_entries[0].get('role','')}",
                    "id_a": prev_entries[0].get("rollout_idx", "0"),
                    "id_b": next_entries[0].get("rollout_idx", "0"),
                    **metrics,
                }
            )

    return pair_rows, group_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def print_summary(group_rows: list[dict], pair_rows: list[dict]) -> None:
    print("\n=== GRPO group similarity (same uid, same turn) ===")
    if not group_rows:
        print("  (no multi-image groups found in manifest)")
    else:
        for row in group_rows:
            if isinstance(row["turn"], int):
                print(
                    f"  uid={row['uid'][:8]}.. turn={row['turn']} role={row['image_role']} "
                    f"n={row['n_images']} combined={row['mean_combined_sim']:.3f} "
                    f"[{row['min_combined_sim']:.3f}, {row['max_combined_sim']:.3f}]"
                )

    cross = [r for r in pair_rows if isinstance(r.get("turn"), str) and "->" in str(r["turn"])]
    if cross:
        print("\n=== Cross-turn similarity (first rollout per uid) ===")
        vals = [r["combined"] for r in cross]
        print(f"  pairs={len(cross)} mean_combined={float(np.mean(vals)):.3f}")
        for row in cross[:8]:
            print(
                f"  uid={row['uid'][:8]}.. {row['turn']} {row['image_role']} "
                f"combined={row['combined']:.3f} cosine={row['pixel_cosine']:.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze image similarity from rollout image dump.")
    parser.add_argument(
        "--dump-dir",
        default=os.environ.get("PROFILE_IMAGE_DUMP_DIR", ""),
        help="Directory with manifest.jsonl and saved PNGs",
    )
    parser.add_argument("--out-dir", default=None, help="CSV output dir (default: dump-dir)")
    args = parser.parse_args()
    if not args.dump_dir:
        raise SystemExit("Provide --dump-dir or set PROFILE_IMAGE_DUMP_DIR")

    dump_dir = Path(args.dump_dir)
    manifest_path = dump_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"Missing {manifest_path}")

    out_dir = Path(args.out_dir or dump_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest(manifest_path)
    pair_rows, group_rows = summarize_groups(manifest_rows, dump_dir)
    write_csv(out_dir / "similarity_pairs.csv", pair_rows)
    write_csv(out_dir / "similarity_groups.csv", group_rows)
    print_summary(group_rows, pair_rows)
    print(f"\nWrote:\n  {out_dir / 'similarity_pairs.csv'}\n  {out_dir / 'similarity_groups.csv'}")


if __name__ == "__main__":
    main()
