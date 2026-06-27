#!/usr/bin/env python3
"""Phase 1 charts: token redundancy from per-token cross-branch similarity CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEY_THRESHOLDS = [0.90, 0.95, 0.98, 0.99, 0.999]


def find_per_token_csvs(root: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for csv_path in sorted(root.rglob("per_token_similarity.csv")):
        group_id = csv_path.parent.parent.name
        if group_id.startswith("group_"):
            group_id = group_id[len("group_") :]
        rows.append((group_id, csv_path))
    return rows


def load_sim_series(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if "sim" not in df.columns:
        raise ValueError(f"missing 'sim' column: {csv_path}")
    return df["sim"].to_numpy(dtype=float)


def pct_above(sim: np.ndarray, threshold: float) -> float:
    return float((sim >= threshold).mean() * 100.0)


def threshold_curve(sim: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.array([pct_above(sim, t) for t in thresholds], dtype=float)


def summarize_group(group_id: str, sim: np.ndarray) -> dict:
    row = {
        "group_id": group_id,
        "n_tokens": len(sim),
        "mean_cos": float(sim.mean()),
        "median_cos": float(np.median(sim)),
        "min_cos": float(sim.min()),
        "max_cos": float(sim.max()),
    }
    for th in KEY_THRESHOLDS:
        row[f"pct_ge_{th:.3f}".replace(".", "_")] = pct_above(sim, th)
    return row


def plot_distribution(groups: list[tuple[str, np.ndarray]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    all_sim = np.concatenate([sim for _, sim in groups])
    bins = np.linspace(0.3, 1.0, 71)

    ax = axes[0]
    ax.hist(all_sim, bins=bins, density=True, alpha=0.75, color="#4C78A8", edgecolor="white", linewidth=0.4)
    ax.axvline(all_sim.mean(), color="#E45756", linestyle="--", linewidth=1.5, label=f"mean={all_sim.mean():.3f}")
    ax.axvline(np.median(all_sim), color="#F58518", linestyle=":", linewidth=1.5, label=f"median={np.median(all_sim):.3f}")
    ax.set_xlabel("Per-token cosine similarity (cross-branch mean)")
    ax.set_ylabel("Density")
    ax.set_title("A. Similarity distribution (all groups pooled)")
    ax.set_xlim(0.3, 1.0)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 1)))
    for (group_id, sim), color in zip(groups, colors):
        ax.hist(
            sim,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            color=color,
            label=f"{group_id[:8]} (n={len(sim)})",
        )
    ax.set_xlabel("Per-token cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title("B. Per-group distributions")
    ax.set_xlim(0.3, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Phase 1 — Cross-branch turn1 refocus token similarity\n"
        "(sim = mean pairwise cosine at each token index)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_curve(groups: list[tuple[str, np.ndarray]], out_path: Path) -> None:
    thresholds = np.linspace(0.50, 0.999, 200)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    curves = []
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 1)))
    for (group_id, sim), color in zip(groups, colors):
        curve = threshold_curve(sim, thresholds)
        curves.append(curve)
        ax.plot(thresholds, curve, linewidth=2.0, color=color, label=f"group {group_id[:8]}")

    mean_curve = np.mean(np.stack(curves, axis=0), axis=0)
    ax.plot(thresholds, mean_curve, linewidth=2.8, color="black", linestyle="--", label="mean across groups")

    for th in KEY_THRESHOLDS:
        ax.axvline(th, color="#BBBBBB", linestyle=":", linewidth=0.9, alpha=0.8)
        y = float(np.mean([pct_above(sim, th) for _, sim in groups]))
        ax.scatter([th], [y], s=36, color="#E45756", zorder=5)
        ax.annotate(
            f"{y:.1f}% @ {th:.2f}",
            xy=(th, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="#333333",
        )

    ax.set_xlabel("Cosine similarity threshold")
    ax.set_ylabel("% tokens with sim ≥ threshold")
    ax.set_title("Phase 1 — Reusable token fraction vs similarity threshold")
    ax.set_xlim(0.50, 1.0)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25)

    note = (
        "Metric: at each token index, sim = mean of 6 cross-branch turn1-refocus pairwise cosines.\n"
        "Conservative workload: each branch edits a different bbox on the same base image."
    )
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=8.5, color="#555555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(groups: list[tuple[str, np.ndarray]], out_path: Path) -> None:
    rows = [summarize_group(gid, sim) for gid, sim in groups]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    md_path = out_path.with_suffix(".md")
    lines = [
        "# Phase 1 token redundancy summary",
        "",
        "Per-token `sim` = mean cross-branch pairwise cosine at each token index (turn1 refocus).",
        "",
        "| group | n_tokens | mean | median | P(≥0.90) | P(≥0.95) | P(≥0.98) | P(≥0.99) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['group_id'][:8]} | {int(row['n_tokens'])} | "
            f"{row['mean_cos']:.4f} | {row['median_cos']:.4f} | "
            f"{row['pct_ge_0_900']:.1f}% | {row['pct_ge_0_950']:.1f}% | "
            f"{row['pct_ge_0_980']:.1f}% | {row['pct_ge_0_990']:.1f}% |"
        )
    pooled = np.concatenate([sim for _, sim in groups])
    lines.extend(
        [
            "",
            f"**Pooled** ({len(pooled)} tokens): mean={pooled.mean():.4f}, median={np.median(pooled):.4f}, "
            f"P(≥0.90)={pct_above(pooled, 0.90):.1f}%, P(≥0.95)={pct_above(pooled, 0.95):.1f}%, "
            f"P(≥0.98)={pct_above(pooled, 0.98):.1f}%, P(≥0.99)={pct_above(pooled, 0.99):.1f}%.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("verl_vision/profile_logs_refocus_chart/similarity_diversified/case_studies_crossbranch"),
        help="Directory containing group_*/per_token_sim/per_token_similarity.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("verl_vision/profile_logs_refocus_chart/similarity_diversified/phase1"),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    root = args.root if args.root.is_absolute() else repo_root / args.root
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = find_per_token_csvs(root)
    if not entries:
        raise SystemExit(f"No per_token_similarity.csv found under {root}")

    groups: list[tuple[str, np.ndarray]] = []
    for group_id, csv_path in entries:
        groups.append((group_id, load_sim_series(csv_path)))
        print(f"loaded {group_id}: {csv_path} ({len(groups[-1][1])} tokens)")

    plot_distribution(groups, out_dir / "phase1_similarity_distribution.png")
    plot_threshold_curve(groups, out_dir / "phase1_reuse_threshold_curve.png")
    write_summary(groups, out_dir / "phase1_summary.csv")

    print(f"Wrote charts to {out_dir}")


if __name__ == "__main__":
    main()
