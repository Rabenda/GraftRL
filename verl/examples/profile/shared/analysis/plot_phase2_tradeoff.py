#!/usr/bin/env python3
"""Plot Phase 2 reuse-ratio trade-off curves from batch aggregate CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODE_STYLE = {
    "high_sim": {"color": "#4C78A8", "label": "high-sim replacement"},
    "random": {"color": "#F58518", "label": "random replacement"},
    "low_sim": {"color": "#E45756", "label": "low-sim replacement"},
}
MODE_ORDER = ["high_sim", "random", "low_sim"]


def plot_tradeoff(agg: pd.DataFrame, out_dir: Path, *, prefix: str = "phase2") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- ΔNLL/token ---
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for mode in MODE_ORDER:
        sub = agg[agg["mode"] == mode].sort_values("reuse_ratio")
        if sub.empty:
            continue
        style = MODE_STYLE[mode]
        ax.plot(
            sub["reuse_ratio"],
            sub["dnll_mean"],
            marker="o",
            linewidth=2.2,
            color=style["color"],
            label=style["label"],
        )
        ax.fill_between(
            sub["reuse_ratio"],
            sub["dnll_mean"],
            sub["dnll_p90"],
            alpha=0.12,
            color=style["color"],
        )
    ax.set_xlabel("Reuse ratio")
    ax.set_ylabel("ΔNLL per token (teacher-forcing)")
    ax.set_title("Phase 2 — Replacement impact vs reuse ratio")
    ax.set_xlim(0.05, 0.85)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)
    note = (
        "Solid line = mean ΔNLL/token across pairs; shaded band = mean→p90.\n"
        "Cross-branch turn1 refocus replacement (projector embeddings)."
    )
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=8.5, color="#555555")
    fig.tight_layout()
    p1 = out_dir / f"{prefix}_reuse_ratio_vs_dnll.png"
    fig.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # --- top-1 match ---
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for mode in MODE_ORDER:
        sub = agg[agg["mode"] == mode].sort_values("reuse_ratio")
        if sub.empty:
            continue
        style = MODE_STYLE[mode]
        ax.plot(
            sub["reuse_ratio"],
            sub["top1_mean"],
            marker="o",
            linewidth=2.2,
            color=style["color"],
            label=style["label"],
        )
        ax.fill_between(
            sub["reuse_ratio"],
            sub["top1_p10"],
            sub["top1_mean"],
            alpha=0.12,
            color=style["color"],
        )
    ax.set_xlabel("Reuse ratio")
    ax.set_ylabel("Top-1 next-token match rate")
    ax.set_title("Phase 2 — Prediction stability vs reuse ratio")
    ax.set_xlim(0.05, 0.85)
    ax.set_ylim(0.85, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.25)
    note = "Solid line = mean top-1 match; shaded band = p10→mean (worst-case tail)."
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=8.5, color="#555555")
    fig.tight_layout()
    p2 = out_dir / f"{prefix}_reuse_ratio_vs_top1.png"
    fig.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {p1}")
    print(f"Wrote {p2}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agg-csv",
        type=Path,
        help="phase2_batch_*_agg.csv (default: newest in phase2 dir)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("profile_logs_refocus_chart/similarity_diversified/phase2"),
    )
    parser.add_argument("--prefix", default="phase2")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir

    if args.agg_csv:
        agg_path = args.agg_csv if args.agg_csv.is_absolute() else repo / args.agg_csv
    else:
        candidates = sorted(out_dir.glob("phase2_batch_*_agg.csv"))
        if not candidates:
            raise SystemExit(f"no *_agg.csv found under {out_dir}")
        agg_path = candidates[-1]

    agg = pd.read_csv(agg_path)
    print(f"Using {agg_path} ({len(agg)} rows)")
    plot_tradeoff(agg, out_dir, prefix=args.prefix)


if __name__ == "__main__":
    main()
