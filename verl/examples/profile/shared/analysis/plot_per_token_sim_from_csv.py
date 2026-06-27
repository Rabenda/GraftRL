#!/usr/bin/env python3
"""Plot per-token cross-branch similarity heatmaps from existing CSV (no GPU).

Reads per_token_similarity.csv (columns: token_id, row, col, sim) and writes:
  similarity_heatmap.png   — 20×29 (or hh×ww) grid, green=high sim, red=low
  similarity_overlay.png   — same grid overlaid on branch0_turn0.png if present

Usage
-----
  # one group:
  python3 examples/profile/plot_per_token_sim_from_csv.py \
    --sim-dir profile_logs_refocus_chart/.../group_36cf828b/per_token_sim

  # all per_token_sim under a tree:
  python3 examples/profile/plot_per_token_sim_from_csv.py \
    --root profile_logs_refocus_chart --root profile_logs_refocus_chart_origin
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def load_sim_csv(csv_path: Path) -> tuple[np.ndarray, int, int]:
    rows: list[tuple[int, int, int, float]] = []
    with csv_path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for line in r:
            rows.append(
                (int(line["token_id"]), int(line["row"]), int(line["col"]), float(line["sim"]))
            )
    if not rows:
        raise ValueError(f"empty csv: {csv_path}")
    hh = max(r for _, r, _, _ in rows) + 1
    ww = max(c for _, _, c, _ in rows) + 1
    grid = np.full((hh, ww), np.nan, dtype=np.float64)
    for tid, row, col, sim in rows:
        if tid != row * ww + col:
            # tolerate out-of-order rows; anchor by row,col
            pass
        grid[row, col] = sim
    if np.isnan(grid).any():
        missing = int(np.isnan(grid).sum())
        raise ValueError(f"{csv_path}: {missing} cells missing in {hh}x{ww} grid")
    return grid, hh, ww


def find_base_image(sim_dir: Path) -> Path | None:
    parent = sim_dir.parent
    cands = sorted(parent.glob("branch*_turn0.png"))
    return cands[0] if cands else None


def plot_grid(
    grid: np.ndarray,
    hh: int,
    ww: int,
    title: str,
    out_dir: Path,
    base_path: Path | None,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmin, vmax = float(np.nanmin(grid)), 1.0

    fig, ax = plt.subplots(figsize=(ww * 0.35 + 2.5, hh * 0.35 + 1.5))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(f"{title}\ncross-branch mean cosine per token ({hh}×{ww})", fontsize=10)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, label="sim", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "similarity_heatmap.png", dpi=dpi)
    plt.close(fig)

    if base_path is not None and base_path.is_file():
        base = Image.open(base_path).convert("RGB")
        W, H = base.size
        fig, ax = plt.subplots(figsize=(9, 9 * H / max(W, 1)))
        ax.imshow(base)
        im = ax.imshow(
            grid,
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax,
            alpha=0.55,
            extent=[0, W, H, 0],
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_title(f"{title}\ngreen=reusable across 4 branches · red=divergent", fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, label="sim", fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / "similarity_overlay.png", dpi=dpi)
        plt.close(fig)


def process_sim_dir(sim_dir: Path, dpi: int) -> None:
    csv_path = sim_dir / "per_token_similarity.csv"
    if not csv_path.is_file():
        print(f"[skip] no csv: {sim_dir}")
        return
    grid, hh, ww = load_sim_csv(csv_path)
    title = sim_dir.parent.name
    base = find_base_image(sim_dir)
    plot_grid(grid, hh, ww, title, sim_dir, base, dpi)
    print(f"  -> {sim_dir}/similarity_heatmap.png")
    if base:
        print(f"  -> {sim_dir}/similarity_overlay.png  (base: {base.name})")


def discover_sim_dirs(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        if (root / "per_token_similarity.csv").is_file():
            found.append(root)
        for d in sorted(root.rglob("per_token_sim")):
            if d.is_dir() and (d / "per_token_similarity.csv").is_file():
                found.append(d)
    # dedupe
    seen: set[str] = set()
    out: list[Path] = []
    for d in found:
        key = str(d.resolve())
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-dir", action="append", default=[], help=".../per_token_sim directory")
    ap.add_argument("--root", action="append", default=[], help="search **/per_token_sim under root")
    ap.add_argument("--dpi", type=int, default=120)
    args = ap.parse_args()

    dirs = [Path(d) for d in args.sim_dir]
    dirs += discover_sim_dirs([Path(r) for r in args.root])
    if not dirs:
        raise SystemExit("pass --sim-dir or --root")

    try:
        import matplotlib  # noqa: F401
    except ImportError as e:
        raise SystemExit("matplotlib required: pip install matplotlib") from e

    for d in dirs:
        print(f"plot {d}")
        process_sim_dir(d, args.dpi)


if __name__ == "__main__":
    main()
