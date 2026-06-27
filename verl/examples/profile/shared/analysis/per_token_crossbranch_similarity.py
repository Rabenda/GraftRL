#!/usr/bin/env python3
"""Per-token cross-branch similarity profile (first version).

Question (advisor's framing)
----------------------------
Within ONE GRPO group, the N branches each produce a turn1 refocus image. Because
different image tokens have different spatial anchors, we want to know, *per token
position*, how similar the branches' tokens are to each other:

    for each token index i:
        take the N branches' ViT token vectors at position i
        sim_i = mean over the N*(N-1)/2 pairwise cosines

Then report which token positions are most / least similar across the branches.

IMPORTANT
---------
* The number of image tokens is NOT fixed. "580" / "20x29" is only one example;
  every group can have a different token count / grid. We derive (n_tokens, hh, ww)
  from the Qwen2.5-VL vision encoder for each group and never hardcode it.
* A final ViT token is NOT a raw pixel patch; it only has a spatial *anchor*. The
  (row, col) we report is that anchor on the (hh x ww) token grid.
* This first version intentionally computes ONLY the per-token cross-branch cosine.
  No edited_count / disagreement / permutation baseline (added later if needed).

Outputs (per group)
-------------------
  per_token_similarity.csv     n_tokens rows: token_id,row,col,sim
  top_bottom_tokens.md         Top-K most similar + Bottom-K least similar
  similarity_heatmap.png       hh x ww similarity heatmap
  similarity_overlay.png       same heatmap upsampled & overlaid on the base image

Inputs
------
  # case-study dir(s) that already contain branch*_turn1_refocus.png (+ branch*_turn0.png):
  python3 examples/profile/per_token_crossbranch_similarity.py \
    --group-dir .../case_studies_crossbranch/group_a48d7a74 \
    --group-dir .../case_studies_crossbranch/group_36cf828b

  # or process every group_* under a parent dir:
  python3 examples/profile/per_token_crossbranch_similarity.py \
    --groups-root .../case_studies_crossbranch

  # or pull a group straight from an image dump via manifest:
  python3 examples/profile/per_token_crossbranch_similarity.py \
    --dump-dir .../image_dump_..._diversified --uid a48d7a74
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Optional

import numpy as np

# reuse the encoder + image loader from the unified analyzer
from analyze_similarity_unified import TokenCache, _load_rgb, load_structure


# --------------------------------------------------------------------------- #
# branch turn1 (and base turn0) image discovery
# --------------------------------------------------------------------------- #
def _branch_turn1_images_from_group_dir(group_dir: Path) -> tuple[list[tuple[str, Path]], Optional[Path]]:
    """Return ([(branch_label, turn1_path), ...], base_turn0_path or None)."""
    turn1: list[tuple[str, Path]] = []
    for p in sorted(group_dir.glob("branch*_turn1_refocus.png")):
        m = re.match(r"(branch\d+)_turn1_refocus\.png$", p.name)
        label = m.group(1) if m else p.stem
        turn1.append((label, p))
    base = None
    cands = sorted(group_dir.glob("branch*_turn0.png"))
    if cands:
        base = cands[0]
    return turn1, base


def _branch_turn1_images_from_dump(dump_dir: Path, uid_prefix: str) -> tuple[list[tuple[str, Path]], Optional[Path]]:
    groups = load_structure(dump_dir)
    uid = uid_prefix if uid_prefix in groups else next(
        (u for u in groups if u.startswith(uid_prefix)), None
    )
    if uid is None:
        raise SystemExit(f"uid not found in dump: {uid_prefix}")
    turn1: list[tuple[str, Path]] = []
    base = None
    for i, (branch, turns) in enumerate(sorted(groups[uid].items())):
        if 1 in turns:
            turn1.append((f"branch{i}", turns[1].path))
        if base is None and 0 in turns:
            base = turns[0].path
    return turn1, base


# --------------------------------------------------------------------------- #
# core: per-token cross-branch mean pairwise cosine
# --------------------------------------------------------------------------- #
def per_token_crossbranch_sim(
    turn1_paths: list[Path], cache: TokenCache
) -> tuple[np.ndarray, int, int]:
    """Return (sim[n_tokens], hh, ww).

    sim_i = mean over all N*(N-1)/2 branch pairs of cosine(branch_a token_i,
    branch_b token_i). Computed in O(N) per position via the identity

        sum_{a<b} cos(ea, eb) = ( || sum_b e_b/||e_b|| ||^2  -  N ) / 2

    on L2-normalized token vectors.
    """
    torch = cache.torch
    F = torch.nn.functional

    toks: list = []
    ref_grid = None
    ref_size = None
    for p in turn1_paths:
        t, g = cache.encode_image(_load_rgb(p))  # t:(n,d) cpu float, g:(T,H,W) patches
        if ref_grid is None:
            ref_grid = g
            ref_size = _load_rgb(p).size  # (W, H) native
        elif g != ref_grid:
            # align grids by resizing to the reference native size and re-encoding
            t, g = cache.encode_image(_load_rgb(p).resize(ref_size))
            if g != ref_grid:
                raise SystemExit(
                    f"grid mismatch within group ({p.name}: {g} vs ref {ref_grid}); "
                    "cannot align token positions"
                )
        toks.append(t)

    n_branches = len(toks)
    if n_branches < 2:
        raise SystemExit("need >=2 branches with turn1 to compare")

    stack = torch.stack(toks, dim=0)              # (B, n, d)
    normed = F.normalize(stack.float(), dim=-1)   # (B, n, d), each token unit-norm
    s = normed.sum(dim=0)                          # (n, d)
    g_pos = (s * s).sum(dim=-1)                    # (n,) = sum over all ordered pairs incl self
    n_pairs = n_branches * (n_branches - 1) / 2.0
    sim = ((g_pos - n_branches) / 2.0) / n_pairs  # (n,) mean pairwise cosine
    sim_np = sim.clamp(-1.0, 1.0).numpy()

    # spatial anchor grid: token order is row-major over (H/merge, W/merge)
    t_, h_, w_ = ref_grid
    ms = cache.merge_size
    hh, ww = h_ // ms, w_ // ms
    if hh * ww != sim_np.shape[0]:
        # cannot map to a clean grid; signal with -1
        hh, ww = -1, -1
    return sim_np, hh, ww


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
def write_csv(sim: np.ndarray, hh: int, ww: int, out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["token_id", "row", "col", "sim"])
        for i, s in enumerate(sim):
            row = i // ww if ww > 0 else -1
            col = i % ww if ww > 0 else -1
            w.writerow([i, row, col, f"{float(s):.6f}"])


def write_top_bottom(sim: np.ndarray, hh: int, ww: int, k: int, group: str, out_path: Path) -> None:
    order = np.argsort(sim)  # ascending: least similar first
    bottom = order[:k]
    top = order[::-1][:k]

    def rc(i):
        return (i // ww, i % ww) if ww > 0 else (-1, -1)

    L = [
        f"# Per-token cross-branch similarity — group {group}",
        "",
        f"- n_tokens = **{sim.shape[0]}**  (grid {hh}×{ww})  — NOTE: token count is per-group, not fixed.",
        f"- sim = mean of the branch pairwise cosines at each token position.",
        f"- overall: mean **{sim.mean():.4f}** · min **{sim.min():.4f}** · max **{sim.max():.4f}**",
        "",
        f"## Top-{k} MOST similar tokens (highest cross-branch cosine)",
        "",
        "| rank | token_id | row,col | sim |",
        "| ---: | ---: | :---: | ---: |",
    ]
    for r, i in enumerate(top, 1):
        rr, cc = rc(int(i))
        L.append(f"| {r} | {int(i)} | {rr},{cc} | {sim[i]:.4f} |")
    L += [
        "",
        f"## Bottom-{k} LEAST similar tokens (lowest cross-branch cosine)",
        "",
        "| rank | token_id | row,col | sim |",
        "| ---: | ---: | :---: | ---: |",
    ]
    for r, i in enumerate(bottom, 1):
        rr, cc = rc(int(i))
        L.append(f"| {r} | {int(i)} | {rr},{cc} | {sim[i]:.4f} |")
    out_path.write_text("\n".join(L), encoding="utf-8")


def save_heatmaps(
    sim: np.ndarray, hh: int, ww: int, base_path: Optional[Path], out_dir: Path, group: str
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        print("  [warn] matplotlib unavailable; skipping heatmaps")
        return
    if ww <= 0:
        print("  [warn] token grid not rectangular; skipping heatmaps")
        return

    grid = sim.reshape(hh, ww)
    vmin, vmax = float(grid.min()), 1.0

    # standalone heatmap
    fig, ax = plt.subplots(figsize=(ww * 0.32 + 2, hh * 0.32 + 1))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(f"{group} — per-token cross-branch cosine ({hh}×{ww}={sim.shape[0]})", fontsize=9)
    ax.set_xlabel("col (token anchor)")
    ax.set_ylabel("row (token anchor)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "similarity_heatmap.png", dpi=110)
    plt.close(fig)

    # overlay on base image
    if base_path is not None and base_path.exists():
        base = _load_rgb(base_path)
        W, H = base.size
        fig, ax = plt.subplots(figsize=(8, 8 * H / max(W, 1)))
        ax.imshow(base)
        im = ax.imshow(
            grid, cmap="RdYlGn", vmin=vmin, vmax=vmax, alpha=0.5,
            extent=[0, W, H, 0], interpolation="nearest", aspect="auto",
        )
        ax.set_title(f"{group} — low(red)=branches diverge / high(green)=reusable", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / "similarity_overlay.png", dpi=110)
        plt.close(fig)


# --------------------------------------------------------------------------- #
def process_group(turn1: list[tuple[str, Path]], base: Optional[Path], group: str,
                  cache: TokenCache, out_dir: Path, k: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [p for _, p in turn1]
    print(f"=== group {group}: {len(paths)} branch turn1 images ===")
    for lbl, p in turn1:
        print(f"    {lbl}: {p.name}")
    sim, hh, ww = per_token_crossbranch_sim(paths, cache)
    print(f"    n_tokens={sim.shape[0]} grid={hh}x{ww} "
          f"sim mean={sim.mean():.4f} min={sim.min():.4f} max={sim.max():.4f}")
    write_csv(sim, hh, ww, out_dir / "per_token_similarity.csv")
    write_top_bottom(sim, hh, ww, k, group, out_dir / "top_bottom_tokens.md")
    save_heatmaps(sim, hh, ww, base, out_dir, group)
    print(f"    -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group-dir", action="append", default=[],
                    help="case-study dir with branch*_turn1_refocus.png (repeatable)")
    ap.add_argument("--groups-root", default=None,
                    help="parent dir; process every group_* subdir")
    ap.add_argument("--dump-dir", default=None, help="image dump dir (manifest.jsonl) — use with --uid")
    ap.add_argument("--uid", default=None, help="uid (full or prefix) when using --dump-dir")
    ap.add_argument("--out-dir", default=None,
                    help="output dir; default = <group dir>/per_token_sim")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--top", type=int, default=20, help="Top-K / Bottom-K to list")
    args = ap.parse_args()

    # collect groups: list of (group_name, turn1_list, base_path, out_dir)
    jobs: list[tuple[str, list[tuple[str, Path]], Optional[Path], Path]] = []

    group_dirs: list[Path] = [Path(d) for d in args.group_dir]
    if args.groups_root:
        root = Path(args.groups_root)
        group_dirs += sorted(p for p in root.glob("group_*") if p.is_dir())
    for gd in group_dirs:
        turn1, base = _branch_turn1_images_from_group_dir(gd)
        if not turn1:
            print(f"[skip] no branch*_turn1_refocus.png in {gd}")
            continue
        out = Path(args.out_dir) / gd.name if args.out_dir else gd / "per_token_sim"
        jobs.append((gd.name, turn1, base, out))

    if args.dump_dir and args.uid:
        turn1, base = _branch_turn1_images_from_dump(Path(args.dump_dir), args.uid)
        name = f"group_{args.uid[:8]}"
        out = Path(args.out_dir) / name if args.out_dir else Path(args.dump_dir).parent / "per_token_sim" / name
        jobs.append((name, turn1, base, out))

    if not jobs:
        raise SystemExit("nothing to do: pass --group-dir / --groups-root / (--dump-dir + --uid)")

    device = args.device or ("cuda" if _cuda_ok() else "cpu")
    print(f"device = {device}")
    cache = TokenCache(args.model, device, args.max_pixels, cap=64)
    for name, turn1, base, out in jobs:
        process_group(turn1, base, name, cache, out, args.top)


def _cuda_ok() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    main()
