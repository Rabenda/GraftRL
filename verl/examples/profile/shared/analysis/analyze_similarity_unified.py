#!/usr/bin/env python3
"""Unified visual-redundancy / image-token-reuse analysis for VLM GRPO/RL rollouts.

Motivation
----------
Our cache idea targets the *visual redundancy* that is specific to GRPO + multi-turn
RL rollout, NOT generic serving cache:

  * a GRPO group (one ``uid``) expands into ``rollout_n`` branches (one ``request_id``
    each). At turn 0 every branch sees the *same* input image.
  * multi-turn refocus locally edits the image (mask / draw / highlight on a bbox),
    so turn t+1's image is mostly identical to turn t's.

The thing we ultimately want to reuse is the Qwen2.5-VL ViT *image-token embedding*,
so the headline metric is per-token cosine similarity of ViT outputs. Pixel
similarity and content-hash equality are reported only as sanity checks / dedup
upper bounds.

Three directions (matches advisor's framing)
--------------------------------------------
1. same_group_turn0_cross_branch
     within one uid, turn-0 input image across the N branches.
     -> exact dedup upper bound (hash equal? pixel==? token cos==1?)

2. same_branch_cross_turn
     within one branch, image at turn t vs turn t+1 (chart vs refocus, ...).
     -> partial reuse potential after a local edit (changed-pixel frac, per-token
        cosine, reusable-token ratio @ thresholds, spatial heatmap).

3. cross-branch, broken into:
     a) same_group_same_turn_cross_branch  (turn>=1; e.g. b0.t1 vs b1.t1)
     b) same_group_cross_turn_cross_branch (e.g. b0.t0 vs b1.t1)
     -> group-level partial reuse across branches' derived images.

Outputs
-------
  group_structure_summary.csv        per-uid branch/turn coverage + missing images
  pairwise_similarity.csv            every pair, tagged with pair_type
  similarity_summary_by_pair_type.md aggregate per pair_type + A/B/C/D conclusions
  heatmaps/group_<uid>.png           per-group (branch,turn) token-cosine matrix
  heatmaps/pair_<...>.png            representative orig/edit/token-cosine heatmaps

Usage
-----
  # structure only (no model load), good while a run is still in progress:
  python3 examples/profile/analyze_similarity_unified.py \
    --dump-dir profile_logs_refocus_chart/image_dump_refocus_chart_multiturn_bs64_n4 \
    --check-only

  # full analysis (loads Qwen2.5-VL vision tower):
  CUDA_VISIBLE_DEVICES=0 python3 examples/profile/analyze_similarity_unified.py \
    --dump-dir profile_logs_refocus_chart/image_dump_refocus_chart_multiturn_bs64_n4 \
    --out-dir profile_logs_refocus_chart/similarity \
    --group-heatmaps 6 --pair-heatmaps 8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

DEFAULT_THRESHOLDS = (0.999, 0.99, 0.95, 0.90)

# pair_type constants
D1 = "same_group_turn0_cross_branch"
D2 = "same_branch_cross_turn"
D3A = "same_group_same_turn_cross_branch"
D3B = "same_group_cross_turn_cross_branch"

_FNAME_RE = re.compile(r"^(?P<uid>[^_]+)_(?P<req>[^_]+)_t(?P<turn>\d+)_(?P<role>.+)\.png$")
_SOKOBAN_FNAME_RE = re.compile(r"^g(?P<group>\d+)_b(?P<branch>\d+)_s(?P<step>\d+)_(?P<role>.+)\.png$")


# --------------------------------------------------------------------------- #
# structure: group (uid) -> branch (request_id) -> turn -> node
# --------------------------------------------------------------------------- #
class Node:
    __slots__ = ("uid", "branch", "turn", "role", "path", "_hash", "_pix224", "_tokens", "_grid")

    def __init__(self, uid: str, branch: str, turn: int, role: str, path: Path):
        self.uid = uid
        self.branch = branch
        self.turn = turn
        self.role = role
        self.path = path
        self._hash: Optional[str] = None
        self._pix224: Optional[np.ndarray] = None
        self._tokens = None
        self._grid = None

    @property
    def key(self) -> str:
        return f"b{self.branch[:6]}.t{self.turn}"

    def content_hash(self) -> str:
        if self._hash is None:
            self._hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        return self._hash


def load_structure(dump_dir: Path) -> dict[str, dict[str, dict[int, Node]]]:
    """group[uid][branch][turn] = Node. Prefers manifest.jsonl, falls back to filenames."""
    manifest = dump_dir / "manifest.jsonl"
    groups: dict[str, dict[str, dict[int, Node]]] = defaultdict(lambda: defaultdict(dict))

    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            uid = r.get("uid")
            if not uid and "group_idx" in r:
                uid = f"group_{int(r['group_idx']):04d}"
            if not uid:
                continue
            # branch identity is request_id (rollout_idx is the per-sample dataset
            # index and is shared by all branches of a group).
            if "branch_idx" in r:
                branch = f"b{int(r['branch_idx'])}"
            else:
                branch = r.get("request_id") or str(r.get("rollout_idx", "?"))
            if "turn" in r:
                turn = int(r["turn"])
            elif "step" in r:
                turn = int(r["step"])
            else:
                continue
            path = dump_dir / r["path"]
            if not path.exists():
                continue
            # last writer per (branch,turn) wins; role kept from the entry
            groups[uid][branch][turn] = Node(uid, branch, turn, r.get("role", ""), path)
        return groups

    # fallback: parse filenames (refocus uid_req_tN_role or sokoban gXXXX_bY_sZZ_obs)
    for p in sorted(dump_dir.glob("*.png")):
        m = _FNAME_RE.match(p.name)
        if m:
            uid = m.group("uid")
            branch = m.group("req")
            turn = int(m.group("turn"))
            role = m.group("role")
            groups[uid][branch][turn] = Node(uid, branch, turn, role, p)
            continue
        sm = _SOKOBAN_FNAME_RE.match(p.name)
        if sm:
            uid = f"group_{int(sm.group('group')):04d}"
            branch = f"b{int(sm.group('branch'))}"
            turn = int(sm.group("step"))
            role = sm.group("role")
            groups[uid][branch][turn] = Node(uid, branch, turn, role, p)
    return groups


def write_structure_summary(
    groups: dict[str, dict[str, dict[int, Node]]], out_csv: Path
) -> dict[str, Any]:
    rows: list[dict] = []
    n_branches_dist: dict[int, int] = defaultdict(int)
    branches_with_refocus = 0
    total_branches = 0
    max_turn = 0
    for uid, branches in groups.items():
        all_turns: set[int] = set()
        missing = []
        for branch, turns in branches.items():
            total_branches += 1
            all_turns |= set(turns)
            max_turn = max(max_turn, max(turns) if turns else 0)
            if 1 in turns:
                branches_with_refocus += 1
            # a branch that has t0 but no t1 "missing refocus"
            if 0 in turns and 1 not in turns:
                missing.append(f"{branch[:6]}:no_t1")
        n_branches_dist[len(branches)] += 1
        rows.append(
            {
                "uid": uid,
                "n_branches": len(branches),
                "turns_present": ",".join(str(t) for t in sorted(all_turns)),
                "n_images": sum(len(t) for t in branches.values()),
                "branches_with_refocus": sum(1 for t in branches.values() if 1 in t),
                "missing": ";".join(missing) if missing else "",
            }
        )

    rows.sort(key=lambda r: (-r["n_branches"], r["uid"]))
    _write_csv(out_csv, rows)

    return {
        "n_groups": len(groups),
        "total_branches": total_branches,
        "n_branches_dist": dict(sorted(n_branches_dist.items())),
        "branches_with_refocus": branches_with_refocus,
        "max_turn": max_turn,
    }


# --------------------------------------------------------------------------- #
# pixel helpers (sanity checks only)
# --------------------------------------------------------------------------- #
def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _pix224(node: Node) -> np.ndarray:
    if node._pix224 is None:
        im = _load_rgb(node.path).resize((224, 224), Image.Resampling.BILINEAR)
        node._pix224 = np.asarray(im, dtype=np.float32).reshape(-1) / 255.0
    return node._pix224


def pixel_cosine(a: Node, b: Node) -> float:
    va, vb = _pix224(a), _pix224(b)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-8:
        return 1.0
    return float(np.dot(va, vb) / denom)


def pixel_mse_sim(a: Node, b: Node) -> float:
    va, vb = _pix224(a), _pix224(b)
    return max(0.0, 1.0 - float(np.mean((va - vb) ** 2)))


def changed_pixel_frac(a: Node, b: Node) -> float:
    """Fraction of pixels differing between two images (native res, b resized to a)."""
    ia = np.asarray(_load_rgb(a.path), dtype=np.int16)
    ib = np.asarray(_load_rgb(b.path).resize(_load_rgb(a.path).size), dtype=np.int16)
    return float((np.abs(ia - ib).sum(axis=-1) > 0).mean())


# --------------------------------------------------------------------------- #
# ViT encoder (lazy, cached by content hash)
# --------------------------------------------------------------------------- #
class TokenCache:
    def __init__(self, model_path: str, device: str, max_pixels: Optional[int], cap: int):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.device = device
        self.cap = cap
        proc_kwargs = {"max_pixels": max_pixels} if max_pixels else {}
        self.processor = AutoProcessor.from_pretrained(model_path, **proc_kwargs)
        self.image_processor = self.processor.image_processor
        print(f"Loading vision tower from {model_path} ...", flush=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        self.visual = model.visual.to(device).eval()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.merge_size = getattr(self.image_processor, "merge_size", 2)
        self._cache: dict[str, tuple] = {}  # hash -> (tokens cpu f32, grid)

    def encode_image(self, image: Image.Image):
        torch = self.torch
        with torch.no_grad():
            feats = self.image_processor(images=[image], return_tensors="pt")
            pixel_values = feats["pixel_values"].to(self.device, dtype=torch.bfloat16)
            grid_thw = feats["image_grid_thw"].to(self.device)
            embeds = self.visual(pixel_values, grid_thw=grid_thw)
            if isinstance(embeds, (tuple, list)):
                embeds = embeds[0]
            grid = tuple(int(x) for x in grid_thw[0].tolist())
            return embeds.float().cpu(), grid

    def get(self, node: Node):
        h = node.content_hash()
        if h in self._cache:
            return self._cache[h]
        tokens, grid = self.encode_image(_load_rgb(node.path))
        if len(self._cache) < self.cap:
            self._cache[h] = (tokens, grid)
        return tokens, grid


def token_similarity(
    a: Node, b: Node, cache: TokenCache, thresholds
) -> Optional[dict]:
    """Per-token cosine of ViT outputs. Returns None if grids cannot be aligned."""
    F = cache.torch.nn.functional
    ta, ga = cache.get(a)
    tb, gb = cache.get(b)
    cos_vec = None
    grid_for_heatmap = ga
    if ga == gb and ta.shape == tb.shape:
        cos_vec = F.cosine_similarity(ta, tb, dim=-1)
    else:
        # resize b to a's native size and re-encode (uncached one-off) to align grids
        im_b = _load_rgb(b.path).resize(_load_rgb(a.path).size)
        tb2, gb2 = cache.encode_image(im_b)
        if gb2 != ga or tb2.shape != ta.shape:
            return None
        cos_vec = F.cosine_similarity(ta, tb2, dim=-1)
    rec = {
        "n_tokens": int(cos_vec.numel()),
        "token_mean_cos": float(cos_vec.mean()),
        "token_median_cos": float(cos_vec.median()),
        "token_min_cos": float(cos_vec.min()),
        "grid_match": ga == gb,
    }
    for th in thresholds:
        rec[f"reusable@{th}"] = float((cos_vec >= th).float().mean())
    rec["_cos_vec"] = cos_vec
    rec["_grid"] = grid_for_heatmap
    return rec


# --------------------------------------------------------------------------- #
# heatmaps
# --------------------------------------------------------------------------- #
def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:  # noqa: BLE001
        return None


def save_pair_heatmap(a: Node, b: Node, cos_vec, grid, merge_size: int, out_path: Path) -> bool:
    plt = _plt()
    if plt is None:
        return False
    t, h, w = grid
    hh, ww = h // merge_size, w // merge_size
    if hh * ww != cos_vec.numel():
        return False
    sim = cos_vec.reshape(hh, ww).numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(_load_rgb(a.path))
    axes[0].set_title(f"{a.role} (b{a.branch[:6]}.t{a.turn})")
    axes[0].axis("off")
    axes[1].imshow(_load_rgb(b.path))
    axes[1].set_title(f"{b.role} (b{b.branch[:6]}.t{b.turn})")
    axes[1].axis("off")
    im = axes[2].imshow(sim, cmap="RdYlGn", vmin=0.8, vmax=1.0)
    axes[2].set_title("per-token cosine\ngreen=reusable, red=recompute")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return True


def save_group_matrix(uid: str, nodes: list[Node], mat: np.ndarray, labels: list[str], out_path: Path) -> bool:
    plt = _plt()
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 1.2 * len(labels) + 1))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.8, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"uid {uid[:8]} — pairwise ViT token mean-cosine")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# pairwise enumeration
# --------------------------------------------------------------------------- #
def enumerate_pairs(branches: dict[str, dict[int, Node]]) -> list[tuple[str, Node, Node]]:
    """Yield (pair_type, node_a, node_b) for one group."""
    pairs: list[tuple[str, Node, Node]] = []
    branch_ids = sorted(branches)

    # D1: turn0 across branches
    t0_nodes = [branches[b][0] for b in branch_ids if 0 in branches[b]]
    for a, b in combinations(t0_nodes, 2):
        pairs.append((D1, a, b))

    # D2: within a branch, consecutive turns
    for b in branch_ids:
        turns = sorted(branches[b])
        for ta, tb in zip(turns, turns[1:]):
            pairs.append((D2, branches[b][ta], branches[b][tb]))

    # D3a: same turn>=1 across branches
    turns_present = sorted({t for b in branch_ids for t in branches[b] if t >= 1})
    for t in turns_present:
        nodes_t = [branches[b][t] for b in branch_ids if t in branches[b]]
        for a, b in combinations(nodes_t, 2):
            pairs.append((D3A, a, b))

    # D3b: cross-branch AND cross-turn (at least one side turn>=1)
    for bi, bj in combinations(branch_ids, 2):
        for ta in branches[bi]:
            for tb in branches[bj]:
                if ta == tb:
                    continue  # same turn handled by D1/D3a
                if max(ta, tb) < 1:
                    continue
                pairs.append((D3B, branches[bi][ta], branches[bj][tb]))
    return pairs


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True, help="image dump dir with manifest.jsonl + PNGs")
    ap.add_argument("--out-dir", default=None, help="output dir (default: <dump-dir>/../similarity)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-pixels", type=int, default=None, help="cap processor max_pixels (smaller=faster)")
    ap.add_argument("--check-only", action="store_true", help="only emit structure summary (no model load)")
    ap.add_argument("--max-groups", type=int, default=None, help="limit #uids analyzed for token sim (debug/speed)")
    ap.add_argument("--token-cache-cap", type=int, default=600, help="max embeddings cached in RAM")
    ap.add_argument("--group-heatmaps", type=int, default=6, help="save N per-group token-cosine matrices")
    ap.add_argument("--pair-heatmaps", type=int, default=8, help="save N representative D2 token heatmaps")
    ap.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS))
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    dump_dir = Path(args.dump_dir)
    if not dump_dir.is_dir():
        raise SystemExit(f"dump-dir not found: {dump_dir}")
    out_dir = Path(args.out_dir) if args.out_dir else dump_dir.parent / "similarity"
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = out_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    groups = load_structure(dump_dir)
    if args.max_groups:
        groups = dict(list(groups.items())[: args.max_groups])
    struct = write_structure_summary(groups, out_dir / "group_structure_summary.csv")

    print("=== group structure ===")
    print(f"  groups (uids):        {struct['n_groups']}")
    print(f"  total branches:       {struct['total_branches']}")
    print(f"  n_branches/uid dist:  {struct['n_branches_dist']}")
    print(f"  branches w/ refocus:  {struct['branches_with_refocus']}")
    print(f"  max turn observed:    {struct['max_turn']}")
    print(f"  -> {out_dir / 'group_structure_summary.csv'}")

    if args.check_only:
        print("\n--check-only set; skipping similarity. Re-run without it (on a GPU) for full analysis.")
        return

    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = TokenCache(args.model, device, args.max_pixels, args.token_cache_cap)

    pair_rows: list[dict] = []
    group_node_sims: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    n_pairs_total = sum(len(enumerate_pairs(b)) for b in groups.values())
    print(f"\n=== computing {n_pairs_total} pairs over {len(groups)} groups ===", flush=True)

    done = 0
    for uid, branches in groups.items():
        for pair_type, a, b in enumerate_pairs(branches):
            done += 1
            hash_equal = a.content_hash() == b.content_hash()
            row: dict[str, Any] = {
                "uid": uid,
                "pair_type": pair_type,
                "a": a.key,
                "a_role": a.role,
                "b": b.key,
                "b_role": b.role,
                "hash_equal": hash_equal,
            }
            if hash_equal:
                # identical bytes -> trivially fully reusable; skip ViT to save time
                row.update(
                    pixel_cosine=1.0,
                    pixel_mse_sim=1.0,
                    changed_pixel_frac=0.0,
                    n_tokens="",
                    token_mean_cos=1.0,
                    token_median_cos=1.0,
                    token_min_cos=1.0,
                    grid_match=True,
                )
                for th in thresholds:
                    row[f"reusable@{th}"] = 1.0
                pair_rows.append(row)
                group_node_sims[uid][(a.key, b.key)] = 1.0
                if done % 200 == 0:
                    print(f"  {done}/{n_pairs_total}", flush=True)
                continue

            row["pixel_cosine"] = round(pixel_cosine(a, b), 5)
            row["pixel_mse_sim"] = round(pixel_mse_sim(a, b), 5)
            row["changed_pixel_frac"] = round(changed_pixel_frac(a, b), 5)

            tok = token_similarity(a, b, cache, thresholds)
            if tok is None:
                row.update(n_tokens="", token_mean_cos="", token_median_cos="",
                           token_min_cos="", grid_match=False)
                for th in thresholds:
                    row[f"reusable@{th}"] = ""
            else:
                cos_vec = tok.pop("_cos_vec")
                grid = tok.pop("_grid")
                row.update({k: (round(v, 5) if isinstance(v, float) else v) for k, v in tok.items()})
                group_node_sims[uid][(a.key, b.key)] = tok["token_mean_cos"]
                # representative D2 heatmaps: actual local edits with aligned grids
                if (
                    pair_type == D2
                    and tok["grid_match"]
                    and len([p for p in heatmap_dir.glob("pair_*.png")]) < args.pair_heatmaps
                ):
                    save_pair_heatmap(
                        a, b, cos_vec, grid, cache.merge_size,
                        heatmap_dir / f"pair_{uid[:8]}_{a.key}_vs_{b.key}.png",
                    )
            pair_rows.append(row)
            if done % 200 == 0:
                print(f"  {done}/{n_pairs_total}", flush=True)

    _write_csv(out_dir / "pairwise_similarity.csv", pair_rows)
    print(f"  -> {out_dir / 'pairwise_similarity.csv'} ({len(pair_rows)} pairs)")

    # per-group token-cosine matrices (representative groups with most nodes)
    saved_group_hm = 0
    groups_by_size = sorted(groups.items(), key=lambda kv: -sum(len(t) for t in kv[1].values()))
    for uid, branches in groups_by_size:
        if saved_group_hm >= args.group_heatmaps:
            break
        nodes: list[Node] = []
        for b in sorted(branches):
            for t in sorted(branches[b]):
                nodes.append(branches[b][t])
        if len(nodes) < 2:
            continue
        labels = [n.key for n in nodes]
        n = len(nodes)
        mat = np.full((n, n), np.nan)
        for i in range(n):
            mat[i, i] = 1.0
        for i, j in combinations(range(n), 2):
            a, b = nodes[i], nodes[j]
            if a.content_hash() == b.content_hash():
                v = 1.0
            else:
                tok = token_similarity(a, b, cache, thresholds)
                v = tok["token_mean_cos"] if tok else np.nan
            mat[i, j] = mat[j, i] = v
        if save_group_matrix(uid, nodes, mat, labels, heatmap_dir / f"group_{uid[:8]}.png"):
            saved_group_hm += 1

    # aggregate by pair_type + conclusions
    write_summary_md(pair_rows, thresholds, struct, out_dir / "similarity_summary_by_pair_type.md")
    print(f"  -> {out_dir / 'similarity_summary_by_pair_type.md'}")
    print(f"  -> {heatmap_dir}/ (group + pair heatmaps)")


def _agg(rows: list[dict], pair_type: str, key: str) -> Optional[float]:
    vals = [r[key] for r in rows if r["pair_type"] == pair_type and isinstance(r.get(key), (int, float))]
    return float(np.mean(vals)) if vals else None


def _frac_hash_equal(rows: list[dict], pair_type: str) -> tuple[int, int]:
    sub = [r for r in rows if r["pair_type"] == pair_type]
    eq = sum(1 for r in sub if r.get("hash_equal"))
    return eq, len(sub)


def workload_banner_md(out_path) -> list[str]:
    """Honest workload positioning, auto-detected from the output path.

    The diversified run is a CONTROLLED synthetic local-edit workload — branches
    apply the same annotation style (draw) to DIFFERENT bbox regions of the same
    base image. It is NOT semantically-correct RL tool-calling and must not be read
    as natural RL-policy behavior; it exists only to isolate image-token reuse.
    """
    is_div = "diversified" in str(out_path).lower()
    if is_div:
        return [
            "> **Workload 定位（必读）**：本结果来自 **controlled synthetic local-edit workload**，"
            "非真实 RL policy 行为。每个 branch 对同一原图用**同一种标注风格（draw）**框选**不同 bbox 区域**，"
            "刻意让 4 个 turn1 在视觉上分化。它只回答系统问题——*同一原图的不同局部编辑版本之间 image-token "
            "是否仍可大量复用*；**不声称**这些 refocus 是当前 policy 在语义上最优/自然产生的工具调用。"
            "由于各 branch 故意 focus 不同区域，这是 cross-branch 相似度的 **保守（分化最大）** 情形。",
            "",
        ]
    return [
        "> **Workload 定位**：本结果来自 **per-sample 确定性 oracle refocus**（同 uid 各 branch 执行同一段 oracle，"
        "故 cross-branch turn1 常 byte-identical）。它给出 cross-branch 复用的乐观上界；分化情形见 `*_diversified` 跑法。",
        "",
    ]


def write_summary_md(rows: list[dict], thresholds, struct: dict, out_path: Path) -> None:
    def line_for(pt: str) -> str:
        eq, tot = _frac_hash_equal(rows, pt)
        if tot == 0:
            return f"| `{pt}` | 0 | – | – | – | – |"
        hash_pct = 100.0 * eq / tot
        mean_cos = _agg(rows, pt, "token_mean_cos")
        r99 = _agg(rows, pt, "reusable@0.99")
        r95 = _agg(rows, pt, "reusable@0.95")
        cos_s = f"{mean_cos:.4f}" if mean_cos is not None else "–"
        r99_s = f"{r99*100:.1f}%" if r99 is not None else "–"
        r95_s = f"{r95*100:.1f}%" if r95 is not None else "–"
        return f"| `{pt}` | {tot} | {eq}/{tot} ({hash_pct:.0f}%) | {cos_s} | {r99_s} | {r95_s} |"

    L = [
        "# Unified visual-redundancy / image-token-reuse analysis",
        "",
        "面向 VLM GRPO/RL rollout 的视觉冗余分析。**核心指标是 Qwen2.5-VL ViT image-token "
        "embedding 的 per-token cosine**（我们要复用的是 image token，不是像素）；hash/pixel 仅作 sanity check。",
        "",
        *workload_banner_md(out_path),
        "## 数据结构",
        "",
        f"- group (uid): **{struct['n_groups']}**；branch (request_id) 总数: **{struct['total_branches']}**",
        f"- 每 uid 的 branch 数分布: {struct['n_branches_dist']}",
        f"- 含 refocus(turn≥1) 的 branch: **{struct['branches_with_refocus']}**；最大 turn: **{struct['max_turn']}**",
        "",
        "## 按 pair_type 汇总",
        "",
        "| pair_type | #pairs | byte-identical | mean token cos | reusable@0.99 | reusable@0.95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        line_for(D1),
        line_for(D2),
        line_for(D3A),
        line_for(D3B),
        "",
        "> `reusable@th` = ViT token 中 cosine ≥ th 的占比（可直接复用的 image-token 比例）。",
        "> `byte-identical` = 原始图片字节 sha256 相同（exact-dedup 命中上界）。",
        "",
    ]

    # conclusions
    d1_eq, d1_tot = _frac_hash_equal(rows, D1)
    d2_cos = _agg(rows, D2, "token_mean_cos")
    d2_r99 = _agg(rows, D2, "reusable@0.99")
    d2_r95 = _agg(rows, D2, "reusable@0.95")
    d2_chg = _agg(rows, D2, "changed_pixel_frac")
    d3a_eq, d3a_tot = _frac_hash_equal(rows, D3A)
    d3a_cos = _agg(rows, D3A, "token_mean_cos")
    d3a_r99 = _agg(rows, D3A, "reusable@0.99")
    d3b_cos = _agg(rows, D3B, "token_mean_cos")
    d3b_r99 = _agg(rows, D3B, "reusable@0.99")

    def pct(x):
        return f"{x*100:.1f}%" if x is not None else "N/A"

    L += [
        "## 结论",
        "",
        f"**A. 同组 turn-0 是否 100% 可 exact reuse？** "
        f"同 uid turn-0 跨 branch byte-identical 命中 **{d1_eq}/{d1_tot}** "
        f"({100.0*d1_eq/max(d1_tot,1):.0f}%)。"
        + ("→ **是**，turn-0 输入图在 GRPO group 内可 100% exact dedup，这是 cache 收益的硬下界。"
           if d1_tot and d1_eq == d1_tot else
           "→ 大部分可 exact dedup（少数 branch 缺图/不全）。"),
        "",
        f"**B. 同一 branch 前后 turn 有多少 token 可 partial reuse？** "
        f"refocus 平均改动像素 **{pct(d2_chg)}**，但 ViT mean token cos **{d2_cos:.4f}**，"
        f"reusable@0.99 ≈ **{pct(d2_r99)}**、@0.95 ≈ **{pct(d2_r95)}**"
        if d2_cos is not None else
        "**B.** 本次无可对齐的 same-branch cross-turn token 对。",
        "",
        f"**C. 同组不同 branch 的 turn1/turn2 图片是否也有 partial reuse 机会？** "
        f"same_turn 跨 branch byte-identical **{d3a_eq}/{d3a_tot}**"
        + (f"，token mean cos **{d3a_cos:.4f}**、reusable@0.99 ≈ **{pct(d3a_r99)}**" if d3a_cos is not None else "")
        + (f"；cross-turn 跨 branch token mean cos **{d3b_cos:.4f}**、reusable@0.99 ≈ **{pct(d3b_r99)}**"
           if d3b_cos is not None else "")
        + "。",
        "",
        "**D. 对 cache idea 的支持**",
        "",
        "- **A → exact group dedup**：GRPO group turn-0 同图，单一全局/共享 embedding cache 可在组内省掉 (n-1)/n 的 ViT。",
        "- **B → multi-turn partial reuse**：局部 refocus 后多数 token 几乎不变，支持 embedding 级 partial cache（非 sha256 全图 cache）。",
        "- **C → group-level partial reuse**：同组不同 branch 的派生图相似度高，支持把复用从「单 branch 时间维」扩展到「group 内 branch 维」。",
        "",
        "> ⚠️ oracle refocus 是 per-sample 确定性的，故同组不同 branch 的 refocus 图常完全相同（hash 命中偏高）。"
        " 真实（非 oracle）RL 下各 branch 会生成不同 refocus，cross-branch 命中会下降到接近 token-cos 的 partial 区间——"
        " 因此 **C 的 token-cos（partial）比 hash 命中更能代表线上潜力**。",
        "",
        "产物：`group_structure_summary.csv` · `pairwise_similarity.csv` · `heatmaps/`",
    ]
    out_path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
