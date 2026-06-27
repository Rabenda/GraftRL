#!/usr/bin/env python3
"""Per-group case study for the image-token-reuse analysis.

For a handful of representative GRPO groups (uids), dump everything needed to
*eyeball* the visual redundancy claims:

  group_<uid>/
    branch0_turn0.png
    branch0_turn1_refocus.png        (if the branch produced a refocus image)
    branch1_turn0.png
    ...
    pairwise_similarity.csv          one row per image pair
    summary.txt                      per-image table + aggregates + A/B/C/D notes

Per image we record: content sha256, ViT image-token count.
Per pair we record: pixel cosine, ViT image-token mean/median/min cosine, and
reusable token COUNT + RATIO at cosine thresholds 0.999 / 0.99 / 0.95 / 0.90.

Group selection (auto, unless --uids given): we read the pairwise_similarity.csv
produced by analyze_similarity_unified.py to rank groups into three buckets:
  1. consistent    turn0 identical + turn1 highly consistent across branches
  2. partial       turn0 identical + turn1 diverges but token reuse stays high
  3. divergent     turn1 changed a lot (low same-branch t0->t1 token cosine)

Usage
-----
  CUDA_VISIBLE_DEVICES=0 python3 examples/profile/case_study_groups.py \
    --dump-dir profile_logs_refocus_chart/image_dump_refocus_chart_multiturn_bs64_n4 \
    --pairwise-csv profile_logs_refocus_chart/similarity/pairwise_similarity.csv \
    --out-dir profile_logs_refocus_chart/similarity/case_studies \
    --per-bucket 1

  # or hand-pick groups:
  CUDA_VISIBLE_DEVICES=0 python3 examples/profile/case_study_groups.py \
    --dump-dir <dump> --uids 4dc1f0c6-...,e7cb3bf6-... --out-dir <out>
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np

from examples.profile.shared.analysis.analyze_similarity_unified import (
    Node,
    TokenCache,
    changed_pixel_frac,
    load_structure,
    pixel_cosine,
    pixel_mse_sim,
)

DEFAULT_THRESHOLDS = (0.999, 0.99, 0.95, 0.90)


# --------------------------------------------------------------------------- #
# group selection
# --------------------------------------------------------------------------- #
def distinct_real_turn1(groups: dict, dump_dir: Path) -> dict[str, int]:
    """Per uid: number of DISTINCT turn1 refocus images that actually differ from turn0.

    Under oracle (deterministic per-sample refocus) branches usually share one turn1
    image; groups with >=2 distinct real turn1 images are the only ones where a
    *cross-branch* turn1 comparison is non-trivial.
    """
    import hashlib

    out: dict[str, int] = {}
    for uid, branches in groups.items():
        t0_hashes = set()
        t1_hashes = set()
        for b, turns in branches.items():
            if 0 in turns:
                t0_hashes.add(turns[0].content_hash())
            if 1 in turns:
                t1_hashes.add(turns[1].content_hash())
        out[uid] = len(t1_hashes - t0_hashes)
    return out


def select_groups_from_csv(csv_path: Path, per_bucket: int) -> dict[str, list[str]]:
    """Rank uids into consistent / partial / divergent buckets using pairwise CSV."""
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))

    def f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    # per-uid aggregates
    d3a_cos: dict[str, list[float]] = defaultdict(list)  # cross-branch same-turn (turn>=1)
    d3a_hashident: dict[str, list[int]] = defaultdict(list)
    d2_cos: dict[str, list[float]] = defaultdict(list)  # same-branch t0->t1
    full_t1: dict[str, int] = defaultdict(int)

    for r in rows:
        uid = r["uid"]
        pt = r["pair_type"]
        cos = f(r.get("token_mean_cos"))
        if pt == "same_group_same_turn_cross_branch":
            if cos is not None:
                d3a_cos[uid].append(cos)
            d3a_hashident[uid].append(1 if r.get("hash_equal") == "True" else 0)
        elif pt == "same_branch_cross_turn":
            if cos is not None:
                d2_cos[uid].append(cos)

    uids = set(d2_cos) | set(d3a_cos)
    stats = {}
    for uid in uids:
        stats[uid] = {
            "d3a_cos": float(np.mean(d3a_cos[uid])) if d3a_cos[uid] else None,
            "d3a_hashident": float(np.mean(d3a_hashident[uid])) if d3a_hashident[uid] else None,
            "d2_cos": float(np.mean(d2_cos[uid])) if d2_cos[uid] else None,
            "n_d3a": len(d3a_cos[uid]),
        }

    # bucket 1: consistent -> many cross-branch t1 pairs, highest d3a_cos
    consistent = sorted(
        (u for u in uids if stats[u]["d3a_cos"] is not None and stats[u]["n_d3a"] >= 3),
        key=lambda u: -stats[u]["d3a_cos"],
    )
    # bucket 2: partial -> cross-branch t1 NOT all byte-identical, but token cos still high
    partial = sorted(
        (
            u
            for u in uids
            if stats[u]["d3a_cos"] is not None
            and stats[u]["d3a_hashident"] is not None
            and stats[u]["d3a_hashident"] < 0.999
            and stats[u]["d3a_cos"] >= 0.95
        ),
        key=lambda u: -stats[u]["d3a_cos"],
    )
    # bucket 3: divergent -> lowest same-branch t0->t1 token cosine (big refocus edit)
    divergent = sorted(
        (u for u in uids if stats[u]["d2_cos"] is not None),
        key=lambda u: stats[u]["d2_cos"],
    )

    picked: dict[str, list[str]] = {"consistent": [], "partial": [], "divergent": []}
    used: set[str] = set()

    def take(bucket_name, ordered):
        for u in ordered:
            if len(picked[bucket_name]) >= per_bucket:
                break
            if u in used:
                continue
            picked[bucket_name].append(u)
            used.add(u)

    take("partial", partial)      # most specific first so it isn't stolen
    take("divergent", divergent)
    take("consistent", consistent)
    return picked


# --------------------------------------------------------------------------- #
# per-group dump
# --------------------------------------------------------------------------- #
def _branch_label(idx: int) -> str:
    return f"branch{idx}"


def process_group(
    uid: str,
    branches: dict[str, dict[int, Node]],
    cache: TokenCache,
    out_dir: Path,
    thresholds,
    bucket: str,
) -> dict:
    F = cache.torch.nn.functional
    gdir = out_dir / f"group_{uid[:8]}"
    gdir.mkdir(parents=True, exist_ok=True)

    # stable branch ordering by request_id
    branch_ids = sorted(branches)
    # named nodes: label -> Node
    named: dict[str, Node] = {}
    img_rows: list[dict] = []
    for bi, b in enumerate(branch_ids):
        for turn, node in sorted(branches[b].items()):
            label = f"{_branch_label(bi)}_turn{turn}" + ("_refocus" if turn >= 1 else "")
            named[label] = node
            # copy image into the group dir under the readable name
            shutil.copyfile(node.path, gdir / f"{label}.png")
            tokens, grid = cache.get(node)
            img_rows.append(
                {
                    "label": label,
                    "branch": b[:12],
                    "turn": turn,
                    "role": node.role,
                    "sha256": node.content_hash(),
                    "n_image_tokens": int(tokens.shape[0]),
                    "grid_thw": "x".join(str(x) for x in grid),
                }
            )

    # pairwise over all named nodes
    labels = list(named)
    pair_rows: list[dict] = []
    for la, lb in combinations(labels, 2):
        a, b = named[la], named[lb]
        hash_equal = a.content_hash() == b.content_hash()
        row = {
            "pair": f"{la} vs {lb}",
            "a": la,
            "b": lb,
            "hash_equal": hash_equal,
        }
        if hash_equal:
            ta, _ = cache.get(a)
            n = int(ta.shape[0])
            row.update(
                pixel_cosine=1.0,
                pixel_mse_sim=1.0,
                changed_pixel_frac=0.0,
                n_tokens=n,
                token_mean_cos=1.0,
                token_median_cos=1.0,
                token_min_cos=1.0,
                grid_match=True,
            )
            for th in thresholds:
                row[f"reusable_count@{th}"] = n
                row[f"reusable_ratio@{th}"] = 1.0
            pair_rows.append(row)
            continue

        row["pixel_cosine"] = round(pixel_cosine(a, b), 5)
        row["pixel_mse_sim"] = round(pixel_mse_sim(a, b), 5)
        row["changed_pixel_frac"] = round(changed_pixel_frac(a, b), 5)

        ta, ga = cache.get(a)
        tb, gb = cache.get(b)
        if ga == gb and ta.shape == tb.shape:
            cos_vec = F.cosine_similarity(ta, tb, dim=-1)
            grid_match = True
        else:
            from PIL import Image  # local import to avoid hard dep at module load

            im_b = Image.open(b.path).convert("RGB").resize(Image.open(a.path).size)
            tb2, gb2 = cache.encode_image(im_b)
            if gb2 != ga or tb2.shape != ta.shape:
                row.update(
                    n_tokens="", token_mean_cos="", token_median_cos="",
                    token_min_cos="", grid_match=False,
                )
                for th in thresholds:
                    row[f"reusable_count@{th}"] = ""
                    row[f"reusable_ratio@{th}"] = ""
                pair_rows.append(row)
                continue
            cos_vec = F.cosine_similarity(ta, tb2, dim=-1)
            grid_match = False

        n = int(cos_vec.numel())
        row.update(
            n_tokens=n,
            token_mean_cos=round(float(cos_vec.mean()), 5),
            token_median_cos=round(float(cos_vec.median()), 5),
            token_min_cos=round(float(cos_vec.min()), 5),
            grid_match=grid_match,
        )
        for th in thresholds:
            cnt = int((cos_vec >= th).sum())
            row[f"reusable_count@{th}"] = cnt
            row[f"reusable_ratio@{th}"] = round(cnt / n, 5)
        pair_rows.append(row)

    _write_csv(gdir / "pairwise_similarity.csv", pair_rows)

    # dedicated turn1-only cross-branch view (the "do branches' refocus images stay
    # cache-similar?" question). matrix of token_mean_cos among all *_turn1_refocus.
    t1_labels = [lab for lab in labels if "turn1" in lab]
    t1_rows = [
        r for r in pair_rows
        if "turn1" in r["a"] and "turn1" in r["b"]
    ]
    _write_csv(gdir / "turn1_cross_branch.csv", t1_rows)

    write_group_summary(uid, bucket, img_rows, pair_rows, t1_labels, named, cache, thresholds, gdir / "summary.txt")
    return {"uid": uid, "bucket": bucket, "dir": gdir, "n_images": len(img_rows), "n_pairs": len(pair_rows)}


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


def _pair_subset(pair_rows, pred):
    return [r for r in pair_rows if pred(r)]


def _mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return float(np.mean(vals)) if vals else None


def write_group_summary(uid, bucket, img_rows, pair_rows, t1_labels, named, cache, thresholds, out_path: Path) -> None:
    def turn_of(label):
        return int(label.split("turn")[1].split("_")[0])

    def branch_of(label):
        return label.split("_")[0]

    # subsets
    t0t0 = _pair_subset(pair_rows, lambda r: turn_of(r["a"]) == 0 and turn_of(r["b"]) == 0)
    same_branch_cross_turn = _pair_subset(
        pair_rows, lambda r: branch_of(r["a"]) == branch_of(r["b"]) and turn_of(r["a"]) != turn_of(r["b"])
    )
    cross_branch_t1 = _pair_subset(
        pair_rows,
        lambda r: branch_of(r["a"]) != branch_of(r["b"]) and turn_of(r["a"]) == 1 and turn_of(r["b"]) == 1,
    )

    is_div = "diversified" in str(out_path).lower()
    L = [
        f"GROUP CASE STUDY — uid={uid}",
        f"bucket: {bucket}",
        "=" * 72,
        "",
    ]
    if is_div:
        L += [
            "[WORKLOAD] controlled synthetic local-edit workload (NOT real RL policy).",
            "  Each branch applies the SAME annotation style (draw) to a DIFFERENT bbox",
            "  region of the same base image -> deliberately divergent turn1 images.",
            "  Measures image-token reuse of different local edits; does NOT claim these",
            "  refocus ops are semantically-optimal/natural tool calls. Conservative",
            "  (max-divergence) case for cross-branch similarity.",
            "",
        ]
    else:
        L += [
            "[WORKLOAD] per-sample deterministic oracle refocus (optimistic upper bound;",
            "  same-uid branches run identical oracle -> cross-branch turn1 often identical).",
            "",
        ]
    L += [
        "[per-image]",
        f"{'label':<26}{'turn':<6}{'n_tokens':<10}{'sha256[:16]'}",
    ]
    for r in img_rows:
        L.append(f"{r['label']:<26}{r['turn']:<6}{r['n_image_tokens']:<10}{r['sha256'][:16]}")

    # turn0 identical check
    t0_hashes = {r["sha256"] for r in img_rows if r["turn"] == 0}
    L += [
        "",
        "[A. turn0 cross-branch exact?]",
        f"  distinct turn0 hashes = {len(t0_hashes)} "
        + ("-> ALL IDENTICAL (100% exact-dedup)" if len(t0_hashes) == 1 else "-> NOT all identical"),
        f"  turn0-vs-turn0 pairs: n={len(t0t0)} "
        f"hash_equal={sum(r['hash_equal'] for r in t0t0)}/{len(t0t0)}",
    ]

    if same_branch_cross_turn:
        L += [
            "",
            "[B. same branch, turn0 -> turn1 refocus (partial reuse after local edit)]",
            f"  pairs n={len(same_branch_cross_turn)}",
            f"  changed_pixel_frac mean = {_fmt_pct(_mean(same_branch_cross_turn,'changed_pixel_frac'))}",
            f"  token_mean_cos    mean = {_fmt(_mean(same_branch_cross_turn,'token_mean_cos'))}",
        ]
        for th in thresholds:
            L.append(
                f"  reusable_ratio@{th} mean = {_fmt_pct(_mean(same_branch_cross_turn, f'reusable_ratio@{th}'))}"
            )

    if cross_branch_t1:
        eq = sum(r["hash_equal"] for r in cross_branch_t1)
        L += [
            "",
            "[C/D. cross-branch turn1 refocus: identical (oracle) or just similar?]",
            f"  pairs n={len(cross_branch_t1)}  hash_equal={eq}/{len(cross_branch_t1)} "
            + ("-> byte-identical across branches (oracle deterministic)"
               if eq == len(cross_branch_t1) else
               "-> NOT all identical; see token cos below"),
            f"  token_mean_cos mean = {_fmt(_mean(cross_branch_t1,'token_mean_cos'))}",
        ]
        for th in thresholds:
            L.append(
                f"  reusable_ratio@{th} mean = {_fmt_pct(_mean(cross_branch_t1, f'reusable_ratio@{th}'))}"
            )

    # explicit turn1 cross-branch matrix (token mean-cos), the headline view for
    # "do different branches' refocus images stay cache-similar?"
    if len(t1_labels) >= 2:
        L += [
            "",
            "[turn1 cross-branch refocus — pairwise ViT token mean-cos matrix]",
            "  (rows/cols = each branch's turn1 refocus image; 1.00 = byte-identical)",
        ]
        # per-image hash tag for turn1
        t1_hashes = {lab: named[lab].content_hash()[:8] for lab in t1_labels}
        L.append("  hashes: " + ", ".join(f"{lab.split('_')[0]}={t1_hashes[lab]}" for lab in t1_labels))
        # build matrix
        idx = {lab: i for i, lab in enumerate(t1_labels)}
        n = len(t1_labels)
        mat = [["  -  "] * n for _ in range(n)]
        for i in range(n):
            mat[i][i] = "1.000"
        lut = {(r["a"], r["b"]): r for r in pair_rows}
        for i in range(n):
            for j in range(i + 1, n):
                la, lb = t1_labels[i], t1_labels[j]
                r = lut.get((la, lb)) or lut.get((lb, la))
                v = r.get("token_mean_cos") if r else None
                s = f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "
                mat[i][j] = mat[j][i] = s
        hdr = "        " + "".join(f"{lab.split('_')[0]:>9}" for lab in t1_labels)
        L.append(hdr)
        for i, lab in enumerate(t1_labels):
            L.append(f"  {lab.split('_')[0]:<6}" + "".join(f"{mat[i][j]:>9}" for j in range(n)))
        n_distinct = len(set(named[lab].content_hash() for lab in t1_labels))
        L.append(
            f"  -> {n_distinct} distinct turn1 image(s) across {n} branches"
            + ("  (all branches produced the SAME refocus — oracle-deterministic / exact reuse)"
               if n_distinct == 1 else
               "  (branches diverged — token cos shows cross-branch PARTIAL reuse potential)")
        )

    L += [
        "",
        "files: branch*_turn*.png (images), pairwise_similarity.csv (all pairs), "
        "turn1_cross_branch.csv (turn1-only)",
    ]
    out_path.write_text("\n".join(L), encoding="utf-8")


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else "N/A"


def _fmt_pct(x):
    return f"{x*100:.2f}%" if isinstance(x, float) else "N/A"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--pairwise-csv", default=None, help="pairwise_similarity.csv for auto group selection")
    ap.add_argument("--uids", default=None, help="comma-separated uids (full or 8-char prefix) to force-select")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--per-bucket", type=int, default=1, help="#groups per bucket when auto-selecting")
    ap.add_argument(
        "--select",
        choices=["auto", "cross-branch"],
        default="auto",
        help="auto: consistent/partial/divergent buckets (needs --pairwise-csv); "
        "cross-branch: groups with the most DISTINCT real turn1 refocus images "
        "(the only non-trivial cross-branch-turn1 cases under oracle).",
    )
    ap.add_argument("--top-groups", type=int, default=5, help="#groups for --select cross-branch")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--token-cache-cap", type=int, default=400)
    ap.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS))
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    dump_dir = Path(args.dump_dir)
    out_dir = Path(args.out_dir) if args.out_dir else dump_dir.parent / "similarity" / "case_studies"
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = load_structure(dump_dir)
    print(f"loaded {len(groups)} groups from {dump_dir}")

    # select uids
    selection: dict[str, list[str]] = {}
    if args.uids:
        wanted = [u.strip() for u in args.uids.split(",") if u.strip()]
        resolved = []
        for w in wanted:
            match = w if w in groups else next((u for u in groups if u.startswith(w)), None)
            if match is None:
                print(f"  [warn] uid not found: {w}")
            else:
                resolved.append(match)
        selection = {"manual": resolved}
    elif args.select == "cross-branch":
        ndist = distinct_real_turn1(groups, dump_dir)
        ranked = sorted((u for u in groups if ndist[u] >= 2), key=lambda u: -ndist[u])
        if not ranked:
            raise SystemExit("no group has >=2 distinct real turn1 refocus images")
        selection = {"cross_branch": ranked[: args.top_groups]}
        print("distinct-real-turn1 ranking (top):", [(u[:8], ndist[u]) for u in ranked[: args.top_groups]])
    else:
        if not args.pairwise_csv:
            raise SystemExit("provide --uids or --pairwise-csv for auto selection")
        selection = select_groups_from_csv(Path(args.pairwise_csv), args.per_bucket)

    flat = [(b, u) for b, us in selection.items() for u in us]
    if not flat:
        raise SystemExit("no groups selected")
    print("selected:")
    for b, u in flat:
        print(f"  [{b}] {u}")

    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = TokenCache(args.model, device, args.max_pixels, args.token_cache_cap)

    results = []
    for bucket, uid in flat:
        print(f"\n=== processing [{bucket}] {uid[:8]} ===", flush=True)
        res = process_group(uid, groups[uid], cache, out_dir, thresholds, bucket)
        results.append(res)
        print(f"  -> {res['dir']} ({res['n_images']} images, {res['n_pairs']} pairs)")

    print("\nDONE. Case-study dirs:")
    for r in results:
        print(f"  [{r['bucket']}] {r['dir']}")
    print(f"\nEach dir: branch*_turn*.png + pairwise_similarity.csv + summary.txt")


if __name__ == "__main__":
    main()
