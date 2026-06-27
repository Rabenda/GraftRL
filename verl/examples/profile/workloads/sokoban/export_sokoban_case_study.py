#!/usr/bin/env python3
"""Export Sokoban rollout images for visual inspection (Refocus case-study style).

Layout per group:
  group_<idx>/
    branch0_step00_obs.png
    branch1_step00_obs.png   (same bytes as branch0 at step0 — same env seed)
    branch0_step01_obs.png
    ...
    contact_sheet.png        grid: rows=step, cols=branch
    summary.txt              step/branch coverage table

Usage:
  python3 examples/profile/sokoban/export_sokoban_case_study.py \\
    --dump-dir profile_logs_sokoban/image_dump_sokoban_bs64_n4 \\
    --out-dir profile_logs_sokoban/case_studies \\
    --groups 4,16,49 --max-step 8

  # auto-pick groups with 4 branches and longest trajectories:
  python3 examples/profile/sokoban/export_sokoban_case_study.py \\
    --dump-dir profile_logs_sokoban/image_dump_sokoban_bs64_n4 \\
    --out-dir profile_logs_sokoban/case_studies \\
    --auto-pick 3 --max-step 10
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_manifest(dump_dir: Path) -> dict[int, dict[int, dict[int, Path]]]:
    """group_idx -> branch_idx -> step -> image path."""
    manifest = dump_dir / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    groups: dict[int, dict[int, dict[int, Path]]] = defaultdict(lambda: defaultdict(dict))
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        p = dump_dir / r["path"]
        if p.is_file():
            groups[int(r["group_idx"])][int(r["branch_idx"])][int(r["step"])] = p
    return groups


def auto_pick_groups(groups: dict, n: int) -> list[int]:
    scored: list[tuple[int, int, int]] = []
    for g, branches in groups.items():
        n_br = len(branches)
        if n_br < 4:
            continue
        if not all(0 in branches[b] for b in range(4)):
            continue
        max_step = max(max(steps) for steps in branches.values())
        scored.append((max_step, n_br, g))
    scored.sort(reverse=True)
    return [g for _, _, g in scored[:n]]


def export_group(
    *,
    group_idx: int,
    branches: dict[int, dict[int, Path]],
    out_dir: Path,
    max_step: int,
    tile_size: int,
    make_sheet: bool,
) -> None:
    gdir = out_dir / f"group_{group_idx:04d}"
    gdir.mkdir(parents=True, exist_ok=True)

    branch_ids = sorted(b for b in branches if any(s <= max_step for s in branches[b]))
    if not branch_ids:
        return
    steps = sorted({s for b in branch_ids for s in branches[b] if s <= max_step})
    if not steps:
        return

    lines = [
        f"group_idx={group_idx}",
        f"branches={branch_ids}",
        f"steps_exported={steps}",
        "",
        "step  branch  path  same_as_step0_branch0",
    ]

    step0_ref = branches.get(branch_ids[0], {}).get(0)
    step0_bytes = step0_ref.read_bytes() if step0_ref else None

    for step in steps:
        for b in branch_ids:
            src = branches[b].get(step)
            if src is None:
                continue
            dst_name = f"branch{b}_step{step:02d}_obs.png"
            dst = gdir / dst_name
            shutil.copy2(src, dst)
            same = ""
            if step == 0 and step0_bytes is not None:
                same = str(dst.read_bytes() == step0_bytes)
            elif step > 0:
                same = "-"
            lines.append(f"{step:4d}  {b:6d}  {dst_name}  {same}")

    lines.extend(["", "Notes:", "- step0 across branches should be byte-identical (same env seed).", "- step1+ diverges as branches take different actions.", ""])

    # Optional: one shared step0 image (all branches identical)
    if 0 in steps and step0_ref is not None:
        shutil.copy2(step0_ref, gdir / "shared_step00_obs.png")

    if make_sheet:
        sheet = _build_contact_sheet(tiles_data=branches, branch_ids=branch_ids, steps=steps, tile_size=tile_size)
        sheet.save(gdir / "contact_sheet.png")

    (gdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_contact_sheet(
    *,
    tiles_data: dict[int, dict[int, Path]],
    branch_ids: list[int],
    steps: list[int],
    tile_size: int,
) -> Image.Image:
    n_cols = len(branch_ids)
    n_rows = len(steps)
    pad = 4
    label_h = 28
    header_w = 72
    cell = tile_size + pad
    w = header_w + n_cols * cell + pad
    h = label_h + n_rows * cell + pad
    canvas = Image.new("RGB", (w, h), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for j, b in enumerate(branch_ids):
        x = header_w + j * cell + pad // 2
        draw.text((x, 6), f"branch{b}", fill=(0, 0, 0), font=font)

    for i, step in enumerate(steps):
        y = label_h + i * cell + pad // 2
        draw.text((8, y + tile_size // 2 - 6), f"step{step:02d}", fill=(0, 0, 0), font=font)
        for j, b in enumerate(branch_ids):
            src = tiles_data[b].get(step)
            if src is None:
                continue
            img = Image.open(src).convert("RGB")
            img = img.resize((tile_size, tile_size), Image.Resampling.NEAREST)
            x = header_w + j * cell + pad // 2
            canvas.paste(img, (x, label_h + i * cell + pad // 2))

    return canvas


def write_index_html(out_dir: Path, group_dirs: list[Path]) -> None:
    import base64

    lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Sokoban case studies</title>",
        "<style>",
        "body{font-family:sans-serif;margin:24px;background:#111;color:#eee}",
        ".group{margin-bottom:48px}",
        "img{image-rendering:pixelated;max-width:100%;border:1px solid #444}",
        "h2{color:#8cf}",
        "a{color:#8cf}",
        "code{color:#ccc}",
        "</style></head><body>",
        "<h1>Sokoban rollout — group × step × branch</h1>",
        "<p>Each row = env step; each column = GRPO branch (n=4). step0 should be identical across branches.</p>",
        f"<p>PNG files: <code>{out_dir}</code></p>",
    ]
    for gdir in sorted(group_dirs):
        sheet = gdir / "contact_sheet.png"
        if not sheet.is_file():
            continue
        gid = gdir.name
        b64 = base64.b64encode(sheet.read_bytes()).decode("ascii")
        lines.append(f"<div class='group'><h2>{gid}</h2>")
        lines.append(f"<p><a href='{gid}/summary.txt'>summary.txt</a> · ")
        lines.append(f"<a href='{gid}/contact_sheet.png'>contact_sheet.png</a></p>")
        lines.append(f"<img src='data:image/png;base64,{b64}' alt='{gid}'></div>")
    lines.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Sokoban image dumps for visual case studies")
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--groups", default="", help="comma-separated group_idx, e.g. 4,16,49")
    ap.add_argument("--auto-pick", type=int, default=0, help="pick N groups with 4 branches and longest steps")
    ap.add_argument("--max-step", type=int, default=10, help="max step to export (inclusive)")
    ap.add_argument("--tile-size", type=int, default=96, help="contact sheet cell size in pixels")
    ap.add_argument("--no-contact-sheet", action="store_true")
    args = ap.parse_args()

    dump_dir = args.dump_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = load_manifest(dump_dir)
    if args.groups.strip():
        pick = [int(x.strip()) for x in args.groups.split(",") if x.strip()]
    elif args.auto_pick > 0:
        pick = auto_pick_groups(groups, args.auto_pick)
    else:
        pick = auto_pick_groups(groups, 3)

    if not pick:
        raise SystemExit("no groups selected; use --groups or --auto-pick")

    exported: list[Path] = []
    for g in pick:
        if g not in groups:
            print(f"skip missing group {g}")
            continue
        export_group(
            group_idx=g,
            branches=groups[g],
            out_dir=out_dir,
            max_step=args.max_step,
            tile_size=args.tile_size,
            make_sheet=not args.no_contact_sheet,
        )
        exported.append(out_dir / f"group_{g:04d}")
        print(f"exported group_{g:04d} -> {out_dir / f'group_{g:04d}'}")

    if not args.no_contact_sheet:
        write_index_html(out_dir, exported)
        print(f"index: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
