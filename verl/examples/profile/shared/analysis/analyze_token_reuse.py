#!/usr/bin/env python3
"""Offline ViT image-token reuse analysis for the image-token-cache idea.

Motivation
----------
SGLang's multimodal cache (mem_cache/multimodal_cache.py) reuses an image
embedding ONLY when the raw image bytes hash to the same value (exact match,
full-image reuse). It has no notion of *partial* reuse for images that are
similar-but-not-identical.

The cache idea this script supports is: in multi-turn / multi-step VLM RL
(e.g. Refocus_Chart), step t+1's image is a *local* edit (mask / draw /
highlight on a bbox) of step t's image. Most of the image is untouched, so
most of the post-ViT image tokens should stay (near-)identical and could be
reused instead of recomputed.

The quantity that actually matters is NOT pixel similarity but the similarity
of the ViT *output tokens*. This script measures it directly:

  for each Refocus_Chart row:
    1. load the original chart image
    2. apply the dataset oracle refocus code -> the edited (step t+1) image
    3. run the Qwen2.5-VL vision encoder on both images
    4. per-token cosine(orig, edited); report the fraction of tokens that are
       "reusable" (cosine >= threshold) and a spatial heatmap.

Because orig and edited share the same resolution, the processor produces the
same image_grid_thw, so the two token sequences align 1:1 and per-token
comparison is well defined.

Usage
-----
  CUDA_VISIBLE_DEVICES=0 python3 examples/profile/analyze_token_reuse.py \
    --parquet /data/refocus_chart_multiturn/train.parquet \
    --num-samples 64 \
    --out-dir profile_logs_refocus_chart/token_reuse \
    --dump-heatmaps 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

from examples.profile.shared.agent.vtool_refocus_tools import (
    RefocusCodeParser,
    inject_refocus_bbox_context,
)

DEFAULT_MODEL = os.environ.get("TOKEN_REUSE_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
DEFAULT_THRESHOLDS = (0.999, 0.99, 0.95, 0.90)


# --------------------------------------------------------------------------- #
# data helpers
# --------------------------------------------------------------------------- #
def _bytes_to_pil(item) -> Image.Image | None:
    if isinstance(item, dict) and item.get("bytes"):
        return Image.open(BytesIO(item["bytes"])).convert("RGB")
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    return None


def _parse_metadata(metadata) -> dict:
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str) and metadata:
        try:
            return json.loads(metadata)
        except json.JSONDecodeError:
            return {}
    return {}


def apply_refocus(
    image: Image.Image,
    oracle_code: str,
    metadata: dict,
    parser: RefocusCodeParser,
) -> Image.Image | None:
    """Run the dataset oracle refocus code (focus_on_*) and return the edited image."""
    captured: dict[str, Image.Image] = {}

    def display(im):
        captured["img"] = im

    context = parser.get_tool_context(display)
    inject_refocus_bbox_context(context, metadata)
    context["display"] = display
    # draw/mask tools mutate the image in place and return it; pass a copy so the
    # caller's original stays pristine for the orig-vs-edited comparison.
    context["image_1"] = image.copy()
    code = parser.ensure_display_call(oracle_code)
    try:
        exec(code, context)  # noqa: S102 - trusted dataset oracle code
    except Exception as exc:  # noqa: BLE001
        print(f"  [refocus exec failed] {type(exc).__name__}: {exc}")
        return None
    out = captured.get("img")
    if isinstance(out, Image.Image) and out.size[0] > 0:
        return out.convert("RGB")
    return None


def changed_pixel_fraction(orig: Image.Image, edited: Image.Image) -> float:
    """Fraction of pixels that differ at all between orig and edited (input-side change)."""
    a = np.asarray(orig, dtype=np.int16)
    b = np.asarray(edited.resize(orig.size), dtype=np.int16)
    diff = np.abs(a - b).sum(axis=-1) > 0
    return float(diff.mean())


# --------------------------------------------------------------------------- #
# vision encoder
# --------------------------------------------------------------------------- #
class VisionEncoder:
    def __init__(self, model_path: str, device: str, max_pixels: int | None):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = device
        proc_kwargs = {}
        if max_pixels:
            proc_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_path, **proc_kwargs)
        self.image_processor = self.processor.image_processor
        print(f"Loading vision tower from {model_path} ...", flush=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.visual = model.visual.to(device).eval()
        # LM weights are not needed for vision-token analysis; free them.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.merge_size = getattr(self.image_processor, "merge_size", 2)

    @torch.no_grad()
    def encode(self, image: Image.Image):
        """Return (tokens[seq, hidden] float32 cpu, grid_thw tuple)."""
        feats = self.image_processor(images=[image], return_tensors="pt")
        pixel_values = feats["pixel_values"].to(self.device, dtype=torch.bfloat16)
        grid_thw = feats["image_grid_thw"].to(self.device)
        embeds = self.visual(pixel_values, grid_thw=grid_thw)
        if isinstance(embeds, (tuple, list)):
            embeds = embeds[0]
        grid = tuple(int(x) for x in grid_thw[0].tolist())
        return embeds.float().cpu(), grid


# --------------------------------------------------------------------------- #
# heatmap
# --------------------------------------------------------------------------- #
def save_heatmap(
    orig: Image.Image,
    edited: Image.Image,
    cos: torch.Tensor,
    grid_thw: tuple[int, int, int],
    merge_size: int,
    out_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return

    t, h, w = grid_thw
    hh, ww = h // merge_size, w // merge_size
    if hh * ww != cos.numel():
        return
    sim = cos.reshape(hh, ww).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig)
    axes[0].set_title("step t (original)")
    axes[0].axis("off")
    axes[1].imshow(edited)
    axes[1].set_title("step t+1 (refocus)")
    axes[1].axis("off")
    im = axes[2].imshow(sim, cmap="RdYlGn", vmin=0.8, vmax=1.0)
    axes[2].set_title("per-token cosine(t, t+1)\ngreen=reusable, red=recompute")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", default="/data/refocus_chart_multiturn/train.parquet")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--out-dir", default="profile_logs_refocus_chart/token_reuse")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-pixels", type=int, default=None, help="cap processor max_pixels (smaller=faster)")
    parser.add_argument("--dump-heatmaps", type=int, default=8, help="save N side-by-side heatmaps")
    parser.add_argument(
        "--thresholds",
        default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
        help="comma-separated cosine thresholds for reusable-token fraction",
    )
    args = parser.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = out_dir / "heatmaps"
    if args.dump_heatmaps:
        heatmap_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(args.parquet)
    rows = table.to_pylist()
    code_parser = RefocusCodeParser()
    encoder = VisionEncoder(args.model, args.device, args.max_pixels)

    per_sample: list[dict] = []
    heatmaps_saved = 0
    processed = 0

    for ridx, row in enumerate(rows):
        if processed >= args.num_samples:
            break
        ei = row.get("extra_info") or {}
        oracle = ei.get("oracle_refocus_code")
        if not oracle:
            continue
        images = row.get("images") or []
        if not images:
            continue
        orig = _bytes_to_pil(images[0])
        if orig is None:
            continue
        metadata = _parse_metadata((ei.get("tools_kwargs") or {}).get("metadata"))
        edited = apply_refocus(orig, oracle, metadata, code_parser)
        if edited is None:
            continue
        if edited.size != orig.size:
            edited = edited.resize(orig.size)

        try:
            tok_o, grid_o = encoder.encode(orig)
            tok_e, grid_e = encoder.encode(edited)
        except Exception as exc:  # noqa: BLE001
            print(f"  [encode failed row {ridx}] {type(exc).__name__}: {exc}")
            continue
        if grid_o != grid_e or tok_o.shape != tok_e.shape:
            print(f"  [grid mismatch row {ridx}] {grid_o} vs {grid_e}; skip")
            continue

        cos = F.cosine_similarity(tok_o, tok_e, dim=-1)  # [seq]
        n_tokens = cos.numel()
        rec = {
            "row_index": ridx,
            "source": metadata.get("source", ""),
            "n_image_tokens": int(n_tokens),
            "mean_cos": float(cos.mean()),
            "median_cos": float(cos.median()),
            "min_cos": float(cos.min()),
            "changed_pixel_frac": round(changed_pixel_fraction(orig, edited), 4),
        }
        for th in thresholds:
            rec[f"reusable_frac@{th}"] = float((cos >= th).float().mean())
        per_sample.append(rec)
        processed += 1

        if args.dump_heatmaps and heatmaps_saved < args.dump_heatmaps:
            save_heatmap(
                orig, edited, cos, grid_o, encoder.merge_size,
                heatmap_dir / f"reuse_{ridx:05d}.png",
            )
            heatmaps_saved += 1

        if processed % 8 == 0:
            print(f"  processed {processed}/{args.num_samples} ...", flush=True)

    if not per_sample:
        raise SystemExit("No samples produced; check oracle code / parquet path.")

    # write per-sample csv
    csv_path = out_dir / "token_reuse_per_sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_sample[0].keys()))
        w.writeheader()
        w.writerows(per_sample)

    # aggregate
    def agg(key: str) -> float:
        return float(np.mean([r[key] for r in per_sample]))

    summary = {
        "n_samples": len(per_sample),
        "mean_cos": agg("mean_cos"),
        "mean_changed_pixel_frac": agg("changed_pixel_frac"),
    }
    for th in thresholds:
        summary[f"mean_reusable_frac@{th}"] = agg(f"reusable_frac@{th}")

    # markdown report
    report = out_dir / "token_reuse_report.md"
    lines = [
        "# ViT image-token reuse analysis (Refocus_Chart, step t -> t+1)",
        "",
        f"- model: `{args.model}`",
        f"- parquet: `{args.parquet}`",
        f"- samples (rows with oracle refocus + valid edit): **{summary['n_samples']}**",
        "",
        "## 这测的是什么",
        "",
        "对每条样本：原图(step t) 与 oracle refocus 后的图(step t+1) 分别过 Qwen2.5-VL ",
        "vision encoder，逐 image token 算 cosine。`reusable_frac@th` = cosine ≥ th 的 token 占比，",
        "即「step t 的 image token cache 中可被 step t+1 直接复用的比例」。",
        "对比 `changed_pixel_frac`（输入像素改了多少）可看出 ViT 把局部像素改动传播到多少 token。",
        "",
        "## 汇总",
        "",
        "| 指标 | 值 |",
        "| --- | ---: |",
        f"| 样本数 | {summary['n_samples']} |",
        f"| 平均 per-token cosine | {summary['mean_cos']:.4f} |",
        f"| 平均改动像素占比 | {summary['mean_changed_pixel_frac']*100:.2f}% |",
    ]
    for th in thresholds:
        lines.append(f"| 平均可复用 token 占比 (cosine ≥ {th}) | {summary[f'mean_reusable_frac@{th}']*100:.2f}% |")
    lines += [
        "",
        "## 解读提示",
        "",
        "- 若「可复用 token 占比」明显高于「1 − 改动像素占比」相近或更高，说明 ViT 的局部性较好，",
        "  局部改图后大部分 token 几乎不变 → **部分 cache 复用可行**。",
        "- 若可复用占比远低于改动像素占比的预期，说明 ViT 全局注意力把改动扩散到了全图 token，",
        "  → 朴素的「按 patch 复用」收益有限，需要更细的方案。",
        "",
        f"逐样本明细：`{csv_path.name}`；热力图：`heatmaps/`",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nWrote:\n  {csv_path}\n  {report}")
    if args.dump_heatmaps:
        print(f"  {heatmap_dir}/ ({heatmaps_saved} heatmaps)")


if __name__ == "__main__":
    main()
