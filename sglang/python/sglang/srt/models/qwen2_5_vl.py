# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/19e6e80e10118f855137b90740936c0b11ac397f/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2-VL model compatible with HuggingFace weights."""
import csv
import logging
import os
import re
import threading
import time
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
)

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.environ import envs
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.utils import RotaryPosMixin, WeightsMapper, permute_inv
from sglang.srt.mem_cache.grpo_similarity_cache import (
    encode_with_grpo_similarity_cache,
    grpo_sim_cache_enabled,
)
from sglang.srt.mem_cache import vlm_cacheblend as _vlm_cacheblend
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda, is_npu

_is_cuda = is_cuda()
_is_npu = is_npu()

logger = logging.getLogger(__name__)


QWEN25_VL_PROFILE_ENABLED = os.environ.get("SGLANG_LOG_INFERENCE_STEP", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
QWEN25_VL_LOG_MERGED_TOKEN_SIM = os.environ.get("SGLANG_GRPO_LOG_MERGED_TOKEN_SIM", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
QWEN25_VL_MERGED_WINDOW_MIN_TOKEN_RATIO = float(
    os.environ.get("SGLANG_GRPO_MERGED_WINDOW_MIN_TOKEN_RATIO", "0.75")
)
QWEN25_VL_PROFILE_CSV_BASE = "vision_encoder_log"
_QWEN25_VL_PROFILE_LOCK = threading.Lock()
_QWEN25_VL_PROFILE_CONTEXT = {
    "global_step": -1,
    "pass_id": -1,
    "mode": "",
    "prefill_tokens": 0,
    "request_ids": [],
    # Maps multimodal item hash -> owning request id, so each encode row can log
    # the single request it belongs to instead of the whole batch's request_ids.
    "item_hash_to_rid": {},
}


def _format_request_ids_for_log(request_ids) -> str:
    if not request_ids:
        return ""
    return "|".join(str(rid) for rid in request_ids if rid is not None and str(rid))


def set_qwen25_vl_profile_context(
    global_step: int,
    pass_id: int,
    mode: str,
    prefill_tokens: int = 0,
    request_ids=None,
    item_hash_to_rid=None,
):
    if not QWEN25_VL_PROFILE_ENABLED and not grpo_sim_cache_enabled():
        return
    with _QWEN25_VL_PROFILE_LOCK:
        _QWEN25_VL_PROFILE_CONTEXT["global_step"] = global_step
        _QWEN25_VL_PROFILE_CONTEXT["pass_id"] = pass_id
        _QWEN25_VL_PROFILE_CONTEXT["mode"] = mode
        _QWEN25_VL_PROFILE_CONTEXT["prefill_tokens"] = int(prefill_tokens or 0)
        if request_ids is None:
            _QWEN25_VL_PROFILE_CONTEXT["request_ids"] = []
        else:
            _QWEN25_VL_PROFILE_CONTEXT["request_ids"] = [
                str(rid) for rid in request_ids if rid is not None and str(rid)
            ]
        _QWEN25_VL_PROFILE_CONTEXT["item_hash_to_rid"] = (
            dict(item_hash_to_rid) if item_hash_to_rid else {}
        )


def _resolve_encode_request_ids(items) -> list:
    """Per-encode request ids: map the items in THIS ViT call back to their
    owning request via the batch's item-hash->rid map. Falls back to the batch
    request_ids context only when the map is unavailable (keeps old behavior)."""
    with _QWEN25_VL_PROFILE_LOCK:
        hash_to_rid = dict(_QWEN25_VL_PROFILE_CONTEXT.get("item_hash_to_rid") or {})
        batch_request_ids = list(_QWEN25_VL_PROFILE_CONTEXT.get("request_ids") or [])
    if hash_to_rid and items:
        resolved: list = []
        seen: set = set()
        for it in items:
            h = getattr(it, "hash", None)
            rid = hash_to_rid.get(h)
            if rid is not None and rid not in seen:
                seen.add(rid)
                resolved.append(rid)
        if resolved:
            return resolved
    return batch_request_ids


def _get_qwen25_vl_profile_csv_path() -> str:
    log_dir = os.environ.get("SGLANG_INFERENCE_LOG_DIR", ".")
    suffix = os.environ.get("SGLANG_INFERENCE_LOG_SUFFIX", "")
    filename = (
        f"{QWEN25_VL_PROFILE_CSV_BASE}_{suffix}.csv"
        if suffix
        else f"{QWEN25_VL_PROFILE_CSV_BASE}.csv"
    )
    return os.path.join(log_dir, filename)


def _sync_qwen25_vl_profile_device():
    if not QWEN25_VL_PROFILE_ENABLED:
        return
    if _is_cuda:
        torch.cuda.synchronize()
    elif _is_npu and hasattr(torch, "npu"):
        torch.npu.synchronize()


def _append_qwen25_vl_profile_log(row: dict):
    if not QWEN25_VL_PROFILE_ENABLED:
        return
    try:
        if get_tensor_model_parallel_rank() != 0:
            return
    except Exception:
        pass
    csv_path = _get_qwen25_vl_profile_csv_path()
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with _QWEN25_VL_PROFILE_LOCK:
        file_exists = os.path.exists(csv_path)
        write_header = not file_exists or os.path.getsize(csv_path) == 0
        if file_exists and not write_header:
            with open(csv_path, "r", newline="") as f:
                first_line = f.readline().strip()
            write_header = first_line.split(",") != list(row.keys())
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def _summarize_qwen25_vl_image_grid(
    image_grid_thw: torch.Tensor,
    patch_size: int,
    spatial_merge_size: int,
) -> Dict[str, Any]:
    """Derive processor grid and approximate processed resolution from image_grid_thw."""
    rows = image_grid_thw.detach().cpu().tolist()
    grid_parts: List[str] = []
    res_parts: List[str] = []
    llm_grid_parts: List[str] = []
    grid_llm_tokens = 0
    for row in rows:
        t, h, w = int(row[0]), int(row[1]), int(row[2])
        llm_h = h // spatial_merge_size
        llm_w = w // spatial_merge_size
        grid_llm_tokens += t * llm_h * llm_w
        grid_parts.append(f"{t}x{h}x{w}")
        llm_grid_parts.append(f"{t}x{llm_h}x{llm_w}")
        res_parts.append(f"{h * patch_size}x{w * patch_size}")
    return {
        "image_grid_thw": "/".join(grid_parts),
        "llm_grid_thw": "/".join(llm_grid_parts),
        "processed_resolution_px": "/".join(res_parts),
        "grid_llm_tokens": grid_llm_tokens,
    }


def _emit_qwen25_vl_image_shape_log(
    *,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    visual,
    image_embed_tokens: int,
    image_count: int,
    cached: bool,
    vision_ms: float,
    request_ids_override=None,
) -> Dict[str, str]:
    patch_size = int(getattr(visual, "patch_size", 14))
    spatial_merge_size = int(getattr(visual, "spatial_merge_size", 2))
    grid_summary = _summarize_qwen25_vl_image_grid(
        image_grid_thw, patch_size, spatial_merge_size
    )
    if request_ids_override is not None:
        request_ids_str = _format_request_ids_for_log(request_ids_override)
    else:
        with _QWEN25_VL_PROFILE_LOCK:
            request_ids_str = _format_request_ids_for_log(
                _QWEN25_VL_PROFILE_CONTEXT.get("request_ids", [])
            )
    shape_info = {
        "pixel_values_shape": "x".join(str(x) for x in pixel_values.shape),
        "num_patches": str(pixel_values.shape[0]),
        "patch_size": str(patch_size),
        "spatial_merge_size": str(spatial_merge_size),
        "image_grid_thw": grid_summary["image_grid_thw"],
        "llm_grid_thw": grid_summary["llm_grid_thw"],
        "processed_resolution_px": grid_summary["processed_resolution_px"],
        "grid_llm_tokens": str(grid_summary["grid_llm_tokens"]),
        "image_embed_tokens": str(image_embed_tokens),
    }
    logger.info(
        "[Qwen2.5-VL image profile] reqs=%s images=%d cached=%d vision_ms=%.2f "
        "pixel_values=%s grid_thw=%s llm_grid=%s processed_px=%s "
        "grid_llm_tokens=%s image_embed_tokens=%d",
        request_ids_str or "-",
        image_count,
        int(cached),
        vision_ms,
        shape_info["pixel_values_shape"],
        shape_info["image_grid_thw"],
        shape_info["llm_grid_thw"],
        shape_info["processed_resolution_px"],
        shape_info["grid_llm_tokens"],
        image_embed_tokens,
    )
    return shape_info


def _record_qwen25_vl_image_profile(
    start_time,
    image_tokens: int,
    image_count: int,
    shape_info: Optional[Dict[str, str]] = None,
    request_ids_override=None,
    grpo_stats=None,
):
    if not QWEN25_VL_PROFILE_ENABLED:
        return
    cached_image_features = int(start_time is None)
    if start_time is not None:
        _sync_qwen25_vl_profile_device()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
    else:
        elapsed_ms = 0.0
    with _QWEN25_VL_PROFILE_LOCK:
        context = dict(_QWEN25_VL_PROFILE_CONTEXT)
    prefill_tokens = int(context.get("prefill_tokens", 0) or 0)
    text_prompt_tokens = max(prefill_tokens - int(image_tokens), 0)
    image_token_ratio = int(image_tokens) / prefill_tokens if prefill_tokens > 0 else 0.0
    request_ids_for_row = (
        request_ids_override
        if request_ids_override is not None
        else context.get("request_ids", [])
    )
    request_ids_str = _format_request_ids_for_log(request_ids_for_row)
    row = {
        "timestamp": f"{time.time():.6f}",
        "pid": os.getpid(),
        "global_step": context.get("global_step", -1),
        "pass_id": context.get("pass_id", -1),
        "mode": context.get("mode", ""),
        "request_ids": request_ids_str,
        "image_count": int(image_count),
        "image_tokens": int(image_tokens),
        "prefill_tokens": prefill_tokens,
        "text_prompt_tokens": text_prompt_tokens,
        "image_token_ratio": f"{image_token_ratio:.4f}",
        "vision_encoder_time_ms": f"{elapsed_ms:.2f}",
        "cached_image_features": cached_image_features,
    }
    if shape_info:
        row.update(shape_info)
    if grpo_stats is not None:
        row.update(grpo_stats.to_log_fields())
    _append_qwen25_vl_profile_log(row)


class Qwen2_5_VLMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        bias: bool = True,
        hidden_act="silu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.tp_rank = 0 if use_data_parallel else get_tensor_model_parallel_rank()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=in_features,
            output_sizes=[hidden_features] * 2,  # [gate_proj, up_proj]
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        self.down_proj = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        self.hidden_act = hidden_act
        if self.hidden_act == "silu":
            self.act = SiluAndMul()
        else:
            base_act = ACT2FN[self.hidden_act]

            def _act_fn(x: torch.Tensor) -> torch.Tensor:
                gate, up = x.chunk(2, dim=-1)
                return base_act(gate) * up

            self.act = _act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act(gate_up)
        x_down, _ = self.down_proj(x)
        return x_down


class Qwen2_5_VisionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int,
        hidden_act="silu",
        norm_layer: Type[nn.Module] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        num_dummy_heads: int = 0,
        rms_norm_eps: float = 1e-6,
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=rms_norm_eps)
        self.norm2 = RMSNorm(dim, eps=rms_norm_eps)

        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,
            proj_bias=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            num_dummy_heads=num_dummy_heads,
            use_data_parallel=use_data_parallel,
        )
        self.mlp = Qwen2_5_VLMLP(
            dim,
            intermediate_dim,
            hidden_act=hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            use_data_parallel=use_data_parallel,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: torch.Tensor,
        output_ws=None,
    ) -> torch.Tensor:
        S, B, H = x.shape
        # norm1: flatten to 2D -> [S*B, H], then reshape back
        x2d = x.reshape(-1, H)
        hidden_states = self.norm1(x2d).reshape(S, B, H)

        # Attention expects [B, S, H]
        hidden_states = rearrange(hidden_states, "s b h -> b s h")
        attn = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            output_ws=output_ws,
        )
        attn = rearrange(attn, "b s h -> s b h")

        # norm2 with fused residual-add: also 2D
        attn2d = attn.reshape(-1, H)
        x_norm_2d, x_after_add_2d = self.norm2(x2d, residual=attn2d)
        x_norm = x_norm_2d.reshape(S, B, H)
        x_after_add = x_after_add_2d.reshape(S, B, H)

        # MLP and final residual
        mlp_out = self.mlp(x_norm)
        x = x_after_add + mlp_out
        return x


class Qwen2_5_VisionPatchMerger(nn.Module):

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = RMSNorm(context_dim, eps=1e-6)
        tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
        tp_rank = 0 if use_data_parallel else get_tensor_model_parallel_rank()
        self.mlp = nn.ModuleList(
            [
                ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp.0", prefix),
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                ),
                nn.GELU(),
                RowParallelLinear(
                    self.hidden_size,
                    dim,
                    bias=True,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp.2", prefix),
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [§4][§5] merger：spatial_merge_unit(=4) 个 patch token → 1 个 merged image token，
        # 再通过 MLP 投到 dim=out_hidden_size（3584，与 LLM hidden 对齐）。
        # x expected shape: [S, B, context_dim]
        S, B, D = x.shape
        x2d = x.reshape(-1, D)
        x2d = self.ln_q(x2d)  # RMSNorm expects 2D
        x2d = x2d.view(-1, self.hidden_size)  # group into spatial_merge_unit
        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x_parallel, _ = mlp_fc1(x2d)
        x_parallel = mlp_act(x_parallel)
        out, _ = mlp_fc2(x_parallel)
        return out


class Qwen2_5_VisionTransformer(nn.Module, RotaryPosMixin):
    """Qwen2.5-VL vision tower.

    End-to-end image encoding pipeline (see also docs/GraftRL_项目全历程.md §4):

        image pixels
          ↓ 1. patch_embed          — patchify; ~4640 patch tokens (1280-d)
          ↓ 2. ViT prefull blocks    — window attention + MLP; layers 0..6
          ↓ 3. ViT fullatt blocks    — global attention; layers 7,15,23,31
          ↓ 4. merger (spatial 4→1)  — ~1160 merged image tokens
          ↓ 5. merger MLP projector  — 1280×4 → 3584 (LLM hidden size)
          ↓ (return to caller)
          ↓ 6. LLM                   — general_mm_embed_routine in Qwen2_5_VLForCausalLM.forward

    GRPO partial reuse hooks mainly at stages 1–3 (patch_hidden similarity,
    skip prefull window/token compute). Stage 4–5 (merged tokens) is where
  offline Part-1 token similarity is measured.
    """

    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        max_context_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        patch_size: int = vision_config.patch_size
        temporal_patch_size: int = vision_config.temporal_patch_size
        spatial_merge_size: int = vision_config.spatial_merge_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_merge_unit: int = spatial_merge_size * spatial_merge_size
        in_channels: int = vision_config.in_channels
        hidden_size: int = vision_config.hidden_size
        depth: int = vision_config.depth
        num_heads: int = vision_config.num_heads
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        mlp_hidden_size: int = ((vision_config.intermediate_size + 7) // 8) * 8
        self.use_data_parallel = use_data_parallel
        self.out_hidden_size = vision_config.out_hidden_size
        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [
                Qwen2_5_VisionBlock(
                    dim=hidden_size,
                    intermediate_dim=mlp_hidden_size,
                    num_heads=num_heads,
                    hidden_act=vision_config.hidden_act,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{i}", prefix),
                    use_data_parallel=use_data_parallel,
                )
                for i in range(depth)
            ]
        )
        self.merger = Qwen2_5_VisionPatchMerger(
            dim=vision_config.out_hidden_size,
            context_dim=hidden_size,
            spatial_merge_size=spatial_merge_size,
            quant_config=quant_config,
            prefix=add_prefix("merger", prefix),
            use_data_parallel=use_data_parallel,
        )

        # Resource prepared for vit cuda graph
        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.max_context_len = max_context_len
        self.enable_cg = _is_cuda and envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get()

        self.cuda_graph_runner: Optional[ViTCudaGraphRunner] = None
        if self.enable_cg:
            self.cuda_graph_runner = ViTCudaGraphRunner(self)

    def get_window_index(self, grid_thw):
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )
        window_index: list = []
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            base = self.rot_pos_ids(h, w, self.spatial_merge_size)
            pos_ids.append(base if t == 1 else base.repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        if self.enable_cg:
            return self.forward_with_cuda_graph(x, grid_thw)

        # --- [§1] patchify / patch_embed ---
        # 把图片切成小块，变成最早期的 patch tokens（典型 ~4640 个，hidden=1280）。
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        # window 重排 + 位置编码（为 §2 window attention 准备 cu_window_seqlens）
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=x.device,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        # Move window_index to the same device as x before using it to index x
        window_index = window_index.to(device=x.device)
        reverse_indices = permute_inv(window_index)

        # Ensure rotary_pos_emb is on the same device/dtype as x
        rotary_pos_emb = rotary_pos_emb.to(device=x.device, dtype=x.dtype)

        seq_len, _ = x.size()

        x = x.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        x = x[window_index, :, :]
        x = x.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        # After building position_embeddings, make sure both cos and sin are on the same device/dtype as the attention input
        position_embeddings = (
            position_embeddings[0].to(x.device, x.dtype),
            position_embeddings[1].to(x.device, x.dtype),
        )

        # compute cu_seqlens - move cu_seqlens to GPU and make it int32
        cu_seqlens = torch.cat(
            [
                torch.tensor([0], device=x.device, dtype=torch.int32),
                (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2])
                .cumsum(dim=0)
                .to(device=x.device, dtype=torch.int32),
            ]
        )
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
        # cu_seqlens must be on cpu because of npu_flash_attention_unpad operator restriction
        if is_npu():
            cu_seqlens = cu_seqlens.to("cpu")
        # --- [§2][§3] ViT blocks (32 layers) ---
        # layer ∉ fullatt_block_indexes → §2 前几层：window attention + MLP（prefull，可 partial reuse）
        # layer ∈ fullatt_block_indexes → §3 full attention：全局 token 交互（layers 7,15,23,31）
        x = x.unsqueeze(1)
        for layer_num, blk in enumerate(self.blocks):
            fullatt_indexes = self.fullatt_block_indexes
            if isinstance(fullatt_indexes, torch.Tensor):
                fullatt_indexes = fullatt_indexes.tolist()
            if layer_num in fullatt_indexes:
                cu_seqlens_now = cu_seqlens  # §3 full attention
            else:
                cu_seqlens_now = cu_window_seqlens  # §2 window attention
            x = blk(
                x, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings
            )

        # --- [§4][§5] merger + projector ---
        # §4: 4 个 patch token 合并成 1 个 merged image token（~1160 个）
        # §5: merger 内 MLP 把维度投到 out_hidden_size=3584，供 LLM 使用
        x = self.merger(x)
        x = x[reverse_indices, :]

        return x

    def _prepare_window_ordered_hidden(
        self, x: torch.Tensor, grid_thw: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """[§1] patch_embed + window 重排，供 partial reuse 路径使用。

        返回的 x 即 patch_hidden（~4640 tokens），是 §2 的输入，也是
        patch-level / window-level 相似度判定的层级。
        """
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)  # [§1]

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=x.device,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        window_index = window_index.to(device=x.device)
        reverse_indices = permute_inv(window_index)
        rotary_pos_emb = rotary_pos_emb.to(device=x.device, dtype=x.dtype)

        seq_len, _ = x.size()
        x = x.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        x = x[window_index, :, :]
        x = x.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (
            emb.cos().to(x.device, x.dtype),
            emb.sin().to(x.device, x.dtype),
        )

        cu_seqlens = torch.cat(
            [
                torch.tensor([0], device=x.device, dtype=torch.int32),
                (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2])
                .cumsum(dim=0)
                .to(device=x.device, dtype=torch.int32),
            ]
        )
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
        if is_npu():
            cu_seqlens = cu_seqlens.to("cpu")
        return x, position_embeddings, cu_seqlens, cu_window_seqlens, window_index, reverse_indices

    def _token_sparse_prefull_layer(
        self,
        blk,
        x: torch.Tensor,
        cu_window_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        reuse_token_mask: torch.Tensor,
        donor_layer_input: torch.Tensor,
        donor_layer_output: torch.Tensor,
    ) -> torch.Tensor:
        """One pre-full-attention window layer with token-level sparse compute.

        Semantics (route B):
          * Attention runs over the full window sequence so changed tokens still
            attend every neighbour. Reused neighbours contribute donor hidden
            states as K/V (filled from ``donor_layer_input``), preserving
            window-attention context instead of dropping it.
          * MLP (FFN) is computed only for changed tokens. Reused tokens take the
            donor's layer output directly (``donor_layer_output``), skipping their
            FFN entirely -- this is where computation is actually saved.

        Changed-token outputs are numerically equal to a full forward when the
        reused tokens truly match the donor (the reuse precondition).
        """
        Sdim, Bdim, Hdim = x.shape
        x_attn_in = x.clone()
        x_attn_in[reuse_token_mask] = donor_layer_input[reuse_token_mask]

        x2d = x_attn_in.reshape(-1, Hdim)
        hidden_states = blk.norm1(x2d).reshape(Sdim, Bdim, Hdim)
        hidden_states = rearrange(hidden_states, "s b h -> b s h")
        attn = blk.attn(
            hidden_states,
            cu_seqlens=cu_window_seqlens,
            position_embeddings=position_embeddings,
        )
        attn = rearrange(attn, "b s h -> s b h")
        attn2d = attn.reshape(-1, Hdim)
        x_norm_2d, x_after_add_2d = blk.norm2(x2d, residual=attn2d)

        out2d = donor_layer_output.reshape(-1, Hdim).clone()
        changed_ids = (~reuse_token_mask).nonzero(as_tuple=True)[0]
        if changed_ids.numel() > 0:
            mlp_in = x_norm_2d[changed_ids].reshape(-1, 1, Hdim)
            mlp_out = blk.mlp(mlp_in).reshape(-1, Hdim)
            out2d[changed_ids] = x_after_add_2d[changed_ids] + mlp_out
        return out2d.reshape(Sdim, Bdim, Hdim)

    def _window_token_indices(
        self, cu_window_seqlens: torch.Tensor, window_ids: torch.Tensor
    ) -> torch.Tensor:
        parts = []
        cu = cu_window_seqlens.detach().to(device=window_ids.device)
        for wid in window_ids.tolist():
            start = int(cu[wid].item())
            end = int(cu[wid + 1].item())
            if end > start:
                parts.append(torch.arange(start, end, device=window_ids.device, dtype=torch.long))
        if not parts:
            return torch.empty(0, device=window_ids.device, dtype=torch.long)
        return torch.cat(parts, dim=0)

    def _subset_cu_window_seqlens(
        self, cu_window_seqlens: torch.Tensor, window_ids: torch.Tensor
    ) -> torch.Tensor:
        lengths = []
        cu = cu_window_seqlens.detach().to(device=window_ids.device)
        for wid in window_ids.tolist():
            lengths.append(int(cu[wid + 1].item() - cu[wid].item()))
        out = [0]
        for length in lengths:
            out.append(out[-1] + max(length, 0))
        return torch.tensor(out, device=window_ids.device, dtype=torch.int32)

    def _raw_window_similarity(
        self,
        target_raw: torch.Tensor,
        donor_raw: torch.Tensor,
        window_index: torch.Tensor,
        cu_window_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = target_raw.shape[0]
        if donor_raw.shape != target_raw.shape:
            raise ValueError("shape_mismatch")
        unit = int(self.spatial_merge_unit)
        if seq_len % unit != 0:
            raise ValueError("seq_len_not_divisible_by_spatial_merge_unit")
        raw_ids = torch.arange(seq_len, device=window_index.device, dtype=torch.long)
        raw_ids = raw_ids.reshape(seq_len // unit, unit)[window_index].reshape(seq_len)
        target_ordered = target_raw.to(device=window_index.device, dtype=torch.float32)[raw_ids]
        donor_ordered = donor_raw.to(device=window_index.device, dtype=torch.float32)[raw_ids]
        row_cos = F.cosine_similarity(target_ordered, donor_ordered, dim=-1, eps=1e-6)
        vals = []
        cu = cu_window_seqlens.detach().to(device=window_index.device)
        for wid in range(max(int(cu.numel()) - 1, 0)):
            start = int(cu[wid].item())
            end = int(cu[wid + 1].item())
            if end <= start:
                vals.append(torch.tensor(-1.0, device=window_index.device))
            else:
                vals.append(row_cos[start:end].mean())
        if not vals:
            return torch.empty(0, device=window_index.device, dtype=torch.float32)
        return torch.stack(vals).to(dtype=torch.float32)

    def forward_with_partial_window_reuse(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        donor_pixel_values: Optional[torch.Tensor] = None,
        donor_partial_cache: Optional[Dict[str, Any]] = None,
        donor_embedding: Optional[torch.Tensor] = None,
        threshold: float = 0.98,
        granularity: str = "window",
        capture_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]], Dict[str, Any]]:
        """Partial ViT reuse prototype.

        Window granularity skips computation for whole similar windows before
        the first full-attention block. Token granularity is a probe path: it
        computes the full target block, then merges donor states for similar
        patch tokens before later layers. Merged granularity computes target
        image embeddings, then merges donor embeddings for similar final image
        tokens. Merged_window uses a patch-embed proxy to score merged-token
        stability, then maps stable merged tokens back to whole ViT windows so
        pre-full-attention window blocks can be skipped.
        """
        raw_input = x
        granularity = (granularity or "window").strip().lower().replace("-", "_")
        stats: Dict[str, Any] = {
            "used": False,
            "granularity": granularity,
            "total_windows": 0,
            "reused_windows": 0,
            "computed_windows": 0,
            "total_window_layers": 0,
            "reused_window_layer_windows": 0,
            "computed_window_layer_windows": 0,
            "window_cosine_min": -1.0,
            "window_cosine_mean": -1.0,
            "window_cosine_max": -1.0,
            "total_tokens": 0,
            "reused_tokens": 0,
            "computed_tokens": 0,
            "total_token_layers": 0,
            "reused_token_layer_tokens": 0,
            "computed_token_layer_tokens": 0,
            "token_cosine_min": -1.0,
            "token_cosine_mean": -1.0,
            "token_cosine_max": -1.0,
            "merged_total_tokens": 0,
            "merged_reused_tokens": 0,
            "merged_token_reuse_ratio": -1.0,
            "merged_token_cosine_min": -1.0,
            "merged_token_cosine_mean": -1.0,
            "merged_token_cosine_max": -1.0,
            "fallback_reason": "",
        }
        if self.enable_cg:
            stats["fallback_reason"] = "vit_cuda_graph_enabled"
            return self.forward(x, grid_thw), None, stats

        try:
            fullatt_indexes = self.fullatt_block_indexes
            if isinstance(fullatt_indexes, torch.Tensor):
                fullatt_indexes = fullatt_indexes.tolist()
            fullatt_indexes = set(int(i) for i in fullatt_indexes)
            first_fullatt = min(fullatt_indexes) if fullatt_indexes else len(self.blocks)
            if first_fullatt <= 0:
                stats["fallback_reason"] = "no_prefull_window_layers"
                return self.forward(x, grid_thw), None, stats

            target_raw = x
            x, position_embeddings, cu_seqlens, cu_window_seqlens, window_index, reverse_indices = (
                self._prepare_window_ordered_hidden(x, grid_thw)  # [§1] → patch_hidden
            )
            total_windows = max(int(cu_window_seqlens.numel()) - 1, 0)
            total_tokens = int(x.shape[0])
            stats["total_windows"] = total_windows
            stats["computed_windows"] = total_windows
            stats["total_window_layers"] = first_fullatt
            stats["computed_window_layer_windows"] = total_windows * first_fullatt
            stats["total_tokens"] = total_tokens
            stats["computed_tokens"] = total_tokens
            stats["total_token_layers"] = first_fullatt
            stats["computed_token_layer_tokens"] = total_tokens * first_fullatt

            cache: Optional[Dict[str, Any]] = None
            if capture_cache:
                cache = {
                    "grid_sig": tuple(int(v) for v in grid_thw.reshape(-1).detach().cpu().tolist()),
                    "window_index": window_index.detach().cpu(),
                    "cu_window_seqlens": cu_window_seqlens.detach().cpu(),
                    "patch_hidden": x.detach(),
                    "prefull_layer_outputs": [],
                    "first_fullatt": int(first_fullatt),
                    "spatial_merge_unit": int(self.spatial_merge_unit),
                }

            reuse_mask = torch.zeros(total_windows, device=x.device, dtype=torch.bool)
            reuse_token_mask = torch.zeros(total_tokens, device=x.device, dtype=torch.bool)
            if donor_partial_cache is None and donor_pixel_values is not None and not capture_cache:
                stats["fallback_reason"] = "missing_donor_partial_cache"
            if donor_partial_cache is not None and donor_pixel_values is not None:
                donor_window_index = donor_partial_cache.get("window_index")
                donor_cu = donor_partial_cache.get("cu_window_seqlens")
                donor_layers = donor_partial_cache.get("prefull_layer_outputs") or []
                if donor_window_index is None or donor_cu is None:
                    raise ValueError("donor_partial_cache_missing_indices")
                if not torch.equal(donor_window_index.to(window_index.device), window_index):
                    raise ValueError("window_index_mismatch")
                if not torch.equal(donor_cu.to(cu_window_seqlens.device), cu_window_seqlens):
                    raise ValueError("cu_window_seqlens_mismatch")
                if int(donor_partial_cache.get("first_fullatt", -1)) != int(first_fullatt):
                    raise ValueError("first_fullatt_mismatch")
                if len(donor_layers) < first_fullatt:
                    raise ValueError("donor_partial_cache_missing_layers")
                if (QWEN25_VL_LOG_MERGED_TOKEN_SIM or granularity == "merged") and donor_embedding is not None:
                    target_merged = self.forward(raw_input, grid_thw)
                    donor_merged = donor_embedding.to(device=target_merged.device, dtype=target_merged.dtype)
                    if donor_merged.shape == target_merged.shape:
                        merged_cos = F.cosine_similarity(
                            target_merged.float(),
                            donor_merged.float(),
                            dim=-1,
                            eps=1e-6,
                        )
                        if merged_cos.numel() > 0:
                            stats["merged_total_tokens"] = int(merged_cos.numel())
                            stats["merged_reused_tokens"] = int((merged_cos >= float(threshold)).sum().item())
                            stats["merged_token_reuse_ratio"] = (
                                stats["merged_reused_tokens"] / stats["merged_total_tokens"]
                            )
                            stats["merged_token_cosine_min"] = float(merged_cos.min().item())
                            stats["merged_token_cosine_mean"] = float(merged_cos.mean().item())
                            stats["merged_token_cosine_max"] = float(merged_cos.max().item())
                            if granularity == "merged":
                                merged_reuse_mask = merged_cos >= float(threshold)
                                if merged_reuse_mask.any():
                                    stats["used"] = True
                                merged_out = target_merged.clone()
                                merged_out[merged_reuse_mask] = donor_merged[merged_reuse_mask]
                                return merged_out, None, stats
                    else:
                        stats["fallback_reason"] = "merged_embedding_shape_mismatch"
                        if granularity == "merged":
                            return target_merged, None, stats
                elif granularity == "merged":
                    stats["fallback_reason"] = "missing_donor_embedding_for_merged"
                    return self.forward(raw_input, grid_thw), None, stats
                donor_patch_hidden = donor_partial_cache.get("patch_hidden")
                if donor_patch_hidden is not None:
                    donor_patch_hidden = donor_patch_hidden.to(device=x.device, dtype=x.dtype)
                    if donor_patch_hidden.shape != x.shape:
                        raise ValueError("donor_patch_hidden_shape_mismatch")
                    patch_cos = F.cosine_similarity(x.float(), donor_patch_hidden.float(), dim=-1, eps=1e-6)
                    if patch_cos.numel() > 0:
                        stats["token_cosine_min"] = float(patch_cos.min().item())
                        stats["token_cosine_mean"] = float(patch_cos.mean().item())
                        stats["token_cosine_max"] = float(patch_cos.max().item())
                    if granularity in ("token", "token_sparse"):
                        reuse_token_mask = patch_cos >= float(threshold)
                        stats["reused_tokens"] = int(reuse_token_mask.sum().item())
                        stats["computed_tokens"] = int(total_tokens - stats["reused_tokens"])
                        if stats["reused_tokens"] > 0:
                            stats["used"] = True
                            stats["reused_token_layer_tokens"] = stats["reused_tokens"] * first_fullatt
                            stats["computed_token_layer_tokens"] = stats["computed_tokens"] * first_fullatt
                        sims = torch.empty(0, device=x.device, dtype=torch.float32)
                    elif granularity == "merged_window":
                        unit = int(self.spatial_merge_unit)
                        if total_tokens % unit != 0:
                            raise ValueError("seq_len_not_divisible_by_spatial_merge_unit")
                        merged_proxy_cos = patch_cos.reshape(total_tokens // unit, unit).mean(dim=1)
                        merged_token_mask = merged_proxy_cos >= float(threshold)
                        stats["merged_total_tokens"] = int(merged_proxy_cos.numel())
                        stats["merged_reused_tokens"] = int(merged_token_mask.sum().item())
                        stats["merged_token_reuse_ratio"] = (
                            stats["merged_reused_tokens"] / stats["merged_total_tokens"]
                            if stats["merged_total_tokens"] > 0
                            else -1.0
                        )
                        if merged_proxy_cos.numel() > 0:
                            stats["merged_token_cosine_min"] = float(merged_proxy_cos.min().item())
                            stats["merged_token_cosine_mean"] = float(merged_proxy_cos.mean().item())
                            stats["merged_token_cosine_max"] = float(merged_proxy_cos.max().item())

                        vals = []
                        window_reuse_vals = []
                        cu = cu_window_seqlens.detach().to(device=x.device)
                        for wid in range(total_windows):
                            start = int(cu[wid].item()) // unit
                            end = int(cu[wid + 1].item()) // unit
                            if end <= start:
                                vals.append(torch.tensor(-1.0, device=x.device))
                                window_reuse_vals.append(torch.tensor(False, device=x.device))
                                continue
                            window_cos = merged_proxy_cos[start:end]
                            window_mask = merged_token_mask[start:end]
                            vals.append(window_cos.mean())
                            window_reuse_vals.append(
                                window_mask.float().mean() >= QWEN25_VL_MERGED_WINDOW_MIN_TOKEN_RATIO
                            )
                        sims = torch.stack(vals).to(dtype=torch.float32) if vals else torch.empty(
                            0, device=x.device, dtype=torch.float32
                        )
                        reuse_mask = torch.stack(window_reuse_vals).to(dtype=torch.bool) if window_reuse_vals else (
                            torch.empty(0, device=x.device, dtype=torch.bool)
                        )
                        stats["reused_windows"] = int(reuse_mask.sum().item())
                        stats["computed_windows"] = int(total_windows - stats["reused_windows"])
                        if stats["reused_windows"] > 0:
                            stats["used"] = True
                            stats["reused_window_layer_windows"] = stats["reused_windows"] * first_fullatt
                            stats["computed_window_layer_windows"] = stats["computed_windows"] * first_fullatt
                    else:
                        vals = []
                        cu = cu_window_seqlens.detach().to(device=x.device)
                        for wid in range(total_windows):
                            start = int(cu[wid].item())
                            end = int(cu[wid + 1].item())
                            vals.append(
                                patch_cos[start:end].mean() if end > start else torch.tensor(-1.0, device=x.device)
                            )
                        sims = torch.stack(vals).to(dtype=torch.float32)
                elif granularity in ("token", "token_sparse"):
                    stats["fallback_reason"] = "missing_donor_patch_hidden_for_token"
                    sims = torch.empty(0, device=x.device, dtype=torch.float32)
                else:
                    sims = self._raw_window_similarity(
                        target_raw,
                        donor_pixel_values,
                        window_index,
                        cu_window_seqlens,
                    )
                if granularity not in ("token", "token_sparse") and sims.numel() != total_windows:
                    raise ValueError("window_similarity_count_mismatch")
                if granularity not in ("token", "token_sparse") and sims.numel() > 0:
                    stats["window_cosine_min"] = float(sims.min().item())
                    stats["window_cosine_mean"] = float(sims.mean().item())
                    stats["window_cosine_max"] = float(sims.max().item())
                    if granularity != "merged_window":
                        reuse_mask = sims >= float(threshold)
                        stats["reused_windows"] = int(reuse_mask.sum().item())
                        stats["computed_windows"] = int(total_windows - stats["reused_windows"])
                        if stats["reused_windows"] > 0:
                            stats["used"] = True
                            stats["reused_window_layer_windows"] = stats["reused_windows"] * first_fullatt
                            stats["computed_window_layer_windows"] = stats["computed_windows"] * first_fullatt

            x = x.unsqueeze(1)
            all_window_ids = torch.arange(total_windows, device=x.device, dtype=torch.long)
            computed_window_ids = all_window_ids[~reuse_mask]
            reused_window_ids = all_window_ids[reuse_mask]

            for layer_num, blk in enumerate(self.blocks):
                if layer_num < first_fullatt:
                    # [§2] prefull window 层（0..first_fullatt-1）：partial reuse 主要在这里
                    if stats["used"] and granularity == "token_sparse":
                        if layer_num == 0:
                            donor_layer_input = (
                                donor_partial_cache["patch_hidden"]
                                .to(device=x.device, dtype=x.dtype)
                                .reshape(x.shape)
                            )
                        else:
                            donor_layer_input = (
                                donor_partial_cache["prefull_layer_outputs"][layer_num - 1]
                                .to(device=x.device, dtype=x.dtype)
                                .reshape(x.shape)
                            )
                        donor_layer_output = (
                            donor_partial_cache["prefull_layer_outputs"][layer_num]
                            .to(device=x.device, dtype=x.dtype)
                            .reshape(x.shape)
                        )
                        x = self._token_sparse_prefull_layer(
                            blk,
                            x,
                            cu_window_seqlens,
                            position_embeddings,
                            reuse_token_mask,
                            donor_layer_input,
                            donor_layer_output,
                        )
                    elif stats["used"] and granularity == "token":
                        next_x = blk(
                            x,
                            cu_seqlens=cu_window_seqlens,
                            position_embeddings=position_embeddings,
                        )
                        if reuse_token_mask.any():
                            donor_layer = donor_partial_cache["prefull_layer_outputs"][layer_num].to(
                                device=x.device, dtype=x.dtype
                            )
                            next_x[reuse_token_mask] = donor_layer[reuse_token_mask]
                        x = next_x
                    elif stats["used"]:
                        next_x = x.clone()
                        if reused_window_ids.numel() > 0:
                            reused_token_ids = self._window_token_indices(cu_window_seqlens, reused_window_ids)
                            donor_layer = donor_partial_cache["prefull_layer_outputs"][layer_num].to(
                                device=x.device, dtype=x.dtype
                            )
                            next_x[reused_token_ids] = donor_layer[reused_token_ids]
                        if computed_window_ids.numel() > 0:
                            computed_token_ids = self._window_token_indices(cu_window_seqlens, computed_window_ids)
                            sub_cu = self._subset_cu_window_seqlens(cu_window_seqlens, computed_window_ids)
                            sub_pos = (
                                position_embeddings[0][computed_token_ids],
                                position_embeddings[1][computed_token_ids],
                            )
                            sub_out = blk(
                                x[computed_token_ids],
                                cu_seqlens=sub_cu,
                                position_embeddings=sub_pos,
                            )
                            next_x[computed_token_ids] = sub_out
                        x = next_x
                    else:
                        x = blk(
                            x,
                            cu_seqlens=cu_window_seqlens,
                            position_embeddings=position_embeddings,
                        )
                    if cache is not None:
                        cache["prefull_layer_outputs"].append(x.detach())
                    continue

                # [§3] full attention 层（fullatt_block_indexes）：必须全量重算，partial 不复用
                cu_seqlens_now = cu_seqlens if layer_num in fullatt_indexes else cu_window_seqlens
                x = blk(
                    x,
                    cu_seqlens=cu_seqlens_now,
                    position_embeddings=position_embeddings,
                )

            # [§4][§5] merger：输出 merged image tokens（~1160×3584）
            # granularity=merged 时在此层做 same-index cosine 与输出替换（ViT 已算完，不能省 §1–§3）
            x = self.merger(x)
            x = x[reverse_indices, :]
            if cache is not None:
                cache["merged_embedding"] = x.detach()
            return x, cache, stats
        except Exception as exc:
            stats["fallback_reason"] = f"partial_exception:{type(exc).__name__}:{str(exc)[:80]}"
            return self.forward(raw_input, grid_thw), None, stats

    def forward_with_cuda_graph(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=x.device,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        window_index = window_index.to(device=x.device)
        reverse_indices = permute_inv(window_index)
        rotary_pos_emb = rotary_pos_emb.to(device=x.device, dtype=x.dtype)

        # patch token num
        seq_len, _ = x.size()

        # [G, M, hidden]
        x = x.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        x = x[window_index, :, :]  # [G, M, hidden]
        x = x.reshape(seq_len, -1)  # [seq_len, hidden]

        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        # After building position_embeddings, make sure both cos and sin are on
        # the same device/dtype as the attention input
        position_embeddings = (
            position_embeddings[0].to(x.device, x.dtype),
            position_embeddings[1].to(x.device, x.dtype),
        )

        # compute cu_seqlens - move cu_seqlens to GPU and make it int32
        cu_seqlens = torch.cat(
            [
                torch.tensor([0], device=x.device, dtype=torch.int32),
                (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2])
                .cumsum(dim=0)
                .to(device=x.device, dtype=torch.int32),
            ]
        )
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])

        return self.cuda_graph_runner.run(
            x=x,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            cu_window_seqlens=cu_window_seqlens,
            output_indices=reverse_indices,
        )


class Qwen2_5_VLForConditionalGeneration(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_up_proj.",
        ".down_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    # To ensure correct weight loading and mapping.
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            "attn.qkv": "attn.qkv_proj",
        },
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            # mapping for original checkpoint
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        },
    )

    def __init__(
        self,
        config: Qwen2_5_VLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.pp_group = get_pp_group()
        self.config = config
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder

        if not self.config.encoder_only:
            self.model = Qwen2Model(
                config,
                quant_config,
                prefix=add_prefix("model", prefix),
            )

            if self.pp_group.is_last_rank:
                if self.pp_group.world_size == 1 and self.config.tie_word_embeddings:
                    self.lm_head = self.model.embed_tokens
                else:
                    self.lm_head = ParallelLMHead(
                        self.config.vocab_size,
                        self.config.hidden_size,
                        quant_config=quant_config,
                        prefix=add_prefix("lm_head", prefix),
                    )
            else:
                # ranks other than the last rank will have a placeholder layer
                self.lm_head = PPMissingLayer()
        else:
            # encoder_only mode: no language model, so no lm_head needed
            self.lm_head = None

        self.visual = Qwen2_5_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            # NOTE: Qwen2_5-VL vision encoder currently supports BitsAndBytes 4-bit quantization.
            # Other quantization methods (e.g., GPTQ, AWQ) are untested and may not be supported.
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
            use_data_parallel=self.use_data_parallel,
            max_context_len=self.config.max_position_embeddings,
        )

        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def _profile_qwen25_vl_image_io(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_embed_tokens: int,
        image_count: int,
        *,
        cached: bool,
        vision_ms: float = 0.0,
        items=None,
    ) -> None:
        if not QWEN25_VL_PROFILE_ENABLED:
            return
        encode_rids = _resolve_encode_request_ids(items)
        shape_info = _emit_qwen25_vl_image_shape_log(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            visual=self.visual,
            image_embed_tokens=image_embed_tokens,
            image_count=image_count,
            cached=cached,
            vision_ms=vision_ms,
            request_ids_override=encode_rids,
        )
        _record_qwen25_vl_image_profile(
            None,
            image_embed_tokens,
            image_count,
            shape_info=shape_info,
            request_ids_override=encode_rids,
        )

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Encode multimodal image items → LLM-ready visual embeddings.

        Runs vision tower stages §1–§5 (``self.visual`` / partial reuse path).
        Returned tensor is consumed at §6 by ``general_mm_embed_routine`` in
        ``Qwen2_5_VLForCausalLM.forward`` (image token slots in the text sequence).
        """
        # in qwen-vl, last dim is the same
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)

        expected_dim = getattr(self.visual, "embed_dim", -1)

        if expected_dim == -1:
            vision_conf = self.config.vision_config
            expected_dim = getattr(
                vision_conf, "embed_dim", getattr(vision_conf, "hidden_size", -1)
            )

        raw_patch_dim = 1176

        if pixel_values.dim() == 2:
            current_dim = pixel_values.shape[-1]
            if current_dim == expected_dim:
                self._profile_qwen25_vl_image_io(
                    pixel_values,
                    image_grid_thw,
                    pixel_values.shape[0],
                    len(items),
                    cached=True,
                    items=items,
                )
                return pixel_values
            if current_dim != raw_patch_dim:
                self._profile_qwen25_vl_image_io(
                    pixel_values,
                    image_grid_thw,
                    pixel_values.shape[0],
                    len(items),
                    cached=True,
                    items=items,
                )
                return pixel_values

        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()

        def _visual_encode(pv: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
            # 完整路径：§1 patch_embed → §2/§3 ViT blocks → §4/§5 merger
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(
                    self.visual, pv, grid_thw.tolist(), rope_type="rope_3d"
                )
            return self.visual(pv, grid_thw=grid_thw)

        def _visual_encode_partial(
            pv: torch.Tensor,
            grid_thw: torch.Tensor,
            *,
            donor_pixel_values: Optional[torch.Tensor] = None,
            donor_partial_cache: Optional[Dict[str, Any]] = None,
            donor_embedding: Optional[torch.Tensor] = None,
            threshold: float = 0.98,
            granularity: str = "window",
            capture_cache: bool = False,
        ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]], Dict[str, Any]]:
            if self.use_data_parallel:
                return (
                    _visual_encode(pv, grid_thw),
                    None,
                    {"used": False, "fallback_reason": "data_parallel_vision"},
                )
            return self.visual.forward_with_partial_window_reuse(
                pv,
                grid_thw,
                donor_pixel_values=donor_pixel_values,
                donor_partial_cache=donor_partial_cache,
                donor_embedding=donor_embedding,
                threshold=threshold,
                granularity=granularity,
                capture_cache=capture_cache,
            )  # §1–§5；partial 主要在 §2 prefull 层省算

        grpo_stats = None
        with _QWEN25_VL_PROFILE_LOCK:
            hash_to_rid = dict(
                _QWEN25_VL_PROFILE_CONTEXT.get("item_hash_to_rid") or {}
            )

        _sync_qwen25_vl_profile_device()
        profile_start = time.perf_counter() if QWEN25_VL_PROFILE_ENABLED else None

        if grpo_sim_cache_enabled() and hash_to_rid:
            # GRPO cache：在 §1–§5 外包一层 donor 查找 / whole-slot skip / partial 调度
            image_embeds, grpo_stats, _ = encode_with_grpo_similarity_cache(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                items=items,
                hash_to_rid=hash_to_rid,
                encode_single_image_fn=_visual_encode,
                encode_partial_image_fn=_visual_encode_partial,
                output_device=self.visual.device,
            )
        else:
            image_embeds = _visual_encode(pixel_values, image_grid_thw)

        if QWEN25_VL_PROFILE_ENABLED:
            _sync_qwen25_vl_profile_device()
            vision_ms = (
                (time.perf_counter() - profile_start) * 1000
                if profile_start is not None
                else 0.0
            )
            encode_rids = _resolve_encode_request_ids(items)
            all_vit_skipped = (
                grpo_stats is not None
                and grpo_stats.vit_calls == 0
                and grpo_stats.vit_skipped > 0
            )
            shape_info = _emit_qwen25_vl_image_shape_log(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                visual=self.visual,
                image_embed_tokens=image_embeds.shape[0],
                image_count=len(items),
                cached=all_vit_skipped,
                vision_ms=vision_ms,
                request_ids_override=encode_rids,
            )
            _record_qwen25_vl_image_profile(
                profile_start if not all_vit_skipped else None,
                image_embeds.shape[0],
                len(items),
                shape_info=shape_info,
                request_ids_override=encode_rids,
                grpo_stats=grpo_stats,
            )
        return image_embeds

    _lora_pattern = re.compile(
        r"^model\.layers\.(\d+)\.(?:self_attn|mlp)\.(?:qkv_proj|o_proj|down_proj|gate_up_proj)$"
    )

    def should_apply_lora(self, module_name: str) -> bool:
        return bool(self._lora_pattern.match(module_name))

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in qwen-vl, last dim is the same
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values, video_grid_thw.tolist(), rope_type="rope_3d"
            )
        else:
            video_embeds = self.visual(pixel_values, grid_thw=video_grid_thw)
        return video_embeds

    def post_process(
        self,
        inputs_embeds,
        modalities: List[Modality],
        embeddings: List[torch.Tensor],
        indices: List[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # Placeholder for post_process
        new_embeddings = []
        for i, (modality, embedding, index) in enumerate(
            zip(modalities, embeddings, indices)
        ):
            if embedding is None or index is None:
                continue

            new_embeddings.append(embedding)
        return new_embeddings, forward_batch

    def get_input_embeddings(self):
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds=None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        """Run forward pass for Qwen2_5-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen2-VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
                (Use input_metadata.mrope_positions to replace it)
        """
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        if not (
            forward_batch.forward_mode.is_decode()
            or not forward_batch.contains_image_inputs()
        ):
            if self.is_mrope_enabled:
                assert positions.ndim == 2 and positions.size(0) == 3, (
                    "multimodal section rotary embedding requires "
                    f"(3, seq_len) positions, but got {positions.size()}"
                )

        # [§6] LLM：把 get_image_feature() 得到的 visual embedding 填进图文序列，与文字 token 一起推理
        if _vlm_cacheblend.cacheblend_enabled():
            _vlm_cacheblend.set_source_input_ids(
                input_ids.detach().clone() if input_ids is not None else None
            )
        try:
            hidden_states = general_mm_embed_routine(
                input_ids=input_ids,
                forward_batch=forward_batch,
                language_model=self.model,
                multimodal_model=self,
                positions=positions,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        finally:
            if _vlm_cacheblend.cacheblend_enabled():
                _vlm_cacheblend.set_source_input_ids(None)

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if (
                self.config.tie_word_embeddings
                and self.pp_group.is_last_rank
                and "model.embed_tokens.weight" in name
            ):
                if "lm_head.weight" in params_dict:
                    lm_head_param = params_dict["lm_head.weight"]
                    weight_loader = getattr(
                        lm_head_param, "weight_loader", default_weight_loader
                    )
                    weight_loader(lm_head_param, loaded_weight)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if (
                    "visual" in name
                    and "up_proj" not in name
                    and "gate_proj" not in name
                ):
                    continue
                name = name.replace(weight_name, param_name)
                layer_id = get_layer_id(name)
                if (
                    layer_id is not None
                    and hasattr(self, "model")
                    and hasattr(self.model, "start_layer")
                    and (
                        layer_id < self.model.start_layer
                        or layer_id >= self.model.end_layer
                    )
                ):
                    continue

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip loading visual/language model weights
                if (
                    self.config.encoder_only or self.config.language_only
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "visual" in name:
                    # adapt to VisionAttention
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")

                try:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name in params_dict.keys():
                        param = params_dict[name]
                    else:
                        continue

                except KeyError:
                    print(params_dict.keys())
                    raise

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        self.capture_aux_hidden_states = True
        self.model.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = [Qwen2_5_VLForConditionalGeneration]
