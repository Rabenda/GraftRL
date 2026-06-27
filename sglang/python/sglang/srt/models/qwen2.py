# Copyright 2023-2024 SGLang Team
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
# ==============================================================================

# Adapted from llama2.py
# Modify details for the adaptation of Qwen2 model.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, make_layers

# VLM-CacheBlend (§6 LLM-prefill visual-KV reuse). Import is always safe; all behaviour
# is gated behind the SGLANG_VLM_CACHEBLEND macro (default off) and a per-request
# context, so the baseline path is unchanged when disabled.
# Design: verl_vision/examples/profile/shared/docs/VLM_CACHEBLEND_DESIGN.md
from sglang.srt.mem_cache import vlm_cacheblend as _vlm_cacheblend

Qwen2Config = None


logger = logging.getLogger(__name__)


class Qwen2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        if get_global_server_args().rl_on_policy_target is not None:
            x = x.bfloat16()

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        reuse_idx = self._cacheblend_reuse_projection_skip_indices(
            hidden_states, forward_batch
        )
        active_idx = None
        if reuse_idx.numel() > 0 and reuse_idx.numel() < hidden_states.shape[0]:
            active_mask = torch.ones(
                hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device
            )
            active_mask[reuse_idx] = False
            active_idx = active_mask.nonzero(as_tuple=False).reshape(-1)
            qkv = hidden_states.new_zeros(
                (hidden_states.shape[0], self.q_size + 2 * self.kv_size)
            )
            qkv_active, _ = self.qkv_proj(hidden_states.index_select(0, active_idx))
            qkv[active_idx] = qkv_active
        elif reuse_idx.numel() >= hidden_states.shape[0]:
            qkv = hidden_states.new_zeros(
                (hidden_states.shape[0], self.q_size + 2 * self.kv_size)
            )
        else:
            qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        if _vlm_cacheblend.cacheblend_enabled():
            object.__setattr__(
                self.attn, "_vlm_cacheblend_rotary_emb", self.rotary_emb
            )
        attn_output = self.attn(q, k, v, forward_batch)
        if (
            _vlm_cacheblend.cacheblend_enabled()
            and _vlm_cacheblend.get_config().unsafe_post_attention_overlay
        ):
            _vlm_cacheblend.apply_recipient_kv_blend_for_layer(
                forward_batch=forward_batch,
                layer_id=self.attn.layer_id,
                rotary_emb=self.rotary_emb,
            )
        if active_idx is not None:
            output = hidden_states.new_zeros(hidden_states.shape)
            projected_active, _ = self.o_proj(attn_output.index_select(0, active_idx))
            output[active_idx] = projected_active
        elif reuse_idx.numel() >= hidden_states.shape[0]:
            output = hidden_states.new_zeros(hidden_states.shape)
        else:
            output, _ = self.o_proj(attn_output)
        return output

    @staticmethod
    def _cacheblend_reuse_projection_skip_indices(
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if not _vlm_cacheblend.cacheblend_enabled():
            return torch.empty(0, dtype=torch.long, device=hidden_states.device)
        cfg = _vlm_cacheblend.get_config()
        if not cfg.fast_path or not cfg.skip_reuse_qkv_proj:
            return torch.empty(0, dtype=torch.long, device=hidden_states.device)
        return _vlm_cacheblend.recipient_reuse_token_indices(
            getattr(forward_batch, "out_cache_loc", None),
            device=hidden_states.device,
        )


class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            dual_chunk_attention_config=dual_chunk_attention_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        reuse_idx = self._cacheblend_reuse_mlp_skip_indices(hidden_states, forward_batch)
        if reuse_idx.numel() > 0 and reuse_idx.numel() < hidden_states.shape[0]:
            active_mask = torch.ones(
                hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device
            )
            active_mask[reuse_idx] = False
            active_idx = active_mask.nonzero(as_tuple=False).reshape(-1)
            mlp_out = torch.zeros_like(hidden_states)
            mlp_out[active_idx] = self.mlp(hidden_states.index_select(0, active_idx))
            hidden_states = mlp_out
        elif reuse_idx.numel() >= hidden_states.shape[0]:
            hidden_states = torch.zeros_like(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    @staticmethod
    def _cacheblend_reuse_mlp_skip_indices(
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if not _vlm_cacheblend.cacheblend_enabled():
            return torch.empty(0, dtype=torch.long, device=hidden_states.device)
        cfg = _vlm_cacheblend.get_config()
        if not cfg.fast_path or not cfg.skip_reuse_mlp:
            return torch.empty(0, dtype=torch.long, device=hidden_states.device)
        return _vlm_cacheblend.recipient_reuse_token_indices(
            getattr(forward_batch, "out_cache_loc", None),
            device=hidden_states.device,
        )


class Qwen2Model(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Qwen2DecoderLayer,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                enable_tp=not is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
                params_dtype=(
                    torch.float32
                    if get_global_server_args().rl_on_policy_target is not None
                    else None
                ),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2DecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen2DecoderLayer
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            norm_kwargs = (
                dict(
                    weight_dtype=torch.float32,
                    cast_x_before_out_mul=True,
                    override_orig_dtype=torch.float32,
                    fp32_residual=True,
                )
                if get_global_server_args().rl_on_policy_target is not None
                else {}
            )
            self.norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
            )
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        self.layers_to_capture = []

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.config, "scale_emb"):
            return self.get_input_embeddings()(input_ids) * self.config.scale_emb
        else:
            return self.get_input_embeddings()(input_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:

        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        self._cacheblend_prepare_recipient_fast_path(input_ids, forward_batch)

        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            if i in self.layers_to_capture:
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )

        # [VLM-CacheBlend §6] donor capture / recipient eligibility probe. The
        # recipient fast path is intentionally not enabled here yet; until the
        # selective layer loop is implemented, recipient requests run full prefill and
        # emit a structured fallback reason.
        self._maybe_cacheblend_after_full_prefill(input_ids, forward_batch)
        if _vlm_cacheblend.cacheblend_enabled():
            _vlm_cacheblend.clear_recipient_blend_plans()

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states

    def _cacheblend_prepare_recipient_fast_path(
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        if not _vlm_cacheblend.cacheblend_enabled():
            return
        _vlm_cacheblend.clear_recipient_blend_plans()
        ctx = _vlm_cacheblend.get_request_context()
        if ctx is None or not forward_batch.forward_mode.is_extend():
            return
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if out_cache_loc is None:
            return
        if input_ids is None:
            input_ids = _vlm_cacheblend.get_source_input_ids()
        if input_ids is None:
            return
        ctxs = tuple(getattr(ctx, "contexts", (ctx,)))
        req_slices = self._cacheblend_request_slices(forward_batch, input_ids)
        positions = getattr(forward_batch, "mrope_positions", None)
        plans = []
        for one_ctx in ctxs:
            if getattr(one_ctx, "role", "") != "recipient":
                continue
            req_index = int(getattr(one_ctx, "request_index", 0))
            if req_index < 0 or req_index >= len(req_slices):
                continue
            start, end = req_slices[req_index]
            img_locs, img_positions, _ = self._cacheblend_locate_image_tokens(
                one_ctx,
                input_ids[start:end],
                out_cache_loc[start:end],
                forward_batch,
                self._cacheblend_positions_slice(positions, start, end),
            )
            if img_locs.numel() == 0:
                continue
            plan, _ = _vlm_cacheblend.build_recipient_kv_blend_plan(
                one_ctx, img_locs, img_positions
            )
            if plan is not None:
                plans.append(plan)
        if plans:
            _vlm_cacheblend.set_recipient_blend_plans(tuple(plans))

    def _maybe_cacheblend_after_full_prefill(
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        """Snapshot donor K/V or log recipient eligibility after full prefill.

        Recipient blending, when enabled, is applied inside the attention backend before
        prefill attention consumes the KV cache. This post-prefill hook only captures
        donors and emits structured stats.
        """
        if not _vlm_cacheblend.cacheblend_enabled():
            return
        ctx = _vlm_cacheblend.get_request_context()
        if ctx is None:
            return
        if not forward_batch.forward_mode.is_extend():
            return
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        ctxs = tuple(getattr(ctx, "contexts", (ctx,)))
        if out_cache_loc is None:
            ctx0 = ctxs[0] if ctxs else ctx
            _vlm_cacheblend.log_stats(
                _vlm_cacheblend.CacheBlendStats(
                    role=getattr(ctx0, "role", "none"),
                    request_id=getattr(ctx0, "request_id", ""),
                    fallback_reason="missing_out_cache_loc",
                ).finalize()
            )
            return
        if input_ids is None:
            input_ids = _vlm_cacheblend.get_source_input_ids()
        if input_ids is None:
            ctx0 = ctxs[0] if ctxs else ctx
            _vlm_cacheblend.log_stats(
                _vlm_cacheblend.CacheBlendStats(
                    role=getattr(ctx0, "role", "none"),
                    request_id=getattr(ctx0, "request_id", ""),
                    fallback_reason="missing_input_ids",
                ).finalize()
            )
            return
        req_slices = self._cacheblend_request_slices(forward_batch, input_ids)
        stats_list = []
        for one_ctx in ctxs:
            req_index = int(getattr(one_ctx, "request_index", 0))
            if req_index < 0 or req_index >= len(req_slices):
                stats_list.append(
                    _vlm_cacheblend.CacheBlendStats(
                        role=getattr(one_ctx, "role", "none"),
                        request_id=getattr(one_ctx, "request_id", ""),
                        fallback_reason=f"request_slice_oob:{req_index}/{len(req_slices)}",
                    ).finalize()
                )
                continue
            start, end = req_slices[req_index]
            stats_list.append(
                self._cacheblend_handle_context(
                    one_ctx,
                    input_ids[start:end],
                    out_cache_loc[start:end],
                    forward_batch,
                    layer_ids=list(range(self.start_layer, self.end_layer)),
                    positions_slice=self._cacheblend_positions_slice(
                        getattr(forward_batch, "mrope_positions", None), start, end
                    ),
                )
            )
        _vlm_cacheblend.log_stats(self._cacheblend_aggregate_stats(stats_list))

    @staticmethod
    def _cacheblend_request_slices(forward_batch: ForwardBatch, input_ids: torch.Tensor):
        lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if lens is None:
            lens = getattr(forward_batch, "extend_seq_lens", None)
            if isinstance(lens, torch.Tensor):
                lens = lens.detach().cpu().tolist()
        if not lens:
            lens = [int(input_ids.numel())]
        slices = []
        start = 0
        total = int(input_ids.numel())
        for length in lens:
            end = min(total, start + int(length))
            slices.append((start, end))
            start = end
        return slices

    @staticmethod
    def _cacheblend_positions_slice(positions, start: int, end: int):
        if positions is None or not isinstance(positions, torch.Tensor):
            return None
        if positions.dim() == 2:
            return positions[:, start:end]
        return None

    @staticmethod
    def _cacheblend_locate_image_tokens(
        ctx,
        input_ids: torch.Tensor,
        out_cache_loc: torch.Tensor,
        forward_batch: ForwardBatch,
        positions_slice,
    ):
        img_locs = _vlm_cacheblend.image_token_locs(
            input_ids,
            out_cache_loc,
            ctx.image_token_id,
            target_image_slot=ctx.target_image_slot,
            image_token_values=ctx.image_token_values,
        )
        img_positions = None
        if (
            img_locs.numel() != 0
            and positions_slice is not None
            and positions_slice.dim() == 2
        ):
            img_mask = _vlm_cacheblend.image_token_mask_for_slot(
                input_ids,
                ctx.image_token_id,
                target_image_slot=ctx.target_image_slot,
                image_token_values=ctx.image_token_values,
            )
            img_positions = positions_slice[:, img_mask]
        if img_locs.numel() == 0:
            img_locs, img_positions, request_locs_reason = (
                _vlm_cacheblend.image_token_locs_from_request(forward_batch, ctx)
            )
        else:
            request_locs_reason = ""
        return img_locs, img_positions, request_locs_reason

    def _cacheblend_handle_context(
        self,
        ctx,
        input_ids: torch.Tensor,
        out_cache_loc: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        layer_ids,
        positions_slice,
    ):
        try:
            img_locs, img_positions, request_locs_reason = (
                self._cacheblend_locate_image_tokens(
                    ctx,
                    input_ids,
                    out_cache_loc,
                    forward_batch,
                    positions_slice,
                )
            )
            try:
                _vlm_cacheblend.maybe_dump_request_debug(
                    forward_batch=forward_batch,
                    ctx=ctx,
                    input_ids=input_ids,
                    out_cache_loc=out_cache_loc,
                    request_locs_reason=request_locs_reason,
                    img_locs=img_locs,
                )
            except Exception as dump_exc:
                logger.warning("VLM-CacheBlend request dump skipped: %s", dump_exc)
            if img_locs.numel() == 0:
                return (
                    _vlm_cacheblend.CacheBlendStats(
                        role=ctx.role,
                        request_id=ctx.request_id,
                        fallback_reason=(
                            request_locs_reason or "no_target_image_tokens"
                        ),
                    ).finalize()
                )
            if ctx.role == "recipient":
                return self._cacheblend_probe_recipient(ctx, img_locs, img_positions)
            _vlm_cacheblend.capture_donor_kv(
                forward_batch=forward_batch,
                layer_ids=layer_ids,
                img_locs=img_locs,
                group_key=ctx.group_key,
                grid_sig=ctx.grid_sig,
                positions=img_positions,
                to_cpu=_vlm_cacheblend.get_config().donor_to_cpu,
            )
            return (
                _vlm_cacheblend.CacheBlendStats(
                    role="donor",
                    request_id=ctx.request_id,
                    n_image_tokens=int(img_locs.numel()),
                    pos_mode=_vlm_cacheblend.get_config().pos_mode,
                    select_mode=_vlm_cacheblend.get_config().select_mode,
                    fallback_reason="donor_captured",
                ).finalize()
            )
        except Exception as exc:  # never break the forward pass on capture failure
            logger.warning("VLM-CacheBlend donor capture skipped: %s", exc)
            return (
                _vlm_cacheblend.CacheBlendStats(
                    role=getattr(ctx, "role", "none"),
                    request_id=getattr(ctx, "request_id", ""),
                    fallback_reason=f"hook_exception:{type(exc).__name__}",
                ).finalize()
            )

    def _cacheblend_probe_recipient(self, ctx, img_locs, img_positions) -> None:
        cfg = _vlm_cacheblend.get_config()
        stats = _vlm_cacheblend.CacheBlendStats(
            role="recipient",
            request_id=ctx.request_id,
            n_image_tokens=int(img_locs.numel()),
            pos_mode=cfg.pos_mode,
            select_mode=cfg.select_mode,
        )
        donor = _vlm_cacheblend.get_donor_store().lookup(ctx.group_key)
        if donor is None or not donor.complete:
            stats.fallback_reason = "donor_not_ready"
            return stats.finalize()
        if tuple(donor.grid_sig) != tuple(ctx.grid_sig):
            stats.fallback_reason = "grid_mismatch"
            return stats.finalize()
        if int(donor.n_image_tokens) != int(img_locs.numel()):
            stats.fallback_reason = "image_token_count_mismatch"
            return stats.finalize()
        if cfg.pos_mode == "same" and not _vlm_cacheblend.positions_match(
            donor.positions, img_positions
        ):
            stats.fallback_reason = "position_mismatch"
            return stats.finalize()

        recompute_mask = _vlm_cacheblend.select_recompute_tokens(
            int(img_locs.numel()),
            cfg,
            device=img_locs.device,
        )
        stats.eligible = True
        stats.recomputed_tokens = int(recompute_mask.sum().item())
        stats.reused_tokens = int(img_locs.numel()) - stats.recomputed_tokens
        if _vlm_cacheblend.recipient_blend_was_used(ctx.request_id):
            stats.used = True
            stats.fallback_reason = "recipient_kv_blended"
            (
                stats.attention_skipped_tokens,
                stats.attention_active_ranges,
            ) = _vlm_cacheblend.recipient_attention_skip_stats(ctx.request_id)
        else:
            stats.fallback_reason = "recipient_fast_path_not_applied"
        return stats.finalize()

    @staticmethod
    def _cacheblend_aggregate_stats(stats_list):
        stats_list = [s for s in stats_list if s is not None]
        if not stats_list:
            return _vlm_cacheblend.CacheBlendStats()
        if len(stats_list) == 1:
            return stats_list[0]
        priority = {"recipient": 3, "donor": 2, "none": 1}
        role = max((s.role for s in stats_list), key=lambda r: priority.get(r, 0))
        reasons = {}
        for stats in stats_list:
            reason = stats.fallback_reason or ""
            if not reason:
                continue
            reasons[reason] = reasons.get(reason, 0) + 1
        fallback = "batch:" + ";".join(
            f"{reason}={count}" for reason, count in sorted(reasons.items())
        )
        cfg = _vlm_cacheblend.get_config()
        return _vlm_cacheblend.CacheBlendStats(
            role=role,
            used=any(s.used for s in stats_list),
            eligible=any(s.eligible for s in stats_list),
            fallback_reason=fallback,
            request_id=",".join(s.request_id for s in stats_list if s.request_id)[:256],
            n_image_tokens=sum(int(s.n_image_tokens) for s in stats_list),
            reused_tokens=sum(int(s.reused_tokens) for s in stats_list),
            recomputed_tokens=sum(int(s.recomputed_tokens) for s in stats_list),
            pos_mode=cfg.pos_mode,
            select_mode=cfg.select_mode,
            attention_skipped_tokens=sum(
                int(getattr(s, "attention_skipped_tokens", 0)) for s in stats_list
            ),
            attention_active_ranges=sum(
                int(getattr(s, "attention_active_ranges", 0)) for s in stats_list
            ),
        ).finalize()

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            tp_rank,
            tp_size,
            self.config.num_hidden_layers,
            self.config.__class__.model_type,
        ):
            if not isinstance(self.layers[layer_idx], nn.Identity):
                layer_self_attn = self.layers[layer_idx].self_attn
            if hasattr(layer_self_attn.attn, "k_scale"):
                layer_self_attn.attn.k_scale = scaling_factor
                layer_self_attn.attn.v_scale = scaling_factor
            else:
                raise RuntimeError(
                    "Self attention has no KV cache scaling " "factor attribute!"
                )


class Qwen2ForCausalLM(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
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

    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen2Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # handle the lm head on different pp ranks
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        # perform weight tying for PP
        if self.pp_group.world_size > 1 and config.tie_word_embeddings:
            if self.pp_group.is_first_rank:
                self.pp_group.send(
                    self.model.embed_tokens.weight, dst=self.pp_group.last_rank
                )
            elif self.pp_group.is_last_rank:
                emb_token_weight = self.pp_group.recv(
                    size=(config.vocab_size, config.hidden_size),
                    dtype=next(self.model.parameters()).dtype,
                    src=self.pp_group.first_rank,
                )
                self.lm_head.weight.copy_(emb_token_weight)

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embedding(input_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
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

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
        # decoder layer
        for i in range(start, end):
            layer = self.model.layers[i]
            forward_batch.hidden_states, forward_batch.residual = layer(
                positions,
                forward_batch.hidden_states,
                forward_batch,
                forward_batch.residual,
            )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                if self.pp_group.world_size > 1 and self.pp_group.is_last_rank:
                    # Handle pp weight tying here
                    # find the embed_tokens.weight in the weights
                    embed_token_weights = next(
                        filter(lambda x: x[0] == "model.embed_tokens.weight", weights)
                    )[1]
                    loaded_weight = embed_token_weights
                else:
                    continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = Qwen2ForCausalLM
