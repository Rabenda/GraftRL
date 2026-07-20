"""Small CUDA kernels used by CacheBlend sparse decoding."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


MAX_FUSED_SPARSE_CONTEXT_WIDTH = 65536


def sparse_decode_warmup_widths(
    max_context_len: int, min_context_len: int = 256
) -> tuple[int, ...]:
    """Return reachable fused-compaction widths without exceeding kernel support.

    A model may advertise a 128K context while a workload admits much shorter
    requests.  That must not make sparse-decode server initialization fail: contexts
    wider than the fused kernel limit simply use the existing dense fallback.
    """

    max_context_len = max(1, int(max_context_len))
    min_context_len = max(1, min(int(min_context_len), max_context_len))
    max_width = min(
        MAX_FUSED_SPARSE_CONTEXT_WIDTH,
        triton.next_power_of_2(max_context_len),
    )
    source_width = triton.next_power_of_2(min_context_len)
    widths: list[int] = []
    while source_width <= max_width:
        widths.append(min(source_width, max_context_len))
        source_width *= 2
    return tuple(widths)


@triton.jit(
    do_not_specialize=[
        "source_row_stride",
        "keep_row_stride",
        "output_row_stride",
        "source_cols",
        "output_cols",
    ],
    do_not_specialize_on_alignment=[
        "source_row_stride",
        "keep_row_stride",
        "output_row_stride",
        "source_cols",
        "output_cols",
    ],
)
def _compact_page_table_kernel(
    source_ptr,
    keep_ptr,
    output_ptr,
    source_row_stride,
    keep_row_stride,
    output_row_stride,
    source_cols,
    output_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Compact one page-table row using an inclusive prefix sum."""

    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    source_valid = cols < source_cols
    keep = tl.load(
        keep_ptr + row * keep_row_stride + cols,
        mask=source_valid,
        other=0,
    ).to(tl.int32)
    compact_cols = tl.cumsum(keep, axis=0) - 1

    # FlashAttention ignores the rectangular tail, but deterministic zero padding
    # keeps the helper's behavior identical to the CPU/reference implementation.
    tl.store(
        output_ptr + row * output_row_stride + cols,
        0,
        mask=cols < output_cols,
    )
    values = tl.load(
        source_ptr + row * source_row_stride + cols,
        mask=source_valid,
        other=0,
    )
    tl.store(
        output_ptr + row * output_row_stride + compact_cols,
        values,
        mask=source_valid & (keep != 0) & (compact_cols < output_cols),
    )


@triton.jit(
    do_not_specialize=[
        "source_row_stride",
        "drop_row_stride",
        "output_row_stride",
        "source_cols",
        "output_cols",
    ],
    do_not_specialize_on_alignment=[
        "source_row_stride",
        "drop_row_stride",
        "output_row_stride",
        "source_cols",
        "output_cols",
    ],
)
def _compact_sparse_page_table_kernel(
    source_ptr,
    drop_ptr,
    seq_lens_ptr,
    keep_recent_ptr,
    keep_first_ptr,
    sparse_lens_ptr,
    output_ptr,
    source_row_stride,
    drop_row_stride,
    output_row_stride,
    source_cols,
    output_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply stable drop positions and compact a decode page table in one kernel."""

    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    seq_len = tl.load(seq_lens_ptr + row).to(tl.int32)
    keep_recent = tl.load(keep_recent_ptr + row).to(tl.int32)
    keep_first = tl.load(keep_first_ptr + row).to(tl.int32)
    source_valid = (cols < source_cols) & (cols < seq_len)
    drop = tl.load(
        drop_ptr + row * drop_row_stride + cols,
        mask=cols < source_cols,
        other=0,
    ).to(tl.int1)
    active_drop = (
        drop
        & (cols >= keep_first)
        & (cols < tl.maximum(seq_len - keep_recent, 0))
    )
    keep = source_valid & ~active_drop
    # Match the reference path's fail-safe: never expose an empty context.
    kept_count = tl.sum(keep.to(tl.int32), axis=0)
    empty_context = kept_count == 0
    keep = keep | (empty_context & (cols == seq_len - 1))
    kept_count += empty_context.to(tl.int32)
    compact_cols = tl.cumsum(keep.to(tl.int32), axis=0) - 1
    tl.store(sparse_lens_ptr + row, kept_count)

    # FA3 consumes only [:cache_seqlens[row]]. The rectangular tail is unreachable,
    # so clearing output_cols entries before overwriting the compact prefix only adds
    # a full-context global-memory write to every compaction.
    values = tl.load(
        source_ptr + row * source_row_stride + cols,
        mask=source_valid,
        other=0,
    )
    tl.store(
        output_ptr + row * output_row_stride + compact_cols,
        values,
        mask=keep & (compact_cols < output_cols),
    )


@triton.jit(
    do_not_specialize=[
        "source_row_stride",
        "drop_row_stride",
        "output_row_stride",
        "source_cols",
        "output_cols",
    ],
    do_not_specialize_on_alignment=[
        "source_row_stride",
        "drop_row_stride",
        "output_row_stride",
        "source_cols",
        "output_cols",
    ],
)
def _compact_sparse_req_to_token_kernel(
    source_ptr,
    request_indices_ptr,
    drop_ptr,
    seq_lens_ptr,
    keep_recent_ptr,
    keep_first_ptr,
    sparse_lens_ptr,
    output_ptr,
    source_row_stride,
    drop_row_stride,
    output_row_stride,
    source_cols,
    output_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Gather the live request row and compact it without a dense intermediate."""

    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    request_row = tl.load(request_indices_ptr + row).to(tl.int64)
    seq_len = tl.load(seq_lens_ptr + row).to(tl.int32)
    keep_recent = tl.load(keep_recent_ptr + row).to(tl.int32)
    keep_first = tl.load(keep_first_ptr + row).to(tl.int32)
    source_valid = (cols < source_cols) & (cols < seq_len)
    drop = tl.load(
        drop_ptr + row * drop_row_stride + cols,
        mask=cols < source_cols,
        other=0,
    ).to(tl.int1)
    active_drop = (
        drop
        & (cols >= keep_first)
        & (cols < tl.maximum(seq_len - keep_recent, 0))
    )
    keep = source_valid & ~active_drop
    kept_count = tl.sum(keep.to(tl.int32), axis=0)
    empty_context = kept_count == 0
    keep = keep | (empty_context & (cols == seq_len - 1))
    kept_count += empty_context.to(tl.int32)
    compact_cols = tl.cumsum(keep.to(tl.int32), axis=0) - 1
    tl.store(sparse_lens_ptr + row, kept_count)

    # Stale tail entries are masked by sparse_lens and never read by attention.
    values = tl.load(
        source_ptr + request_row * source_row_stride + cols,
        mask=source_valid,
        other=0,
    )
    tl.store(
        output_ptr + row * output_row_stride + compact_cols,
        values,
        mask=keep & (compact_cols < output_cols),
    )


@triton.jit(
    do_not_specialize=[
        "source_row_stride",
        "output_row_stride",
    ],
    do_not_specialize_on_alignment=[
        "source_row_stride",
        "output_row_stride",
    ],
)
def _append_sparse_page_table_kernel(
    source_ptr,
    seq_lens_ptr,
    previous_seq_lens_ptr,
    sparse_lens_ptr,
    output_ptr,
    source_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Append newly decoded locations after the stable sparse prefix."""

    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    old_seq_len = tl.load(previous_seq_lens_ptr + row).to(tl.int32)
    new_seq_len = tl.load(seq_lens_ptr + row).to(tl.int32)
    old_sparse_len = tl.load(sparse_lens_ptr + row).to(tl.int32)
    growth = new_seq_len - old_seq_len
    valid = offsets < growth
    values = tl.load(
        source_ptr + row * source_row_stride + old_seq_len + offsets,
        mask=valid,
        other=0,
    )
    tl.store(
        output_ptr + row * output_row_stride + old_sparse_len + offsets,
        values,
        mask=valid,
    )
    tl.store(previous_seq_lens_ptr + row, new_seq_len)
    tl.store(sparse_lens_ptr + row, old_sparse_len + growth)


@triton.jit(
    do_not_specialize=[
        "source_row_stride",
        "output_row_stride",
    ],
    do_not_specialize_on_alignment=[
        "source_row_stride",
        "output_row_stride",
    ],
)
def _append_sparse_req_to_token_kernel(
    source_ptr,
    request_indices_ptr,
    seq_lens_ptr,
    previous_seq_lens_ptr,
    sparse_lens_ptr,
    output_ptr,
    source_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Append the live request row's causal tail directly into the graph buffer."""

    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    request_row = tl.load(request_indices_ptr + row).to(tl.int64)
    old_seq_len = tl.load(previous_seq_lens_ptr + row).to(tl.int32)
    new_seq_len = tl.load(seq_lens_ptr + row).to(tl.int32)
    old_sparse_len = tl.load(sparse_lens_ptr + row).to(tl.int32)
    growth = new_seq_len - old_seq_len
    valid = offsets < growth
    values = tl.load(
        source_ptr + request_row * source_row_stride + old_seq_len + offsets,
        mask=valid,
        other=0,
    )
    tl.store(
        output_ptr + row * output_row_stride + old_sparse_len + offsets,
        values,
        mask=valid,
    )
    tl.store(previous_seq_lens_ptr + row, new_seq_len)
    tl.store(sparse_lens_ptr + row, old_sparse_len + growth)


def compact_page_table(
    page_table: torch.Tensor,
    keep_mask: torch.Tensor,
    output_cols: int,
) -> torch.Tensor:
    """Fuse prefix-rank, nonzero and scatter into one row-wise CUDA kernel."""

    if not page_table.is_cuda or not keep_mask.is_cuda:
        raise ValueError("compact_page_table requires CUDA tensors")
    if page_table.ndim != 2 or keep_mask.shape != page_table.shape:
        raise ValueError("page_table and keep_mask must have the same 2-D shape")
    rows, source_cols = page_table.shape
    output_cols = int(output_cols)
    if rows == 0 or source_cols == 0 or output_cols <= 0:
        return page_table.new_zeros((int(rows), max(output_cols, 0)))

    block_size = triton.next_power_of_2(int(source_cols))
    if block_size > MAX_FUSED_SPARSE_CONTEXT_WIDTH:
        raise ValueError(f"sparse page table is too wide for fused compaction: {source_cols}")
    output = page_table.new_empty((int(rows), output_cols))
    _compact_page_table_kernel[(int(rows),)](
        page_table,
        keep_mask,
        output,
        page_table.stride(0),
        keep_mask.stride(0),
        output.stride(0),
        int(source_cols),
        output_cols,
        BLOCK_SIZE=block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return output


def compact_sparse_page_table(
    page_table: torch.Tensor,
    drop_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    keep_recent: torch.Tensor,
    keep_first: torch.Tensor,
    output_cols: int,
    *,
    output: torch.Tensor | None = None,
    sparse_lens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse tail protection, sparse masking, prefix rank, and page-table scatter.

    ``output`` and ``sparse_lens`` let the decode path reuse stable per-batch CUDA
    storage.  Avoiding two allocator calls per generated token matters because this
    helper runs once for every model forward, while the buffers are consumed on the
    same CUDA stream before their next overwrite.
    """

    if not page_table.is_cuda or not drop_mask.is_cuda:
        raise ValueError("compact_sparse_page_table requires CUDA tensors")
    if page_table.ndim != 2 or drop_mask.ndim != 2:
        raise ValueError("page_table and drop_mask must be 2-D")
    rows, source_cols = page_table.shape
    if int(drop_mask.shape[0]) != int(rows) or int(drop_mask.shape[1]) < int(source_cols):
        raise ValueError("drop_mask must cover every source page-table column")
    output_cols = int(output_cols)
    if rows == 0 or source_cols == 0 or output_cols <= 0:
        return (
            page_table.new_zeros((int(rows), max(output_cols, 0))),
            seq_lens.new_zeros((int(rows),), dtype=torch.int32),
        )

    block_size = triton.next_power_of_2(int(source_cols))
    if block_size > MAX_FUSED_SPARSE_CONTEXT_WIDTH:
        raise ValueError(f"sparse page table is too wide for fused compaction: {source_cols}")
    if output is None:
        output = page_table.new_empty((int(rows), output_cols))
    else:
        if (
            output.device != page_table.device
            or output.dtype != page_table.dtype
            or output.ndim != 2
            or int(output.shape[0]) < int(rows)
            or int(output.shape[1]) < output_cols
        ):
            raise ValueError("output does not cover the requested compact table")
        output = output[: int(rows), :output_cols]
    if sparse_lens is None:
        sparse_lens = seq_lens.new_empty((int(rows),), dtype=torch.int32)
    else:
        if (
            sparse_lens.device != seq_lens.device
            or sparse_lens.dtype != torch.int32
            or sparse_lens.ndim != 1
            or int(sparse_lens.numel()) < int(rows)
        ):
            raise ValueError("sparse_lens does not cover every page-table row")
        sparse_lens = sparse_lens[: int(rows)]
    _compact_sparse_page_table_kernel[(int(rows),)](
        page_table,
        drop_mask,
        seq_lens,
        keep_recent,
        keep_first,
        sparse_lens,
        output,
        page_table.stride(0),
        drop_mask.stride(0),
        output.stride(0),
        int(source_cols),
        output_cols,
        BLOCK_SIZE=block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return output, sparse_lens


def compact_sparse_req_to_token(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    drop_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    keep_recent: torch.Tensor,
    keep_first: torch.Tensor,
    source_cols: int,
    output_cols: int,
    *,
    output: torch.Tensor,
    sparse_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directly compact selected request rows into a stable attention buffer.

    The previous CUDA-Graph path first materialized ``req_to_token[req_indices]``
    and then compacted that temporary tensor. This fused gather+compact path reads
    the live request rows once and avoids both the intermediate allocation and its
    full-context gather kernel on every generated token.
    """

    if not req_to_token.is_cuda or not output.is_cuda or not drop_mask.is_cuda:
        raise ValueError("compact_sparse_req_to_token requires CUDA tensors")
    if req_to_token.ndim != 2 or drop_mask.ndim != 2 or output.ndim != 2:
        raise ValueError("request table, drop mask, and output must be 2-D")
    rows = int(req_pool_indices.numel())
    source_cols = int(source_cols)
    output_cols = int(output_cols)
    if (
        req_pool_indices.ndim != 1
        or seq_lens.shape != (rows,)
        or int(drop_mask.shape[0]) < rows
        or int(drop_mask.shape[1]) < source_cols
        or int(output.shape[0]) < rows
        or int(output.shape[1]) < output_cols
        or sparse_lens.shape != (rows,)
        or source_cols > int(req_to_token.shape[1])
    ):
        raise ValueError("direct sparse decode buffers have incompatible shapes")
    if rows == 0 or source_cols <= 0 or output_cols <= 0:
        return output[:rows, : max(output_cols, 0)], sparse_lens[:rows]
    block_size = triton.next_power_of_2(source_cols)
    if block_size > MAX_FUSED_SPARSE_CONTEXT_WIDTH:
        raise ValueError(
            f"sparse page table is too wide for fused compaction: {source_cols}"
        )
    output = output[:rows, :output_cols]
    _compact_sparse_req_to_token_kernel[(rows,)](
        req_to_token,
        req_pool_indices,
        drop_mask,
        seq_lens,
        keep_recent,
        keep_first,
        sparse_lens,
        output,
        req_to_token.stride(0),
        drop_mask.stride(0),
        output.stride(0),
        source_cols,
        output_cols,
        BLOCK_SIZE=block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return output, sparse_lens


def append_sparse_page_table(
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    previous_seq_lens: torch.Tensor,
    sparse_lens: torch.Tensor,
    output: torch.Tensor,
    max_growth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append only the causal tail when the sparse drop set is already stable.

    Once decode has moved beyond the final dropped position plus ``keep_recent``,
    every newly generated token is retained. Reusing the compact prefix changes
    the per-token work from scanning the full context to copying the new tail.
    """

    if not page_table.is_cuda or not output.is_cuda:
        raise ValueError("append_sparse_page_table requires CUDA tensors")
    rows = int(page_table.shape[0])
    if (
        page_table.ndim != 2
        or output.ndim != 2
        or int(output.shape[0]) < rows
        or seq_lens.shape != (rows,)
        or previous_seq_lens.shape != (rows,)
        or sparse_lens.shape != (rows,)
    ):
        raise ValueError("append sparse buffers do not match the page-table rows")
    max_growth = int(max_growth)
    if rows == 0 or max_growth <= 0:
        return output[:rows], sparse_lens[:rows]
    block_size = triton.next_power_of_2(max_growth)
    _append_sparse_page_table_kernel[(rows,)](
        page_table,
        seq_lens,
        previous_seq_lens,
        sparse_lens,
        output,
        page_table.stride(0),
        output.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=1,
    )
    return output[:rows], sparse_lens[:rows]


def append_sparse_req_to_token(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    previous_seq_lens: torch.Tensor,
    sparse_lens: torch.Tensor,
    output: torch.Tensor,
    max_growth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append new request-token locations without gathering the dense context."""

    if not req_to_token.is_cuda or not output.is_cuda:
        raise ValueError("append_sparse_req_to_token requires CUDA tensors")
    rows = int(req_pool_indices.numel())
    if (
        req_to_token.ndim != 2
        or req_pool_indices.ndim != 1
        or output.ndim != 2
        or int(output.shape[0]) < rows
        or seq_lens.shape != (rows,)
        or previous_seq_lens.shape != (rows,)
        or sparse_lens.shape != (rows,)
    ):
        raise ValueError("direct append buffers do not match the request rows")
    max_growth = int(max_growth)
    if rows == 0 or max_growth <= 0:
        return output[:rows], sparse_lens[:rows]
    block_size = triton.next_power_of_2(max_growth)
    _append_sparse_req_to_token_kernel[(rows,)](
        req_to_token,
        req_pool_indices,
        seq_lens,
        previous_seq_lens,
        sparse_lens,
        output,
        req_to_token.stride(0),
        output.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=1,
    )
    return output[:rows], sparse_lens[:rows]


def warmup_sparse_decode_kernels(
    device: torch.device | str,
    *,
    max_context_len: int = 8192,
    min_context_len: int = 256,
) -> None:
    """Compile every reachable context-width bucket before accepting work.

    Triton specializes compaction on the next-power-of-two source width. The old
    fixed list stopped at 8K even when the server admitted 16K+ contexts, so the first
    long request still paid cold compilation inside rollout latency.
    """

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    for actual_source_cols in sparse_decode_warmup_widths(
        max_context_len, min_context_len
    ):
        source_cols = triton.next_power_of_2(actual_source_cols)
        page_table = torch.zeros(
            (1, actual_source_cols), dtype=torch.int32, device=device
        )
        drop_mask = torch.zeros(
            (1, actual_source_cols), dtype=torch.bool, device=device
        )
        seq_lens = torch.full(
            (1,), actual_source_cols, dtype=torch.int32, device=device
        )
        keep_recent = torch.full((1,), 64, dtype=torch.int32, device=device)
        keep_first = torch.zeros((1,), dtype=torch.int32, device=device)
        compact_sparse_page_table(
            page_table,
            drop_mask,
            seq_lens,
            keep_recent,
            keep_first,
            actual_source_cols,
        )
        req_to_token = torch.zeros(
            (2, actual_source_cols + 1), dtype=torch.int32, device=device
        )
        req_pool_indices = torch.ones((1,), dtype=torch.int64, device=device)
        direct_output = page_table.new_empty((1, source_cols))
        direct_sparse_lens = seq_lens.new_empty((1,), dtype=torch.int32)
        compact_sparse_req_to_token(
            req_to_token,
            req_pool_indices,
            drop_mask,
            seq_lens,
            keep_recent,
            keep_first,
            actual_source_cols,
            actual_source_cols,
            output=direct_output,
            sparse_lens=direct_sparse_lens,
        )
        previous_seq_lens = seq_lens.clone()
        sparse_lens = seq_lens.clone()
        output = page_table.clone()
        seq_lens.add_(1)
        append_sparse_page_table(
            torch.nn.functional.pad(page_table, (0, 1)),
            seq_lens,
            previous_seq_lens,
            sparse_lens,
            torch.nn.functional.pad(output, (0, 1)),
            1,
        )
        direct_previous_seq_lens = seq_lens - 1
        direct_append_lens = seq_lens
        append_sparse_req_to_token(
            req_to_token,
            req_pool_indices,
            direct_append_lens,
            direct_previous_seq_lens,
            direct_sparse_lens,
            torch.nn.functional.pad(direct_output, (0, 1)),
            1,
        )
    torch.cuda.synchronize(device)
