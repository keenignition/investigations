"""
Liger-kernel softmax (self-contained, no liger_kernel package required).

Adapted from:
  https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/softmax.py

Strategy: single block when n_cols fits inside one Triton block (≤65536),
multi-block (2-pass online) otherwise.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Helpers (inline equivalents of liger_kernel.ops.utils)
# ---------------------------------------------------------------------------

def _calculate_settings(n: int) -> Tuple[int, int]:
    """Return (BLOCK_SIZE, num_warps) for a row of length n."""
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > 65536:
        raise ValueError(f"n_cols={n} is too large for a single-block launch (>65536)")
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _softmax_single_block_forward_kernel(
    Y_ptr, Y_row_stride,
    X_ptr, X_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x = tl.load(X_ptr + row_id * X_row_stride + offs, mask=mask, other=-float("inf"), cache_modifier=".ca")
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    d = tl.sum(e, axis=0)
    y = e / d
    tl.store(Y_ptr + row_id * Y_row_stride + offs, y, mask=mask, cache_modifier=".cs")


@triton.jit
def _softmax_multi_block_forward_kernel(
    Y_ptr, Y_row_stride,
    X_ptr, X_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    m = tl.full([], -float("inf"), dtype=tl.float32)
    d = tl.full([], 0.0, dtype=tl.float32)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        xblk = tl.load(X_ptr + row_id * X_row_stride + idx, mask=mask, other=-float("inf"), cache_modifier=".ca")
        blk_max = tl.max(xblk, axis=0)
        new_m = tl.maximum(m, blk_max)
        d = d * tl.exp(m - new_m) + tl.sum(tl.exp(xblk - new_m), axis=0)
        m = new_m

    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        xblk = tl.load(X_ptr + row_id * X_row_stride + idx, mask=mask, other=-float("inf"), cache_modifier=".ca")
        yblk = tl.exp(xblk - m) / d
        tl.store(Y_ptr + row_id * Y_row_stride + idx, yblk, mask=mask, cache_modifier=".cs")


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

# BLOCK_SIZE cap for single-block path
_MAX_SINGLE_BLOCK_COLS = 65536


def liger_softmax(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    *batch, n_cols = x.shape
    x2d = x.view(-1, n_cols)
    n_rows = x2d.shape[0]
    y2d = torch.empty_like(x2d)

    if n_cols <= _MAX_SINGLE_BLOCK_COLS:
        BLOCK_SIZE, num_warps = _calculate_settings(n_cols)
        _softmax_single_block_forward_kernel[(n_rows,)](
            y2d, y2d.stride(0),
            x2d, x2d.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        # Multi-block online pass — use a fixed chunk size
        BLOCK_SIZE = 4096
        num_warps = 8
        _softmax_multi_block_forward_kernel[(n_rows,)](
            y2d, y2d.stride(0),
            x2d, x2d.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    return y2d.view(*batch, n_cols)
