"""
Online softmax in Triton.

Unlike the standard fused kernel (which loads the full row into SRAM at once
and caps at MAX_BLOCK_COLS=16384), this streams through the row in fixed-size
BLOCK_SIZE chunks, keeping only a running (max, sum) scalar between chunks.

Algorithm:
  Pass 1 — for each chunk of BLOCK_SIZE cols:
      m_new = max(running_max, chunk_max)
      s     = s * exp(running_max - m_new) + sum(exp(chunk - m_new))
      running_max = m_new
  Pass 2 — re-read each chunk, write exp(x - running_max) / s

Cost vs standard fused:
  Global reads:  2× (pass 1 + pass 2) instead of 1×
  Global writes: 1× (same)
  Benefit: N is unlimited — no SRAM capacity wall.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _online_softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    row_ptr = input_ptr + row_idx * input_row_stride
    out_ptr = output_ptr + row_idx * output_row_stride

    # Pass 1: stream through row computing online (max, sum)
    m = -float("inf")  # running max
    s = 0.0            # running corrected sum

    for col_off in tl.range(0, n_cols, BLOCK_SIZE, num_stages=2):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(row_ptr + cols, mask=mask, other=-float("inf"))
        m_chunk = tl.max(x, axis=0)
        m_new = tl.maximum(m, m_chunk)
        # Rescale old sum, add new chunk's contribution
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

    inv_s = 1.0 / s

    # Pass 2: re-read row, normalize and store
    for col_off in tl.range(0, n_cols, BLOCK_SIZE, num_stages=2):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(row_ptr + cols, mask=mask, other=0.0)
        tl.store(out_ptr + cols, tl.exp(x - m) * inv_s, mask=mask)


def online_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "x must be on CUDA"
    assert x.dtype == torch.float32, "x must be float32"
    assert x.dim() == 2, "x must be 2D"

    n_rows, n_cols = x.shape
    y = torch.empty_like(x)

    # BLOCK_SIZE: chunk of columns processed per loop iteration.
    # Larger = fewer loop iterations (better for large N), but uses more SRAM.
    # 4096 fits comfortably and gives good memory coalescing.
    BLOCK_SIZE = triton.next_power_of_2(min(n_cols, 4096))
    num_warps = 8

    _online_softmax_kernel[(n_rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y
