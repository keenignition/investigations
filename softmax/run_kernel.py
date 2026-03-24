#!/usr/bin/env python3
import sys

import torch  # must load before softmax_kernel so libc10 etc. are resolved
import softmax_kernel

from triton_kernels.fused import fused_softmax
from triton_kernels.online import online_softmax


def main():
    provider = (sys.argv[1] or "softmax_fused_all").strip().lower()
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    M = 2048
    x = torch.randn((M, N), device="cuda")

    if provider == "softmax_naive":
        fn = lambda: softmax_kernel.softmax_naive(x)
    elif provider == "softmax_wr":
        fn = lambda: softmax_kernel.softmax_wr(x)
    elif provider == "fused":  # triton
        fn = lambda: fused_softmax(x)
    elif provider == "softmax_fused_warp":
        fn = lambda: softmax_kernel.softmax_fused_warp(x)
    elif provider == "softmax_fused_block":
        fn = lambda: softmax_kernel.softmax_fused_block(x)
    elif provider == "softmax_online":
        fn = lambda: softmax_kernel.softmax_online(x)
    elif provider == "triton_online":
        fn = lambda: online_softmax(x)
    else:
        print(f"Unknown provider: {provider}", file=sys.stderr)
        sys.exit(1)

    # Warmup
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    # Profiled runs
    for _ in range(5):
        fn()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
