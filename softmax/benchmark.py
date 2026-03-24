from typing import Callable

import torch  # must load before softmax_kernel so libc10 etc. are resolved
import matplotlib.pyplot as plt
import softmax_kernel
import triton

from triton_kernels.fused import fused_softmax
from triton_kernels.online import online_softmax

# N=262144: standard Triton and CUDA fused/block kernels return 0 (unsupported/OOM).
# Triton online handles all sizes (streams in 4096-element chunks, no SRAM cap).
# CUDA online handles up to 262144 but spills registers heavily past 65536.
configs = [
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(10, 19)],  # 1024..262144
        line_arg="provider",
        line_vals=[
            "torch",
            "softmax_naive",
            "softmax_wr",
            "fused",
            "triton_online",
            "softmax_fused_warp",
            "softmax_fused_block",
            "softmax_online",
        ],
        line_names=[
            "Torch",
            "Softmax naive kernel",
            "Softmax warp reduction",
            "Fused Softmax (triton, 1-pass SRAM)",
            "Online Softmax (triton, 2-pass streaming)",
            "Softmax Fused Warp (CUDA)",
            "Softmax Fused Block (CUDA)",
            "Softmax Online (CUDA)",
        ],
        ylabel="TFLOPS",
        plot_name="softmax-performance",
        args={},
    )
]


def _gbps(x: torch.Tensor, *times_ms: float) -> tuple[float, ...]:
    """Convert one or more timings in ms to GB/s."""
    bytes_moved = 2 * x.numel() * x.element_size()  # read + write
    return tuple(bytes_moved * 1e-9 / (ms * 1e-3) for ms in times_ms)


@triton.testing.perf_report(configs)
def benchmark(N: int, provider: str, quantiles: list[float] = [0.5, 0.2, 0.8]):
    x = torch.randn((2048, N), device="cuda")

    def run_provider() -> Callable:
        if provider == "torch":
            return lambda: torch.softmax(x, dim=-1)
        elif provider == "softmax_naive":
            return lambda: softmax_kernel.softmax_naive(x)
        elif provider == "softmax_wr":
            return lambda: softmax_kernel.softmax_wr(x)
        elif provider == "fused":  # triton 1-pass
            return lambda: fused_softmax(x)
        elif provider == "triton_online":
            return lambda: online_softmax(x)
        elif provider == "softmax_fused_warp":
            return lambda: softmax_kernel.softmax_fused_warp(x)
        elif provider == "softmax_fused_block":
            return lambda: softmax_kernel.softmax_fused_block(x)
        elif provider == "softmax_online":
            return lambda: softmax_kernel.softmax_online(x)
        else:
            raise KeyError(f"Unknown provider {provider!r}.")

    try:
        ms, min_ms, max_ms = triton.testing.do_bench(
            run_provider(), quantiles=quantiles
        )
        return _gbps(x, ms, max_ms, min_ms)
    except (triton.CompilationError, ValueError, RuntimeError):
        return 0.0, 0.0, 0.0


df = benchmark.run(return_df=True)[0]
print(df)

# Plot the results
fig = plt.figure(figsize=(10, 6))
for column in df.columns[1:]:
    plt.plot(df["N"], df[column], label=column, marker="o")

plt.xscale("log", base=2)
plt.xlabel("Tensor width")
plt.ylabel("GB/s")
plt.title("Softmax Implementation Performance")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()

fig.savefig("softmax_performances.png")
print("Saved softmax_performances.png")
