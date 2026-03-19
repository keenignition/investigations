from typing import Callable

import matplotlib.pyplot as plt
import torch  # must load before softmax_kernel so libc10 etc. are resolved
import softmax_kernel
import triton

from triton_kernels.fused import fused_softmax

# Stop at N=131072; N=262144 exceeds shared memory / OOM for most kernels (Triton uses full row as BLOCK_SIZE).
configs = [
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(10, 18)],
        line_arg="provider",
        line_vals=["torch", "softmax_naive", "fused", "softmax_fused", "softmax_wr"],
        line_names=[
            "Torch",
            "Softmax naive kernel",
            "Fused Softmax (triton)",
            "Softmax fused (CUDA, N=1024)",
            "Softmax warp reduction",
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
        elif provider == "fused":
            return lambda: fused_softmax(x)
        elif provider == "softmax_fused":
            return lambda: softmax_kernel.softmax_fused(x)
        elif provider == "softmax_wr":
            return lambda: softmax_kernel.softmax_wr(x)
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
