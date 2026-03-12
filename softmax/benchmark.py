import torch
import softmax_kernel
from typing import Callable
import triton
import matplotlib.pyplot as plt

configs = [
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(10, 21)],
        line_arg="provider",
        line_vals=["torch", "softmax_naive"],
        line_names=[
            "Torch",
            "Softmax naive kernel (one block per row)",
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
    x = torch.randn((128, N), device="cuda")

    def run_provider() -> Callable:
        if provider == "torch":
            return lambda: torch.softmax(x, dim=-1)
        elif provider == "softmax_naive":
            return lambda: softmax_kernel.softmax_naive(x)
        else:
            raise KeyError(f"Unknown provider {provider!r}.")

    try:
        ms, min_ms, max_ms = triton.testing.do_bench(run_provider(), quantiles=quantiles)
        return _gbps(x, ms, max_ms, min_ms)
    except (triton.CompilationError, ValueError):
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
