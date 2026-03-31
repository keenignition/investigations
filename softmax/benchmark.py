from typing import Callable

import torch  # must load before softmax_kernel so libc10 etc. are resolved
import matplotlib.pyplot as plt
import softmax_kernel
import triton

from arch_config import load_config
from triton_kernels.fused import fused_softmax
from triton_kernels.liger import liger_softmax
from triton_kernels.online import online_softmax

_arch_cfg = load_config()
_bench = _arch_cfg["benchmark"]
_X_VALS: list[int] = _bench["x_vals"]


def _safe_batch_size(max_n: int, config_bs: int) -> int:
    try:
        free, _ = torch.cuda.mem_get_info()
        # input + output = 2 tensors of (batch_size × max_n × 4 bytes)
        bs = int(free * 0.4) // (max_n * 4 * 2)
        bs = max(1, min(bs, config_bs))
        if bs < config_bs:
            free_mb = free / 1024**2
            print(
                f"[benchmark] GPU has {free_mb:.0f} MiB free — "
                f"capping batch_size {config_bs} → {bs} to avoid OOM"
            )
        return bs
    except Exception:
        return config_bs


_BATCH_SIZE: int = _safe_batch_size(max(_X_VALS), _bench["batch_size"])

configs = [
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=_X_VALS,
        line_arg="provider",
        line_vals=[
            "torch",
            "softmax_naive",
            "softmax_wr",
            "fused",
            "triton_online",
            "liger",
            "softmax_fused_warp",
            "softmax_fused_block",
            "softmax_online",
            "softmax_online_v2",
        ],
        line_names=[
            "Torch",
            "Softmax naive kernel",
            "Softmax warp reduction",
            "Fused Softmax (triton, 1-pass SRAM)",
            "Online Softmax (triton, 2-pass streaming)",
            "Liger Softmax (triton)",
            "Softmax Fused Warp (CUDA)",
            "Softmax Fused Block (CUDA)",
            "Softmax Online (CUDA)",
            "Softmax Online V2 (CUDA, multi-block)",
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
    x = torch.randn((_BATCH_SIZE, N), device="cuda")

    def run_provider() -> Callable:
        if provider == "torch":
            return lambda: torch.softmax(x, dim=-1)
        if provider == "softmax_naive":
            return lambda: softmax_kernel.softmax_naive(x)
        elif provider == "softmax_wr":
            return lambda: softmax_kernel.softmax_wr(x)
        elif provider == "fused":
            return lambda: fused_softmax(x)
        elif provider == "triton_online":
            return lambda: online_softmax(x)
        elif provider == "liger":
            return lambda: liger_softmax(x)
        elif provider == "softmax_fused_warp":
            return lambda: softmax_kernel.softmax_fused_warp(x)
        elif provider == "softmax_fused_block":
            return lambda: softmax_kernel.softmax_fused_block(x)
        elif provider == "softmax_online":
            return lambda: softmax_kernel.softmax_online(x)
        elif provider == "softmax_online_v2":
            return lambda: softmax_kernel.softmax_online_v2(x)
        else:
            raise KeyError(f"Unknown provider {provider!r}.")

    try:
        ms, min_ms, max_ms = triton.testing.do_bench(
            run_provider(), quantiles=quantiles
        )
        return _gbps(x, ms, max_ms, min_ms)
    except (RuntimeError, torch.OutOfMemoryError) as e:
        if any(s in str(e) for s in ("not supported", "N must be", "out of memory")):
            return 0.0, 0.0, 0.0
        raise


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
