import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.4.0"

# Load .env if present (KEY=VALUE, no quotes), so pip install works without sourcing env.sh
_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from arch_config import detect_arch, load_config as load_arch_config  # noqa: E402


def generate_kernel_config(cfg: dict) -> None:
    # Import here so the path manipulation in gen_kernel_config isn't needed
    sys.path.insert(0, str(_ROOT / "scripts"))
    from gen_kernel_config import generate_header  # noqa: PLC0415
    header = generate_header(cfg)
    out_path = _ROOT / "csrc" / "kernel_config.h"
    out_path.write_text(header)
    print(f"[setup] wrote {out_path.relative_to(_ROOT)}")


CUDA_ARCH = os.environ.get("CUDA_ARCH") or detect_arch()
print(f"[setup] building for arch={CUDA_ARCH}")

_cfg = load_arch_config(CUDA_ARCH)
generate_kernel_config(_cfg)
_maxrregcount = _cfg["compile"]["maxrregcount"]

ext_modules = [
    CUDAExtension(
        name="softmax_kernel",
        sources=[
            "csrc/binding.cpp",
            "csrc/torchBind.cu",
            "csrc/naive.cu",
            "csrc/warpReduction.cu",
            "csrc/fused.cu",
            "csrc/fusedBlock.cu",
            "csrc/online.cu",
            "csrc/online_v2.cu",
        ],
        extra_compile_args={
            "cxx": [
                "-O3",
                "-DNDEBUG",
                "-ffast-math",
                "-march=native",
                "-funroll-loops",
                "-std=c++17",
            ],
            "nvcc": [
                "-O3",
                "-use_fast_math",
                "-lineinfo",
                "-Xptxas=-v",
                "-lcurand",
                f"-maxrregcount={_maxrregcount}",
                f"-arch={CUDA_ARCH}",
            ],
        },
    )
]

setup(
    name="softmax_kernel",
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    packages=[],  # only build the C++ extension; no Python packages to discover
)
