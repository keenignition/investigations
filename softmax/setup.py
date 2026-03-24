import os
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

CUDA_ARCH = os.environ.get("CUDA_ARCH", "sm_75")

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
                "-maxrregcount=128",
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
