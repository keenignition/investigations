from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.4.0"

ext_modules = [
    CUDAExtension(
        name='softmax_kernel',
        sources=[
            'csrc/binding.cpp',
            'csrc/torchBind.cu',
            'csrc/naive.cu',
            'csrc/wr.cu',
        ],
        extra_compile_args={
            'cxx': [
                '-O3',
                '-DNDEBUG',
                '-ffast-math',
                '-march=native',
                '-funroll-loops',
                '-std=c++17'
            ],
            'nvcc': [
                '-O3',
                '-use_fast_math',
                '-lineinfo',
                '-Xptxas=-v',
                '-lcurand',
                '-maxrregcount=128',
                '-arch=sm_75'
            ]
        }
    )
]

setup(
    name="softmax_kernel",
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    packages=[],  # only build the C++ extension; no Python packages to discover
)
