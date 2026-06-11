"""
setup.py
nvcc でビルドするためのスクリプト。

使い方:
    python setup.py build_ext --inplace

ビルド後、同ディレクトリに
    monotonic_align_cuda_core.cpython-3XX-linux-gnu.so
が生成される。
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="monotonic_align_cuda_core",
    ext_modules=[
        CUDAExtension(
            name="monotonic_align_cuda_core",
            sources=["core.cu"],
            extra_compile_args={
                "nvcc": [
                    # SM アーキテクチャ: 環境に合わせて追加・削除してください
                    # Turing  (RTX 20xx)       : sm_75
                    # Ampere  (RTX 30xx, A100) : sm_80, sm_86
                    # Ada     (RTX 40xx)       : sm_89
                    # Hopper  (H100)           : sm_90
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_90,code=sm_90",
                    "-gencode=arch=compute_100,code=sm_100",
                    "-gencode=arch=compute_120,code=sm_120",
                    "--use_fast_math",
                    "-O3",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
