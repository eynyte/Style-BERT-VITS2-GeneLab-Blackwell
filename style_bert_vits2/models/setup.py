"""
setup.py
nvcc でビルドするためのスクリプト。
使い方:
    python setup.py build_ext --inplace
ビルド後、同ディレクトリに
    monotonic_align_cuda_core.cpython-3XX-linux-gnu.so
が生成される。

現在接続されているすべての GPU の SM アーキテクチャを自動検出し、
そのアーキテクチャだけをターゲットにビルドします。
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch


def get_gencode_flags() -> list[str]:
    """接続中の GPU から SM を自動検出して -gencode フラグを返す。"""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA が利用できません。GPU ドライバと PyTorch (CUDA 版) を確認してください。"
        )

    seen: set[tuple[int, int]] = set()
    flags: list[str] = []

    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cap = (major, minor)
        if cap in seen:
            continue
        seen.add(cap)

        arch = f"{major}{minor}"
        flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
        print(
            f"  GPU {i}: {torch.cuda.get_device_name(i)}  →  sm_{arch} を追加"
        )

    return flags


print(">>> GPU SM アーキテクチャを自動検出中...")
gencode_flags = get_gencode_flags()
print(f">>> 使用する gencode フラグ: {gencode_flags}\n")

setup(
    name="monotonic_align_cuda_core",
    ext_modules=[
        CUDAExtension(
            name="monotonic_align_cuda_core",
            sources=["core.cu"],
            extra_compile_args={
                "nvcc": [
                    *gencode_flags,
                    "--use_fast_math",
                    "-O3",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)