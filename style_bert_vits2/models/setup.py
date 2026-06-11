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

Note:
    親ディレクトリに pyproject.toml が存在する場合、setuptools が自動で
    読み込んで ValueError を起こすことがある。
    これを防ぐため dist.Distribution を直接インスタンス化し、
    pyproject.toml の探索をスキップしている。
"""
import sys
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools.dist import Distribution
from setuptools.command.build_ext import build_ext as _build_ext  # noqa: F401


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
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  →  sm_{arch} を追加")

    return flags


print(">>> GPU SM アーキテクチャを自動検出中...")
gencode_flags = get_gencode_flags()
print(f">>> 使用する gencode フラグ: {gencode_flags}\n")

ext = CUDAExtension(
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

# Distribution を直接生成することで pyproject.toml の自動探索を回避する
dist = Distribution(
    attrs={
        "name": "monotonic_align_cuda_core",
        "ext_modules": [ext],
        "cmdclass": {"build_ext": BuildExtension},
    }
)
dist.script_name = sys.argv[0]
dist.script_args = sys.argv[1:]
dist.parse_command_line()

# setup.py のあるディレクトリへ移動し、終了後(例外時も)元のディレクトリへ戻る
_original_dir = os.getcwd()
try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    dist.run_commands()
finally:
    os.chdir(_original_dir)