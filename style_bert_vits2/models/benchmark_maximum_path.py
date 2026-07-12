"""
monotonic_alignment.maximum_path の
  1) Numba (CPU) 版
  2) PyTorch ベクトル化 GPU 版 (_maximum_path_gpu)
  3) Triton GPU 版 (_maximum_path_gpu_triton, forward/backward 各1回のカーネル起動)
の正しさ検証 + 速度比較を行うスクリプト。

使い方:
    プロジェクトルート (style_bert_vits2 パッケージが import できる場所) で

        python benchmark_maximum_path.py

    として実行してください。CUDA が使えれば3実装すべての速度比較まで行い、
    CUDA が無ければ正しさの検証のみ行います。
    Triton がインストールされていない環境では、Triton 版の検証・計測は
    自動的にスキップされます (maximum_path はその場合 PyTorch ベクトル化版に
    フォールバックするので、学習が壊れることはありません)。
"""

from __future__ import annotations

import time

import torch

from style_bert_vits2.models.monotonic_alignment import (
    _TRITON_AVAILABLE,
    _maximum_path_gpu,
    maximum_path,
)


if _TRITON_AVAILABLE:
    from style_bert_vits2.models.monotonic_alignment import _maximum_path_gpu_triton


def make_inputs(
    batch_size: int, ty_max: int, tx_max: int, device: str, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    t_y = torch.randint(1, ty_max + 1, (batch_size,), generator=g)
    t_x = torch.randint(1, tx_max + 1, (batch_size,), generator=g)
    neg_cent = torch.randn(batch_size, ty_max, tx_max, generator=g)

    y_range = torch.arange(ty_max).view(1, ty_max, 1)
    x_range = torch.arange(tx_max).view(1, 1, tx_max)
    mask = (y_range < t_y.view(batch_size, 1, 1)) & (x_range < t_x.view(batch_size, 1, 1))
    mask = mask.float()

    return neg_cent.to(device), mask.to(device)


def check_correctness() -> None:
    print("=== 正しさ検証 (CPU の Numba 版を基準に比較) ===")
    print(f"Triton available: {_TRITON_AVAILABLE}")
    configs = [
        (1, 1, 1),
        (2, 10, 4),
        (8, 50, 12),
        (5, 1, 10),
        (5, 10, 1),
        (16, 80, 30),
        (4, 200, 60),
    ]
    all_ok = True
    for bsz, ty, tx in configs:
        neg_cent_cpu, mask_cpu = make_inputs(bsz, ty, tx, "cpu", seed=ty * tx + bsz)
        path_ref = maximum_path(neg_cent_cpu.clone(), mask_cpu.clone())

        # _maximum_path_gpu は純粋な PyTorch テンソル演算なので device を問わず動く。
        # CUDA が無い環境でも CPU テンソルのまま呼び出せばアルゴリズム自体の
        # 正しさは検証できる。
        # (一方 _maximum_path_gpu_triton は実際に GPU 上で実行する Triton カーネル
        #  なので、CPU テンソルでは検証できない。CUDA が使える場合のみ下で検証する。)
        path_vectorized = _maximum_path_gpu(neg_cent_cpu.clone(), mask_cpu.clone())
        ok_vec = torch.equal(path_ref, path_vectorized)
        all_ok &= ok_vec
        print(f"  B={bsz:3d} Ty={ty:4d} Tx={tx:4d} -> vectorized={'OK' if ok_vec else 'NG'}")

    if torch.cuda.is_available():
        neg_cent_cuda, mask_cuda = make_inputs(8, 300, 80, "cuda", seed=123)
        path_ref = maximum_path(neg_cent_cuda.cpu(), mask_cuda.cpu())

        path_cuda = maximum_path(neg_cent_cuda, mask_cuda)  # 実際に使われる経路 (自動振り分け)
        ok = torch.equal(path_cuda.cpu(), path_ref)
        all_ok &= ok
        print(f"  CUDA dispatch (実際に使われる経路) check -> {'OK' if ok else 'NG'}")

        if _TRITON_AVAILABLE:
            path_triton = _maximum_path_gpu_triton(neg_cent_cuda, mask_cuda)
            ok_tri = torch.equal(path_triton.cpu(), path_ref)
            all_ok &= ok_tri
            print(f"  Triton版 (CUDA テンソルで直接検証) -> {'OK' if ok_tri else 'NG'}")
    elif _TRITON_AVAILABLE:
        print("  (CUDA が無いため Triton 版の検証はスキップ。Triton カーネルは実 GPU 上でのみ実行できます)")

    print("ALL OK" if all_ok else "MISMATCH DETECTED")
    print()


def benchmark() -> None:
    if not torch.cuda.is_available():
        print("CUDA が利用できないため、速度比較はスキップします。")
        print("(このスクリプトは正しさの検証のみ行いました)")
        return

    print("=== 速度比較 (CUDA) ===")
    n_warmup = 3
    n_iters = 20
    ty_max, tx_max = 600, 150

    for bsz in [8, 32, 64, 128]:
        neg_cent_gpu, mask_gpu = make_inputs(bsz, ty_max, tx_max, "cuda", seed=bsz)
        neg_cent_cpu, mask_cpu = neg_cent_gpu.cpu(), mask_gpu.cpu()

        def timeit(fn, *args, n=n_iters):
            for _ in range(n_warmup):
                _ = fn(*args)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                _ = fn(*args)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n

        numba_time = timeit(maximum_path, neg_cent_cpu, mask_cpu)
        vec_time = timeit(_maximum_path_gpu, neg_cent_gpu, mask_gpu)

        line = (
            f"  B={bsz:4d}  Ty={ty_max} Tx={tx_max}  "
            f"numba(cpu,転送込み)={numba_time*1000:7.2f}ms  "
            f"vectorized(gpu)={vec_time*1000:7.2f}ms ({numba_time/vec_time:5.2f}x)"
        )

        if _TRITON_AVAILABLE:
            triton_time = timeit(_maximum_path_gpu_triton, neg_cent_gpu, mask_gpu)
            line += (
                f"  triton(gpu)={triton_time*1000:7.2f}ms "
                f"({numba_time/triton_time:5.2f}x / vectorized比 {vec_time/triton_time:5.2f}x)"
            )
        print(line)

    print()
    print("※ numba(cpu) 側は、CUDA テンソルを渡した場合と同条件にするため")
    print("   あえて CPU テンソルを渡して計測しています(実際の学習ループで")
    print("   CUDA テンソルを渡したときと同じ .cpu()/.numpy() 転送コストを含みます)。")


if __name__ == "__main__":
    check_correctness()
    benchmark()
