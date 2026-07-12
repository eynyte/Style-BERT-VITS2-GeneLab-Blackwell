"""
monotonic_alignment.maximum_path の GPU 実装 (_maximum_path_gpu) の
正しさ検証 + 速度計測用スクリプト。

使い方:
    プロジェクトルート (style_bert_vits2 パッケージが import できる場所) で

        python benchmark_maximum_path.py

    として実行してください。CUDA が使えれば GPU 版と Numba(CPU) 版の
    速度比較まで行い、CUDA が無ければ CPU 上での正しさ検証のみ行います。
"""

from __future__ import annotations

import time

import torch

from style_bert_vits2.models.monotonic_alignment import _maximum_path_gpu, maximum_path


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
    print("=== 正しさ検証 (CPU の Numba 版と比較) ===")
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
        path_cpu_ref = maximum_path(neg_cent_cpu.clone(), mask_cpu.clone())

        # _maximum_path_gpu はテンソル演算のみなので device を問わず動く。
        # CUDA が無い環境でも CPU テンソルのまま呼び出せばアルゴリズム自体の
        # 正しさは検証できる。
        path_vectorized = _maximum_path_gpu(neg_cent_cpu.clone(), mask_cpu.clone())
        ok = torch.equal(path_cpu_ref, path_vectorized)
        all_ok &= ok
        print(f"  B={bsz:3d} Ty={ty:4d} Tx={tx:4d} -> {'OK' if ok else 'NG'}")

    if torch.cuda.is_available():
        neg_cent_cuda, mask_cuda = make_inputs(8, 300, 80, "cuda", seed=123)
        path_cuda = maximum_path(neg_cent_cuda, mask_cuda)  # dispatch -> _maximum_path_gpu
        path_ref = maximum_path(neg_cent_cuda.cpu(), mask_cuda.cpu())
        ok = torch.equal(path_cuda.cpu(), path_ref)
        all_ok &= ok
        print(f"  CUDA dispatch check -> {'OK' if ok else 'NG'}")

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

        for _ in range(n_warmup):
            _ = maximum_path(neg_cent_cpu, mask_cpu)  # Numba (CPU) 経路
            _ = maximum_path(neg_cent_gpu, mask_gpu)  # GPU ベクトル化経路
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = maximum_path(neg_cent_cpu, mask_cpu)
        t1 = time.perf_counter()
        numba_time = (t1 - t0) / n_iters

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = maximum_path(neg_cent_gpu, mask_gpu)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        gpu_time = (t1 - t0) / n_iters

        print(
            f"  B={bsz:4d}  Ty={ty_max} Tx={tx_max}  "
            f"numba(cpu, 転送込み)={numba_time*1000:7.2f}ms  "
            f"gpu(ベクトル化)={gpu_time*1000:7.2f}ms  "
            f"speedup={numba_time/gpu_time:5.2f}x"
        )
    print()
    print("※ numba(cpu) 側は、CUDA テンソルを渡した場合と同条件にするため")
    print("   あえて CPU テンソルを渡して計測しています(実際の学習ループで")
    print("   CUDA テンソルを渡したときと同じ .cpu()/.numpy() 転送コストを含みます)。")


if __name__ == "__main__":
    check_correctness()
    benchmark()
