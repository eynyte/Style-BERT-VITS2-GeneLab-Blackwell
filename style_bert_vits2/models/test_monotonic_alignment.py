"""
test_monotonic_alignment.py

CUDA実装とNumba実装の出力一致確認 + 速度比較。

使い方:
    python test_monotonic_alignment.py
"""

import time
import torch
import numpy as np


# ------------------------------------------------------------------ #
# Numba 実装（比較基準）
# ------------------------------------------------------------------ #
import numba
from numpy import float32, int32, zeros


@numba.jit(
    numba.void(
        numba.int32[:, :, ::1],
        numba.float32[:, :, ::1],
        numba.int32[::1],
        numba.int32[::1],
    ),
    nopython=True,
    nogil=True,
)
def _maximum_path_jit(paths, values, t_ys, t_xs):
    b = paths.shape[0]
    max_neg_val = -1e9
    for i in range(int(b)):
        path  = paths[i]
        value = values[i]
        t_y   = t_ys[i]
        t_x   = t_xs[i]
        v_prev = v_cur = 0.0
        index  = t_x - 1
        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                v_cur  = max_neg_val if x == y else value[y-1, x]
                v_prev = (0.0 if y == 0 else max_neg_val) if x == 0 else value[y-1, x-1]
                value[y, x] += max(v_prev, v_cur)
        for y in range(t_y - 1, -1, -1):
            path[y, index] = 1
            if index != 0 and (index == y or value[y-1, index] < value[y-1, index-1]):
                index -= 1


def maximum_path_numba(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    device = neg_cent.device
    dtype  = neg_cent.dtype
    nc_np  = neg_cent.data.cpu().numpy().astype(float32)
    path   = zeros(nc_np.shape, dtype=int32)
    t_t    = mask.sum(1)[:, 0].data.cpu().numpy().astype(int32)
    t_s    = mask.sum(2)[:, 0].data.cpu().numpy().astype(int32)
    _maximum_path_jit(path, nc_np, t_t, t_s)
    return torch.from_numpy(path).to(device=device, dtype=dtype)


# ------------------------------------------------------------------ #
# CUDA 実装
# ------------------------------------------------------------------ #
import importlib

def maximum_path_cuda(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    ext = importlib.import_module("monotonic_align_cuda_core")
    device = neg_cent.device
    dtype  = neg_cent.dtype
    B, T_y, T_x = neg_cent.shape
    nc_f32  = neg_cent.float().contiguous()
    t_y_max = mask[:, :, 0].sum(dim=1).to(torch.int32).contiguous()
    t_x_max = mask[:, 0, :].sum(dim=1).to(torch.int32).contiguous()
    path = torch.zeros(B, T_y, T_x, dtype=torch.int32, device=device)
    ext.maximum_path_cuda(path, nc_f32, t_y_max, t_x_max)
    return path.to(dtype=dtype)


# ------------------------------------------------------------------ #
# テストユーティリティ
# ------------------------------------------------------------------ #
def make_test_data(B=8, T_y=300, T_x=100, device="cuda"):
    """
    ランダムなneg_centとマスクを生成する。
    各サンプルの有効長をランダムに設定してパディングを模擬する。
    """
    neg_cent = torch.randn(B, T_y, T_x, device=device)

    # 各バッチの有効長をランダムに設定
    t_y_lens = torch.randint(T_x, T_y + 1, (B,))      # t_y >= t_x を保証
    t_x_lens = torch.randint(1, T_x + 1, (B,))

    mask = torch.zeros(B, T_y, T_x, device=device)
    for i in range(B):
        ty = t_y_lens[i].item()
        tx = t_x_lens[i].item()
        mask[i, :ty, :tx] = 1.0

    return neg_cent.float(), mask.float()


def benchmark(fn, neg_cent, mask, warmup=5, repeat=50, label=""):
    # ウォームアップ
    for _ in range(warmup):
        fn(neg_cent, mask)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        fn(neg_cent, mask)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeat * 1000  # ms/iter

    print(f"  {label:30s}: {elapsed:.3f} ms/iter")
    return elapsed


# ------------------------------------------------------------------ #
# メインテスト
# ------------------------------------------------------------------ #
def test_correctness(B=4, T_y=150, T_x=60):
    print("=" * 55)
    print("正当性テスト")
    print("=" * 55)
    device = "cuda"
    neg_cent, mask = make_test_data(B, T_y, T_x, device=device)

    path_numba = maximum_path_numba(neg_cent, mask)
    path_cuda  = maximum_path_cuda(neg_cent, mask)
    torch.cuda.synchronize()

    match = torch.equal(path_numba.cpu().int(), path_cuda.cpu().int())
    print(f"  出力一致: {match}")
    if not match:
        diff = (path_numba.cpu().int() - path_cuda.cpu().int()).abs()
        print(f"  最大差分: {diff.max().item()}")
        print(f"  不一致箇所数: {(diff > 0).sum().item()}")
    assert match, "CUDA実装とNumba実装の出力が一致しません！"
    print("  → OK\n")


def test_speed(B=16, T_y=300, T_x=100):
    print("=" * 55)
    print(f"速度比較  (B={B}, T_y={T_y}, T_x={T_x})")
    print("=" * 55)
    device = "cuda"
    neg_cent, mask = make_test_data(B, T_y, T_x, device=device)

    t_numba = benchmark(maximum_path_numba, neg_cent, mask, label="Numba JIT (元実装)")
    t_cuda  = benchmark(maximum_path_cuda,  neg_cent, mask, label="CUDA カーネル")
    speedup = t_numba / t_cuda
    print(f"\n  速度向上: {speedup:.2f}x\n")


def test_edge_cases():
    print("=" * 55)
    print("エッジケーステスト")
    print("=" * 55)
    device = "cuda"

    # バッチサイズ1
    nc, mask = make_test_data(1, 50, 20, device=device)
    p_n = maximum_path_numba(nc, mask)
    p_c = maximum_path_cuda(nc, mask)
    assert torch.equal(p_n.cpu().int(), p_c.cpu().int()), "B=1 で不一致"
    print("  B=1: OK")

    # t_y == t_x (最小パス幅)
    nc, mask = make_test_data(4, 50, 50, device=device)
    # 全サンプルを t_y = t_x = 30 に固定
    mask[:] = 0
    mask[:, :30, :30] = 1
    p_n = maximum_path_numba(nc, mask)
    p_c = maximum_path_cuda(nc, mask)
    assert torch.equal(p_n.cpu().int(), p_c.cpu().int()), "t_y==t_x で不一致"
    print("  t_y == t_x: OK")

    print()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA が利用できません。テストをスキップします。")
        exit(1)

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.version.cuda}")
    print(f"GPU:     {torch.cuda.get_device_name(0)}\n")

    # Numba JIT の初回コンパイルをトリガー（タイミングから除外するため）
    print("Numba JIT をウォームアップ中...")
    _dummy_nc, _dummy_mask = make_test_data(1, 10, 5)
    maximum_path_numba(_dummy_nc, _dummy_mask)
    print("完了\n")

    test_correctness()
    test_edge_cases()
    test_speed()
    test_speed(B=32, T_y=500, T_x=150)

    print("すべてのテストを通過しました。")
