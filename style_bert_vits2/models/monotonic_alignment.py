"""
モノトニックアライメント探索 (VITS/Glow-TTS 系で使われる maximum path 計算) の実装。

アルゴリズム(動的計画法の漸化式)そのものは変更せず、CPU/GPU間のオーバーヘッドを
減らすことだけを目的に以下を変更している:
  1. t_t_max/t_s_max の算出を「全体を合計してから1列/1行だけ使う」から
     「先に1列/1行だけ切り出してから合計する」に変更し、無駄なGPU演算を削減。
     (元の mask.sum(1)[:, 0] は使わない列も含めて T_s 列すべてを合計していた)
  2. neg_cent と t_t_max/t_s_max をまとめて転送することで、CPU<->GPU間の
     同期(転送)回数を 3回 → 2回 に削減。
  3. dtype変換とcontiguous化をCPU転送前にGPU側で済ませ、CPU側での不要な
     コピー(astype等)を回避。既に float32 & contiguous なら丸ごとゼロコピー。
  4. Numba側の動的計画法をバッチ次元で並列化 (numba.prange)。各バッチ要素の
     計算は完全に独立しているため、出力はシングルスレッド版とビット単位で一致する。
  5. GPUへ書き戻すバッファを pinned memory で確保し、H2D転送(non_blocking)を高速化。
     CUDA以外のデバイス(CPU/MPS等)では pinned memory を使わず従来通り同期転送する。
"""

from typing import Any

import numba
import torch


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    与えられた負の中心とマスクを使用して最大パスを計算する。

    Args:
        neg_cent (torch.Tensor): 負の中心を表すテンソル (B, T_t, T_s)
        mask (torch.Tensor): マスクを表すテンソル (B, T_t, T_s)

    Returns:
        Tensor: 計算された最大パスを表すテンソル (neg_cent と同じ device/dtype)
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    is_cuda = device.type == "cuda"

    # dtype変換・contiguous化はGPU側で行い、CPU転送後の再コピーを避ける。
    # 既に float32 かつ contiguous であれば、以下はどちらもコピーが発生しない。
    neg_cent = neg_cent.detach()
    if neg_cent.dtype != torch.float32:
        neg_cent = neg_cent.float()
    neg_cent = neg_cent.contiguous()

    # t_t_max/t_s_max: 使う1列/1行だけ先に切り出してから合計する
    # (mask.sum(1)[:, 0] と数学的に等価だが、GPU側の演算量が
    #  O(B*T_t*T_s) から O(B*T_t + B*T_s) に減る)。
    t_t_max = mask[:, :, 0].sum(dim=1)
    t_s_max = mask[:, 0, :].sum(dim=1)
    lengths = torch.stack((t_t_max, t_s_max), dim=0).to(torch.int32)

    # CPUへの転送は neg_cent と lengths の2回にまとめる(元は3回)。
    neg_cent_np = neg_cent.to("cpu", non_blocking=is_cuda).numpy()
    lengths_np = lengths.to("cpu", non_blocking=is_cuda).numpy()
    t_t_max_np, t_s_max_np = lengths_np[0], lengths_np[1]

    # 戻り値用バッファ。CUDAへ書き戻す場合は pinned memory にして
    # H2D転送(DMA)を高速化する。CUDA以外では通常のメモリを使う。
    path_t = torch.zeros(neg_cent_np.shape, dtype=torch.int32, pin_memory=is_cuda)
    path_np = path_t.numpy()

    __maximum_path_jit(path_np, neg_cent_np, t_t_max_np, t_s_max_np)

    return path_t.to(device=device, dtype=dtype, non_blocking=is_cuda)


@numba.jit(
    numba.void(
        numba.int32[:, :, ::1],
        numba.float32[:, :, ::1],
        numba.int32[::1],
        numba.int32[::1],
    ),
    nopython=True,
    nogil=True,
    parallel=True,
    cache=True,
)  # type: ignore
def __maximum_path_jit(paths: Any, values: Any, t_ys: Any, t_xs: Any) -> None:
    """
    与えられたパス、値、およびターゲットの y と x 座標を使用して JIT で最大パスを計算する。

    バッチ内の各サンプル i の計算は paths[i]/values[i] のみを読み書きし、他の i とは
    完全に独立しているため、numba.prange でバッチ次元を複数CPUスレッドに分散している。
    1サンプル内の処理順序・漸化式は元の実装から変更していないため、出力はスレッド数に
    関わらずシングルスレッド版とビット単位で一致する。cache=True によりコンパイル結果を
    ディスクにキャッシュし、プロセス再起動時の再コンパイルコストも削減する。

    Args:
        paths: 計算されたパスを格納するための整数型の 3 次元配列
               (呼び出し前に 0 で初期化されている必要がある)
        values: 値を格納するための浮動小数点型の 3 次元配列
        t_ys: ターゲットの y 座標を格納するための整数型の 1 次元配列
        t_xs: ターゲットの x 座標を格納するための整数型の 1 次元配列
    """

    b = paths.shape[0]
    max_neg_val = -1e9
    for i in numba.prange(b):
        path = paths[i]
        value = values[i]
        t_y = t_ys[i]
        t_x = t_xs[i]

        v_prev = v_cur = 0.0
        index = t_x - 1

        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                if x == y:
                    v_cur = max_neg_val
                else:
                    v_cur = value[y - 1, x]
                if x == 0:
                    if y == 0:
                        v_prev = 0.0
                    else:
                        v_prev = max_neg_val
                else:
                    v_prev = value[y - 1, x - 1]
                value[y, x] += max(v_prev, v_cur)

        for y in range(t_y - 1, -1, -1):
            path[y, index] = 1
            if index != 0 and (
                index == y or value[y - 1, index] < value[y - 1, index - 1]
            ):
                index = index - 1
