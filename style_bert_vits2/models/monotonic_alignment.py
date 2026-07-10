"""
以下に記述されている関数のコメントはリファクタリング時に GPT-4 に生成させたもので、
コードと完全に一致している保証はない。あくまで参考程度とすること。
"""

from typing import Any

import numba
import torch
from numpy import float32, int32, zeros


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    与えられた負の中心とマスクを使用して最大パス (Monotonic Alignment Search) を計算する

    `neg_cent` が CUDA テンソルの場合は GPU 上だけで完結する純粋な PyTorch 実装
    (`_maximum_path_gpu`) を使用する。従来の Numba/CPU 実装は、テンソルを毎ステップ
    GPU→CPU に転送して Numba (CPU, シングルスレッド) で計算し、結果をまた GPU に
    転送し直す必要があり、これが学習ループ全体を CPU との同期待ちで止めてしまう
    大きなボトルネックになっていた。GPU 実装ではこの往復が発生しない。

    CPU テンソルが渡された場合(CPU 上でのデバッグ実行時など)は、従来通り
    Numba JIT による CPU 実装にフォールバックする。

    Args:
        neg_cent (torch.Tensor): 負の中心を表すテンソル。形状は [batch, t_y_max, t_x_max]
        mask (torch.Tensor): マスクを表すテンソル。形状は [batch, t_y_max, t_x_max]

    Returns:
        Tensor: 計算された最大パスを表すテンソル (neg_cent と同じ device / dtype)
    """
    if neg_cent.is_cuda:
        return _maximum_path_gpu(neg_cent, mask)
    return _maximum_path_cpu(neg_cent, mask)


def _maximum_path_cpu(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    CPU (Numba JIT) 版の実装。GPU テンソルが渡されなかった場合のフォールバック用。
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent_np = neg_cent.data.cpu().numpy().astype(float32)
    path = zeros(neg_cent_np.shape, dtype=int32)

    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(int32)
    __maximum_path_jit(path, neg_cent_np, t_t_max, t_s_max)

    return torch.from_numpy(path).to(device=device, dtype=dtype)


@numba.jit(
    numba.void(
        numba.int32[:, :, ::1],
        numba.float32[:, :, ::1],
        numba.int32[::1],
        numba.int32[::1],
    ),
    nopython=True,
    nogil=True,
)  # type: ignore
def __maximum_path_jit(paths: Any, values: Any, t_ys: Any, t_xs: Any) -> None:
    """
    与えられたパス、値、およびターゲットの y と x 座標を使用して JIT で最大パスを計算する

    Args:
        paths: 計算されたパスを格納するための整数型の 3 次元配列
        values: 値を格納するための浮動小数点型の 3 次元配列
        t_ys: ターゲットの y 座標を格納するための整数型の 1 次元配列
        t_xs: ターゲットの x 座標を格納するための整数型の 1 次元配列
    """

    b = paths.shape[0]
    max_neg_val = -1e9
    for i in range(int(b)):
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


def _maximum_path_gpu_impl(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    GPU (純粋な PyTorch 演算) による Monotonic Alignment Search のベクトル化実装。

    元の Numba 実装は、バッチ内の要素ごとに (y, x) の 2 重ループを CPU 上で逐次実行
    するアルゴリズムだったが、行 y の DP 更新は 1 つ前の行 (y-1) の値のみに依存する
    ため、「バッチ次元・列(x)次元は全てまとめてベクトル演算し、行(y)方向のみ逐次
    ループする」という形に書き換えることができる。これにより計算全体を GPU 上に
    保持したまま (CPU に同期することなく) 実行できる。

    Args:
        neg_cent: [b, t_y_max, t_x_max] 負の中心を表すテンソル
        mask: [b, t_y_max, t_x_max] 有効領域を表すマスク (sequence_mask の外積)

    Returns:
        [b, t_y_max, t_x_max] の 0/1 パス行列 (neg_cent と同じ device / dtype)
    """
    dtype = neg_cent.dtype
    device = neg_cent.device
    neg_inf = -1e9

    # DP 計算は float32 で行う。元の Numba 実装も内部では float32 にキャストして
    # おり、bf16/fp16 のまま比較・加算を繰り返すと精度不足で誤った経路を選んで
    # しまう可能性があるため、それに合わせている。
    value = neg_cent.detach().to(torch.float32).clone()
    b, t_y_max, t_x_max = value.shape

    t_y = mask.sum(1)[:, 0].to(torch.long)
    t_x = mask.sum(2)[:, 0].to(torch.long)

    # ---- forward pass: DP テーブルの構築 ----
    # 行 y の更新は行 y-1 のみに依存するため、y についてのみ逐次ループし、
    # バッチ・列(x)方向はまとめてベクトル演算する。
    for y in range(1, t_y_max):
        row_prev = value[:, y - 1, :]
        v_cur = row_prev.clone()
        if y < t_x_max:
            v_cur[:, y] = neg_inf
        v_prev = torch.full_like(row_prev, neg_inf)
        if t_x_max > 1:
            v_prev[:, 1:] = row_prev[:, : t_x_max - 1]
        value[:, y, :] = value[:, y, :] + torch.maximum(v_prev, v_cur)

    # ---- backward pass: 経路の復元 ----
    path = torch.zeros((b, t_y_max, t_x_max), dtype=torch.float32, device=device)
    index = (t_x - 1).clamp(min=0)
    batch_idx = torch.arange(b, device=device)

    for y in range(t_y_max - 1, -1, -1):
        active = y < t_y  # このバッチ要素にとって行 y がまだ有効範囲内かどうか
        idx_clamped = index.clamp(min=0, max=t_x_max - 1)
        path[batch_idx, y, idx_clamped] = active.to(path.dtype)

        if y > 0:
            row_prev = value[:, y - 1, :]
            cur_val = row_prev.gather(1, idx_clamped.unsqueeze(1)).squeeze(1)
            prev_idx = (idx_clamped - 1).clamp(min=0)
            prev_val = row_prev.gather(1, prev_idx.unsqueeze(1)).squeeze(1)
            cond = (idx_clamped == y) | (cur_val < prev_val)
        else:
            cond = idx_clamped == y
        should_dec = active & (index != 0) & cond
        index = torch.where(should_dec, index - 1, index)

    return path.to(dtype=dtype)


try:
    # Python ループ部分を TorchScript でコンパイルし、インタプリタのオーバーヘッドを
    # 削減する(commons.py の fused_add_tanh_sigmoid_multiply と同じ方針)。
    # 環境によってはコンパイルできない可能性があるため、失敗時は eager 実装に
    # フォールバックする。
    _maximum_path_gpu = torch.jit.script(_maximum_path_gpu_impl)
except Exception:
    _maximum_path_gpu = _maximum_path_gpu_impl
