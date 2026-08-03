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
    与えられた負の中心とマスクを使用して最大パスを計算する

    Args:
        neg_cent (torch.Tensor): 負の中心を表すテンソル
        mask (torch.Tensor): マスクを表すテンソル

    Returns:
        Tensor: 計算された最大パスを表すテンソル
    """

    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = neg_cent.data.cpu().numpy().astype(float32)
    path = zeros(neg_cent.shape, dtype=int32)

    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(int32)
    __maximum_path_jit(path, neg_cent, t_t_max, t_s_max)

    return torch.from_numpy(path).to(device=device, dtype=dtype)


from super_monotonic_align import maximum_path as _sma_maximum_path
def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    sbv2/VITS 側: neg_cent, mask は [B, T_mel, T_text]
    super-monotonic-align 側: [B, T_text, T_mel] (Glow-TTS 由来) を要求するため
    dim1/dim2 を入れ替えてから渡し、結果をまた入れ替えて戻す。
    """
    dtype = neg_cent.dtype
    neg_cent_t = neg_cent.transpose(1, 2).float()   # [B, T_text, T_mel], fp32
    mask_t = mask.transpose(1, 2).to(torch.int32)   # [B, T_text, T_mel]
    path_t = _sma_maximum_path(neg_cent_t, mask_t, dtype=torch.float32)
    return path_t.transpose(1, 2).to(dtype)          # [B, T_mel, T_text] に戻す

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
