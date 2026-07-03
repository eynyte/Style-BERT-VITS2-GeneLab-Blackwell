"""
以下に記述されている関数のコメントはリファクタリング時に GPT-4 に生成させたもので、
コードと完全に一致している保証はない。あくまで参考程度とすること。
"""

from typing import Any

import numba
import torch


# Monotonic Alignment Search (MAS) は動的計画法であり本質的に逐次処理のため、
# GPU 上では効率よく計算できず CPU (numba JIT) へ都度持って行く必要がある。
# ここで発生する GPU<->CPU 往復コピーは、
#   1) 元実装が毎回新しくページアブル（通常の）ホストメモリを確保していた
#   2) neg_cent / t_t_max / t_s_max のコピーをそれぞれ別々に同期していた（3回同期）
# という2点で無駄が多く、特にステップ時間が短い高速な GPU ほど無視できない
# 同期待ち（GPU アイドル）を生んでいた。
# 以下では、
#   1) 一度確保した pinned（ページロック）メモリのバッファを使い回す
#   2) 3つの入力コピーをまとめて発行し、同期は1回にする
# ことで、往復にかかる時間そのものを削減する。numba 側の計算アルゴリズム
# （`__maximum_path_jit`）は一切変更していないため、計算結果は元実装と bit-exact
# に一致する。
_STAGING_BUFFERS: dict[str, torch.Tensor] = {}
_PIN_MEMORY_AVAILABLE = True


def _get_staging_buffer(
    key: str, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    """
    `shape` ぴったりの（C-contiguous な）ホスト側バッファを、`key` ごとに使い回して
    返す。

    pinned メモリの確保は通常のメモリ確保よりコストが高いため、毎ステップ確保する
    のではなく、これまでに見た最大の要素数を保持するフラットな 1 次元バッファを
    保持して使い回す（バッチサイズは基本一定、系列長もデータセットの上限で頭打ち
    になるため、学習の早い段階でバッファサイズは安定する）。
    多次元のバッファをそのまま大きく育てて一部だけスライスすると、末尾次元が
    バッファ側より小さい場合に非連続なビューになり、numba 側の C-contiguous
    要求（シグネチャの `::1`）を満たせなくなる。そのため実体は常にフラットな
    1 次元バッファとして保持し、`.view(shape)` で必要な形状に reshape する
    （1 次元バッファの先頭 numel 要素を reshape するだけなので、常に
    C-contiguous になる）。
    一部の環境（メモリロック上限が低いコンテナ等）では pinned メモリの確保に
    失敗することがあるため、その場合は通常のページアブルメモリに自動フォール
    バックし、学習が落ちないようにする。
    """
    global _PIN_MEMORY_AVAILABLE
    numel = 1
    for s in shape:
        numel *= s
    buf = _STAGING_BUFFERS.get(key)
    needs_new = buf is None or buf.dtype != dtype or buf.numel() < numel
    if needs_new:
        new_numel = numel
        if buf is not None and buf.dtype == dtype and buf.numel() > numel:
            new_numel = buf.numel()
        if _PIN_MEMORY_AVAILABLE:
            try:
                buf = torch.empty(new_numel, dtype=dtype, pin_memory=True)
            except (RuntimeError, OSError):
                # pinned メモリが確保できない環境向けのフォールバック。
                _PIN_MEMORY_AVAILABLE = False
                buf = torch.empty(new_numel, dtype=dtype)
        else:
            buf = torch.empty(new_numel, dtype=dtype)
        _STAGING_BUFFERS[key] = buf
    return buf[:numel].view(*shape)


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
    is_cuda = neg_cent.is_cuda
    b, t_y, t_x = neg_cent.shape

    neg_cent_pinned = _get_staging_buffer("neg_cent", (b, t_y, t_x), torch.float32)
    t_t_max_pinned = _get_staging_buffer("t_t_max", (b,), torch.int32)
    t_s_max_pinned = _get_staging_buffer("t_s_max", (b,), torch.int32)
    path_pinned = _get_staging_buffer("path", (b, t_y, t_x), torch.int32)

    # 3つの入力コピーをまとめて (non_blocking で) 発行してから1回だけ同期する。
    # dtype 変換（例えば bf16 -> float32）はコピー中に行われ、値は変わらない
    # （bf16 -> float32 は厳密に無誤差の変換）。
    neg_cent_pinned.copy_(neg_cent.detach(), non_blocking=is_cuda)
    t_t_max_pinned.copy_(mask.sum(1)[:, 0], non_blocking=is_cuda)
    t_s_max_pinned.copy_(mask.sum(2)[:, 0], non_blocking=is_cuda)
    if is_cuda:
        # numba は CUDA ストリームを介さず直接ホストメモリを読むため、ここで
        # 明示的に同期し、上記コピーの完了を保証する。
        torch.cuda.current_stream(device).synchronize()

    path_pinned.zero_()
    __maximum_path_jit(
        path_pinned.numpy(),
        neg_cent_pinned.numpy(),
        t_t_max_pinned.numpy(),
        t_s_max_pinned.numpy(),
    )

    # 戻り値は敢えて blocking 転送にする。path_pinned は次回呼び出し時に
    # 再利用するため、直前の転送が完了してから関数を抜ける必要がある
    # （そうしないと次回の zero_() 等と競合するおそれがある）。
    return path_pinned.to(device=device, dtype=dtype)


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
