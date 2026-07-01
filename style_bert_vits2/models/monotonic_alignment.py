"""
以下に記述されている関数のコメントはリファクタリング時に GPT-4 に生成させたもので、
コードと完全に一致している保証はない。あくまで参考程度とすること。
"""

import os
from typing import Any

import numba
import torch
from numpy import float32, int32, zeros


# NOTE: maximum_path() は neg_cent を一旦 CPU に転送し (同期発生)、numba の
# JIT 関数でバッチ内の各サンプルに対して独立な動的計画法 (モノトニック
# アラインメント探索) を解いてから GPU に戻す。この CPU 計算はデータの
# 依存関係上どうしても学習ループを毎 step ブロックするため、GPU は
# その間アイドルする。GPU の計算自体が速いマシンほど、この
# シングルスレッド CPU 待ちがステップ時間に占める割合が大きくなり、
# GPU 使用率が下がる一因になる (RTX PRO 6000 → B200 での使用率低下と
# 整合する)。
#
# バッチ内の各サンプルは完全に独立な計算なので、下の __maximum_path_jit
# は parallel=True + numba.prange でバッチ次元を並列化し、CPU のブロッキング
# 時間そのものを短縮する。
#
# ただし DDP で 1 ノードに複数 GPU (= 複数プロセス) を積む構成では、
# 各プロセスが無制限に全 CPU コアを取り合うとかえってスレッド競合で
# 遅くなる可能性があるため、torchrun が設定する LOCAL_WORLD_SIZE を見て
# 1 プロセスあたりのスレッド数を頭割りする。明示的に NUMBA_NUM_THREADS が
# 設定されている場合はそちらを尊重し、上書きしない。
if "NUMBA_NUM_THREADS" not in os.environ:
    try:
        _local_world_size = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
    except ValueError:
        _local_world_size = 1
    _cpu_count = os.cpu_count() or 4
    _numba_threads = max(1, _cpu_count // _local_world_size)
    try:
        numba.set_num_threads(_numba_threads)
    except Exception:
        # 万一失敗しても致命的ではないのでデフォルトのスレッド数のまま続行する
        pass


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
)  # type: ignore
def __maximum_path_jit(paths: Any, values: Any, t_ys: Any, t_xs: Any) -> None:
    """
    与えられたパス、値、およびターゲットの y と x 座標を使用して JIT で最大パスを計算する

    バッチ内の各サンプル (インデックス i) の計算は互いに独立しているため、
    numba.prange でバッチ次元を並列化している (parallel=True と対)。

    Args:
        paths: 計算されたパスを格納するための整数型の 3 次元配列
        values: 値を格納するための浮動小数点型の 3 次元配列
        t_ys: ターゲットの y 座標を格納するための整数型の 1 次元配列
        t_xs: ターゲットの x 座標を格納するための整数型の 1 次元配列
    """

    b = paths.shape[0]
    max_neg_val = -1e9
    for i in numba.prange(int(b)):
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
