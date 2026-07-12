"""
できます。`monotonic_alignment.py` の `maximum_path` は、CUDA テンソルが渡されたときに CPU 転送も Numba も一切使わない実装 (`_maximum_path_gpu`) に自動で振り分けるよう書き換えました。

## 結論から

**完全に「GPU だけ」にできました。** ただし1点だけ、アルゴリズム自体が持つ本質的な制約があります。

`value[y, x]` は `value[y-1, x]` と `value[y-1, x-1]` にしか依存しない、つまり**前の行(y-1)が確定しないと次の行(y)が計算できない**動的計画法なので、y方向のループそのものを消すことはできません(これをやろうとすると anti-diagonal wavefront 方式の専用 CUDA/Triton カーネルを書く必要があり、かなり大掛かりになります)。

その代わり、**1行分の更新(本来は x についての内側ループ)を、バッチ次元・x次元まとめて PyTorch のテンソル演算でベクトル化**しました。Numba 版はバッチも直列(`for i in range(b)`、`prange` 不使用)で処理していたので、特にバッチサイズが大きいほど恩恵が大きいはずです。逆探索(backtracking)も同様にバッチ方向にベクトル化しています。

## 変更内容

- `maximum_path`: `neg_cent.is_cuda` なら `_maximum_path_gpu` へ。CPU テンソルの場合は元の Numba 実装のまま(挙動は完全に不変)。
- `_maximum_path_gpu`: 新規追加。`.cpu()` / `.numpy()` / numba を一切呼ばず、ループ回数も `neg_cent.shape[1]`(テンソルの静的な shape)を使うため、`.item()` のような GPU→CPU 同期も発生しません。

## 検証したこと

このサンドボックスに実 GPU が無いため実速度は測れませんでしたが、正しさは徹底的に検証しました:
- バッチ内で系列長がバラバラなケース、`Ty > Tx` / `Tx > Ty` 両方、極端に短い系列(長さ1)、全要素同じ長さ、非contiguousなテンソル、float32/float64/boolマスクなど多数の条件で、**元の Numba 版と完全一致(`torch.equal`)**することを確認済みです。
- `requires_grad` は元の実装同様 `False`(勾配は流れません)。

## 同梱した `benchmark_maximum_path.py`

お手元の Blackwell 環境のプロジェクトルートに置いて実行すると、まず正しさを再検証し、CUDA が使えれば Numba版(CPU転送込み)とGPU版の実速度比較まで行います。小さいバッチでは Numba 版が有利な可能性もあるので、実際のバッチサイズ・系列長でこのスクリプトを回して確認することをおすすめします。

さらに踏み込みたい場合(`torch.compile` でループのカーネル起動オーバーヘッドを削る、または wavefront 方式の専用カーネルで y ループ自体を減らす等)も対応できるので、必要であれば教えてください。
"""
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

    neg_cent が CUDA テンソルの場合は、CPU への転送や Numba を一切使わずに
    PyTorch のテンソル演算だけで完結する GPU 実装 (_maximum_path_gpu) を使う。
    それ以外 (CPU テンソル) の場合は、従来通り Numba JIT 版を使う。

    Args:
        neg_cent (torch.Tensor): 負の中心を表すテンソル
        mask (torch.Tensor): マスクを表すテンソル

    Returns:
        Tensor: 計算された最大パスを表すテンソル
    """

    if neg_cent.is_cuda:
        return _maximum_path_gpu(neg_cent, mask)

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


@torch.no_grad()
def _maximum_path_gpu(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Maximum Path (monotonic alignment search) を GPU 上だけで計算する実装。

    Numba 版と同じ動的計画法を、以下のように書き換えている:
      - "y" (系列長方向, dim=1) についてのループだけは Python ループとして残す。
        これは value[y, x] が value[y-1, x] と value[y-1, x-1] にしか依存しない
        (=前の行が確定しないと次の行が計算できない) という、このアルゴリズム自体が
        持つ逐次依存性のため、完全に除去することはできない。
      - その1行分の更新 (本来は x についての内側ループ) は、バッチ次元・x 次元の
        両方について PyTorch のテンソル演算でまとめて計算する。これにより
        Numba 版では素の for ループでバッチも直列処理していた部分が、GPU 上で
        バッチ×x方向にまとめて並列実行される。
      - 経路の逆探索 (backtracking) も同様に、y についての Python ループ + バッチ
        方向のベクトル化で実装する。

    データは一度も CPU / numpy 側に転送されない (.cpu() や .numpy() を呼ばない)。
    ループ回数は neg_cent.shape[1] (テンソルの静的な shape) を使うため、
    途中で GPU 上の値を読み出して同期する (.item() など) 箇所も存在しない。

    Args:
        neg_cent (torch.Tensor): [B, Ty, Tx] の負の中心を表す CUDA テンソル
        mask (torch.Tensor): [B, Ty, Tx] のマスクを表す CUDA テンソル

    Returns:
        Tensor: neg_cent と同じ shape / dtype / device を持つ、計算された最大パス
    """

    device = neg_cent.device
    orig_dtype = neg_cent.dtype
    b, ty_max, tx_max = neg_cent.shape
    neg_inf = -1e9

    # value[y, x] を計算過程でそのまま書き換えていく (Numba 版の value 配列に相当)。
    # 呼び出し元の neg_cent を破壊しないよう、必ず新しいバッファに複製してから使う。
    value = neg_cent.detach().to(torch.float32).clone()

    # 各バッチ要素ごとの有効長。int32 で来ても int64 で来てもよいように round してから変換する。
    t_y = mask.sum(dim=1)[:, 0].round().to(torch.int64)  # [B] 有効な y の個数
    t_x = mask.sum(dim=2)[:, 0].round().to(torch.int64)  # [B] 有効な x の個数

    x_idx = torch.arange(tx_max, device=device).unsqueeze(0)  # [1, Tx]
    y_ar = torch.arange(ty_max, device=device).unsqueeze(1)  # [Ty, 1]
    t_x_row = t_x.unsqueeze(0)  # [1, B]
    t_y_row = t_y.unsqueeze(0)  # [1, B]

    # 各 y ごとに有効な x の範囲 [L, U) を全バッチ分まとめて事前計算しておく
    # (numba版の `for x in range(max(0, t_x+y-t_y), min(t_x, y+1))` に相当)
    l_all = (t_x_row + y_ar - t_y_row).clamp(min=0)  # [Ty, B]
    u_all = torch.minimum(t_x_row, y_ar + 1)  # [Ty, B]

    for y in range(ty_max):
        if y == 0:
            v_cur = torch.full((b, tx_max), neg_inf, device=device, dtype=torch.float32)
        else:
            prev = value[:, y - 1, :]  # [B, Tx]  (1つ前の行、既に確定済み)
            v_cur = prev.clone()
            if y < tx_max:
                # x == y のときは前の行を参照せず max_neg_val を使う
                v_cur[:, y] = neg_inf

        v_prev = torch.full((b, tx_max), neg_inf, device=device, dtype=torch.float32)
        if y > 0:
            # v_prev[x] = value[y-1, x-1] (右に1つシフト)
            v_prev[:, 1:] = prev[:, :-1]
        # x == 0 のときは、y == 0 なら 0.0 (始点)、それ以外は max_neg_val
        v_prev[:, 0] = 0.0 if y == 0 else neg_inf

        l = l_all[y].unsqueeze(1)  # [B, 1]
        u = u_all[y].unsqueeze(1)  # [B, 1]
        valid = (x_idx >= l) & (x_idx < u)  # [B, Tx] このバッチ要素・この y で有効な x

        updated = value[:, y, :] + torch.maximum(v_prev, v_cur)
        value[:, y, :] = torch.where(valid, updated, value[:, y, :])

    # ---- 逆探索 (backtracking) ----
    path = torch.zeros((b, ty_max, tx_max), dtype=torch.int32, device=device)
    index = (t_x - 1).clamp(min=0)  # [B] 現在の探索位置 (x 座標)
    batch_ar = torch.arange(b, device=device)

    for y in range(ty_max - 1, -1, -1):
        active = y < t_y  # [B] このバッチ要素にとって y が有効な範囲内かどうか
        idx_safe = index.clamp(0, tx_max - 1)

        # active な要素だけ path[i, y, index[i]] = 1 を立てる
        # (inactive な要素は元々 0 の場所に 0 を書くだけなので無害)
        path[batch_ar, y, idx_safe] = active.to(path.dtype)

        if y > 0:
            prev_row = value[:, y - 1, :]  # [B, Tx]
            idx0 = idx_safe
            idxm1 = (index - 1).clamp(0, tx_max - 1)
            v_a = torch.gather(prev_row, 1, idx0.unsqueeze(1)).squeeze(1)
            v_b = torch.gather(prev_row, 1, idxm1.unsqueeze(1)).squeeze(1)

            nz = index != 0
            eq = index == y
            lt = v_a < v_b
            # active でない要素は絶対に更新しない (無効な行の value を使って
            # index が壊れるのを防ぐ)
            should_dec = active & nz & (eq | lt)
            index = torch.where(should_dec, index - 1, index)

    return path.to(dtype=orig_dtype)
