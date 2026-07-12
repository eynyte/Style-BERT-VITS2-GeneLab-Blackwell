"""
以下に記述されている関数のコメントはリファクタリング時に GPT-4 に生成させたもので、
コードと完全に一致している保証はない。あくまで参考程度とすること。
"""

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
Triton の解釈実行モード(`TRITON_INTERPRET=1`)を使うと、実 GPU が無いこのサンドボックスでも Triton カーネルの正しさだけは検証できることが分かったので、ループを1回のカーネル起動に収める実装まで書いて検証します。Triton の解釈実行モード(`TRITON_INTERPRET=1`)を使い、実 GPU が無いこのサンドボックスでも正しさだけは検証できたので、y方向のループそのものを1回のカーネル起動に収める Triton 実装を追加しました。

## やったこと

`_maximum_path_gpu`(前回のPyTorchベクトル化版)は正しく動きますが、y方向のループ1回ごとに `clone`/`where`/比較など数個の小さな CUDA カーネルが起動していました。系列長が数百あると、この「小さなカーネルを何百回も起動するオーバーヘッド」がボトルネックになり得ます。

そこで **Triton カーネルを新規に書き**、y のループを Python 側ではなくカーネル内部のループとして実行するようにしました。これにより:

- **forward: 1回、backward: 1回、合計たった2回のカーネル起動**で系列長に関わらず計算が完了します(以前は系列長×数個のカーネル起動が必要でした)。
- バッチ次元は「1バッチ要素 = 1プログラム」として GPU 上で並列実行され、x方向は各プログラム内で `BLOCK_X` 幅のベクトル演算として並列化されます。

## 自動フォールバック付き

Triton は必須依存にしていません。`maximum_path` は次の優先順位で自動的に切り替わります:

1. CUDA テンソル + Triton が使える → Triton版(最速)
2. CUDA テンソル + Triton が無い/実行時に失敗 → PyTorch ベクトル化版(1回だけ警告を出してフォールバック)
3. CPU テンソル → 従来の Numba 版(変更なし)

なので `pip install triton` していない環境や、万一 Blackwell 環境で Triton が何か問題を起こしても、学習が落ちることはありません。

## 検証について(重要な注意点)

このサンドボックスに実 GPU が無いため、Triton の`TRITON_INTERPRET=1`という「実GPUなしでカーネルのロジックだけをPythonエミュレーションで検証するモード」を使い、多数のバッチサイズ・系列長の組み合わせで Numba 版と完全一致することを確認しました。ただし**これはロジック検証であり、実GPU上でのコンパイル・実行そのものはご確認いただく必要があります**。`benchmark_maximum_path.py` を実行すれば、実機で自動的に正しさの再検証と速度比較(Numba / ベクトル化版 / Triton版)が行われます。

さらに欲を言えば forward/backward を1カーネルに統合してカーネル起動をさらに1回減らす余地もありますが、効果は小さく複雑さが増すため、今回は見送りました。必要であれば対応します。
"""
exec_cmd("pip install triton")

import warnings
from typing import Any

import numba
import torch
from numpy import float32, int32, zeros


try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # noqa: BLE001 - triton は無くても動く任意の高速化経路のため
    _TRITON_AVAILABLE = False

_warned_triton_fallback = False


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    与えられた負の中心とマスクを使用して最大パスを計算する

    neg_cent が CUDA テンソルの場合、CPU への転送や Numba を一切使わずに計算する。
    その中でも、Triton が利用可能ならバッチ全体を1回のカーネル起動 (forward /
    backward でそれぞれ1回、計2回) だけで処理する _maximum_path_gpu_triton を
    優先して使う。Triton が入っていない場合や、Triton 版が何らかの理由で失敗
    した場合は、PyTorch のテンソル演算だけで書いた _maximum_path_gpu に自動で
    フォールバックする。
    CPU テンソルの場合は、従来通り Numba JIT 版を使う (この経路は変更していない)。

    Args:
        neg_cent (torch.Tensor): 負の中心を表すテンソル
        mask (torch.Tensor): マスクを表すテンソル

    Returns:
        Tensor: 計算された最大パスを表すテンソル
    """

    if neg_cent.is_cuda:
        if _TRITON_AVAILABLE:
            try:
                return _maximum_path_gpu_triton(neg_cent, mask)
            except Exception as e:  # noqa: BLE001
                global _warned_triton_fallback
                if not _warned_triton_fallback:
                    warnings.warn(
                        "Triton 版 maximum_path の実行に失敗したため、"
                        f"PyTorch ベクトル化版にフォールバックします: {e!r}"
                    )
                    _warned_triton_fallback = True
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
    Maximum Path (monotonic alignment search) を GPU 上だけ (PyTorch テンソル演算
    のみ) で計算する実装。Triton が使えない環境向けのフォールバックでもある。

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


if _TRITON_AVAILABLE:

    @triton.jit
    def __maximum_path_fwd_triton_kernel(
        value_ptr,
        t_y_ptr,
        t_x_ptr,
        Ty,
        Tx,
        stride_b,
        stride_y,
        stride_x,
        BLOCK_X: tl.constexpr,
    ):
        """
        バッチ要素1個 = 1プログラムとして、forward の動的計画法を丸ごと
        1回のカーネル起動で計算する。y についてのループはカーネル内部の
        (Python レベルではない) ループとして実行され、x とバッチはこの
        カーネル自体の並列実行 (プログラム間 = バッチ、BLOCK_X = x) によって
        並列化される。
        """

        pid_b = tl.program_id(0)

        t_y = tl.load(t_y_ptr + pid_b)
        t_x = tl.load(t_x_ptr + pid_b)

        x_off = tl.arange(0, BLOCK_X)
        in_bounds = x_off < Tx
        base = pid_b * stride_b
        neg_inf = -1e9

        for y in range(0, Ty):
            cur_ptr = value_ptr + base + y * stride_y + x_off * stride_x
            orig_val = tl.load(cur_ptr, mask=in_bounds, other=0.0)

            if y == 0:
                v_cur = tl.full([BLOCK_X], neg_inf, dtype=tl.float32)
                v_prev = tl.where(x_off == 0, 0.0, neg_inf)
            else:
                prev_ptr = value_ptr + base + (y - 1) * stride_y + x_off * stride_x
                prev_row = tl.load(prev_ptr, mask=in_bounds, other=neg_inf)
                v_cur = tl.where(x_off == y, neg_inf, prev_row)

                shift_mask = in_bounds & (x_off >= 1)
                prevm1_ptr = (
                    value_ptr + base + (y - 1) * stride_y + (x_off - 1) * stride_x
                )
                v_prev = tl.load(prevm1_ptr, mask=shift_mask, other=neg_inf)

            band_lo = tl.maximum(t_x + y - t_y, 0)
            band_hi = tl.minimum(t_x, y + 1)
            valid = in_bounds & (x_off >= band_lo) & (x_off < band_hi)

            updated = orig_val + tl.maximum(v_prev, v_cur)
            final_val = tl.where(valid, updated, orig_val)
            tl.store(cur_ptr, final_val, mask=in_bounds)

    @triton.jit
    def __maximum_path_bwd_triton_kernel(
        value_ptr,
        path_ptr,
        t_y_ptr,
        t_x_ptr,
        Ty,
        stride_vb,
        stride_vy,
        stride_vx,
        stride_pb,
        stride_py,
        stride_px,
    ):
        """
        逆探索 (backtracking) を1バッチ要素=1プログラムで計算する。
        x 方向の並列性は無い (index はバッチ要素ごとのスカラー) が、
        バッチ全体はプログラム間で並列に実行される。
        """

        pid_b = tl.program_id(0)
        t_y = tl.load(t_y_ptr + pid_b)
        t_x = tl.load(t_x_ptr + pid_b)

        vbase = pid_b * stride_vb
        pbase = pid_b * stride_pb

        index = tl.maximum(t_x - 1, 0)

        for yy in range(0, Ty):
            y = Ty - 1 - yy
            active = y < t_y

            write_val = tl.where(active, 1, 0)
            p_ptr = path_ptr + pbase + y * stride_py + index * stride_px
            tl.store(p_ptr, write_val)

            if y > 0:
                va_ptr = value_ptr + vbase + (y - 1) * stride_vy + index * stride_vx
                idxm1_safe = tl.maximum(index - 1, 0)
                vb_ptr = (
                    value_ptr + vbase + (y - 1) * stride_vy + idxm1_safe * stride_vx
                )
                v_a = tl.load(va_ptr)
                v_b = tl.load(vb_ptr)

                nz = index != 0
                eq = index == y
                lt = v_a < v_b
                should_dec = active & nz & (eq | lt)
                index = tl.where(should_dec, index - 1, index)

    @torch.no_grad()
    def _maximum_path_gpu_triton(
        neg_cent: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        _maximum_path_gpu と全く同じ結果を、forward 1回・backward 1回、
        合計2回のカーネル起動だけで計算する Triton 実装。

        _maximum_path_gpu では y の Python ループの1回分ごとに
        (clone / slice代入 / 比較 / where など) 数個の CUDA カーネルが
        起動されていたが、こちらは y についてのループそのものを Triton
        カーネル内部のループとして実行するため、系列長 (Ty) の長さに
        関わらずカーネル起動は forward/backward 合わせて2回で済む。
        """

        device = neg_cent.device
        orig_dtype = neg_cent.dtype
        b, ty_max, tx_max = neg_cent.shape

        value = neg_cent.detach().to(torch.float32).contiguous().clone()
        t_y = mask.sum(dim=1)[:, 0].round().to(torch.int32).contiguous()
        t_x = mask.sum(dim=2)[:, 0].round().to(torch.int32).contiguous()

        block_x = triton.next_power_of_2(max(tx_max, 1))

        __maximum_path_fwd_triton_kernel[(b,)](
            value,
            t_y,
            t_x,
            ty_max,
            tx_max,
            value.stride(0),
            value.stride(1),
            value.stride(2),
            BLOCK_X=block_x,
        )

        path = torch.zeros((b, ty_max, tx_max), dtype=torch.int32, device=device)
        __maximum_path_bwd_triton_kernel[(b,)](
            value,
            path,
            t_y,
            t_x,
            ty_max,
            value.stride(0),
            value.stride(1),
            value.stride(2),
            path.stride(0),
            path.stride(1),
            path.stride(2),
        )

        return path.to(dtype=orig_dtype)
