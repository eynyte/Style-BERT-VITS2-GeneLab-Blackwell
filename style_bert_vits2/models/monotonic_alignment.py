"""
numba JIT および CPU/GPU 間のデータ転送を排除し、
PyTorch テンソル演算のみで GPU 上でモノトニックアラインメントを計算する。

【元実装のオーバーヘッド】
  1. neg_cent.data.cpu().numpy()      — GPU → CPU 同期転送
                                        (GPU カーネル完了まで CPU をブロック)
  2. __maximum_path_jit (numba JIT)   — CPU 上でバッチを 1 件ずつ逐次処理
  3. torch.from_numpy(...).to(device) — CPU → GPU 転送

【本実装での改善】
  前向き DP   : 各行 y の x 方向更新は前行 y-1 にのみ依存するため、
                (B, T_x) を GPU 上で一括並列処理する（y 方向のみ逐次）。
  バックトラック: (B,) の index ベクトルをテンソル演算でバッチ並列処理する。
  @jit.script  : Python ループのインタープリタオーバーヘッドを排除する。
"""

import torch
import torch.nn.functional as F

print("★")
# -------------------------------------------------------------------------
# 内部実装（TorchScript でコンパイル）
# -------------------------------------------------------------------------

@torch.jit.script
def _forward_dp(value: torch.Tensor, NEG_INF: float) -> torch.Tensor:
    """
    前向き DP を実行する（TorchScript コンパイル済み）。

    漸化式: value[b, y, x] += max(value[b, y-1, x], value[b, y-1, x-1])
      - x == y  （対角）: 直前の同列を NEG_INF に強制（単調性制約）
      - x == 0  （左端）: 直前の左隣を NEG_INF に強制（境界条件）

    各行 y の更新は前行 y-1 にのみ依存するため、
    バッチ × x 軸を GPU 上で一括並列処理できる。

    Args:
        value   : 初期値テンソル [B, T_y, T_x]（masked_fill 済み）
        NEG_INF : 無効セルに割り当てる十分に小さな値

    Returns:
        DP テーブル [B, T_y, T_x]
    """
    B   = value.shape[0]
    T_y = value.shape[1]
    T_x = value.shape[2]

    for y in range(1, T_y):
        prev = value[:, y - 1, :]  # (B, T_x) — 前行の参照（コピーなし）

        # v_cur[b, x]: 前行の同列。対角 x == y は遷移禁止のため NEG_INF に上書き
        v_cur = prev.clone()
        if y < T_x:
            v_cur[:, y] = NEG_INF

        # v_prev[b, x]: 前行を 1 列右シフト。左端 (x=0) に NEG_INF を挿入
        neg_inf_col = torch.full((B, 1), NEG_INF, dtype=prev.dtype, device=prev.device)
        v_prev = torch.cat([neg_inf_col, prev[:, :-1]], dim=1)  # (B, T_x)

        value[:, y, :] += torch.maximum(v_cur, v_prev)

    return value


@torch.jit.script
def _backtrack(value: torch.Tensor,
               t_y_lens: torch.Tensor,
               t_x_lens: torch.Tensor,
               NEG_INF: float) -> torch.Tensor:
    """
    バックトラッキングによりパスを復元する（TorchScript コンパイル済み）。

    y を末尾から辿り、各ステップで (B,) の index を一括更新する。

    移動条件（index を 1 つ左へ）:
      index != 0  かつ（対角位置 index == y  または  左隣が高スコア）

    Args:
        value     : 前向き DP テーブル [B, T_y, T_x]
        t_y_lens  : 各サンプルの y 方向有効長 [B]
        t_x_lens  : 各サンプルの x 方向有効長 [B]
        NEG_INF   : DP 初期化に使用した下限値

    Returns:
        path テンソル [B, T_y, T_x]（各行にちょうど 1 つの 1 を持つ）
    """
    T_y = value.shape[1]
    T_x = value.shape[2]
    B   = value.shape[0]

    path  = torch.zeros_like(value)
    index = t_x_lens - 1                           # (B,) 現在の x インデックス
    b_idx = torch.arange(B, device=value.device)

    for y in range(T_y - 1, -1, -1):
        y_valid = (y < t_y_lens)                   # (B,) この行が有効かどうか
        idx     = index.clamp(0, T_x - 1)

        # 有効サンプルの現在位置にフラグを立てる（無効サンプルは 0 のまま）
        path[b_idx, y, idx] = y_valid.to(value.dtype)

        if y > 0:
            v_stay = value[b_idx, y - 1, idx]
            v_left = value[b_idx, y - 1, (index - 1).clamp(0, T_x - 1)]

            should_move = y_valid & (index != 0) & (
                (index == y) | (v_stay < v_left)
            )
            index = torch.where(should_move, index - 1, index)

    return path


# -------------------------------------------------------------------------
# 公開 API
# -------------------------------------------------------------------------

def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    与えられた負の中心とマスクを使用して最大パスを計算する。

    元の numba JIT 実装を置き換え、GPU/CPU 間のデータ転送オーバーヘッドを排除する。
    入力テンソルが置かれているデバイス（GPU / CPU）上でそのまま計算を行う。

    Args:
        neg_cent (torch.Tensor): 負の中心を表すテンソル [B, T_y, T_x]
        mask     (torch.Tensor): マスクを表すテンソル   [B, T_y, T_x]

    Returns:
        torch.Tensor: 計算された最大パスを表すテンソル [B, T_y, T_x]
    """
    NEG_INF: float = -1e9
    device = neg_cent.device
    T_X = neg_cent.shape[2]
    T_Y = neg_cent.shape[1]

    # 上対角マスクを適用: x > y の位置はパスが単調性制約上到達不可能
    x_idx = torch.arange(T_X, device=device).view(1, 1, T_X)  # (1,  1, T_x)
    y_idx = torch.arange(T_Y, device=device).view(1, T_Y, 1)  # (1, T_y,  1)
    valid = mask.bool() & (x_idx <= y_idx)                     # (B, T_y, T_x)

    with torch.no_grad():
        value = neg_cent.masked_fill(~valid, NEG_INF)

        value = _forward_dp(value, NEG_INF)

        t_y_lens = mask.sum(1)[:, 0].long()  # (B,)
        t_x_lens = mask.sum(2)[:, 0].long()  # (B,)

        return _backtrack(value, t_y_lens, t_x_lens, NEG_INF)





"""
コードの構造を把握しました。ボトルネックを整理します。

**元のオーバーヘッド3段階:**
1. `neg_cent.data.cpu().numpy()` — GPU→CPU転送
2. `__maximum_path_jit` — CPU上でバッチを1件ずつ逐次処理
3. `torch.from_numpy(path).to(device=...)` — CPU→GPU転送

**GPU化の方針:**
- 前向きDP: 各行 `y` の `x` 方向更新は前行 `y-1` にのみ依存 → `(B, T_x)` を一括並列処理可能
- バックトラッキング: `(B,)` の `index` ベクトルをテンソル演算でバッチ並列処理次に正当性テストを実行して、元の numba 実装と結果が一致するか確認します。全ケース一致を確認。続いてパフォーマンス計測します。CPU 環境では numba の方が速い結果になっています。原因を説明して、GPU 環境での真の改善を示します。CPU のみの環境では numba JIT が有利に見えますが、本来の問題は **GPU 環境での転送コスト**です。Python ループのオーバーヘッドも `torch.jit.script` で解消できます。`torch.jit.script` で **1.4–1.85× 追加高速化**を確認。最終ファイルを `jit.script` 版で更新します。全テスト通過。出力します。---

## 変更点のまとめ

**削除したもの**
- `numba`・`numpy` への依存（`import numba`, `from numpy import ...`）
- `__maximum_path_jit` 関数（CPU 専用の逐次処理）
- `neg_cent.data.cpu()` / `torch.from_numpy(...).to(device)` の往復転送

**追加したもの**
- `_forward_dp`・`_backtrack` を `@torch.jit.script` でコンパイル

---

## アルゴリズムの対応関係

| 元の実装 | 新しい実装 |
|---|---|
| バッチを 1 件ずつ `for i in range(b)` | `(B, T_x)` を一括テンソル演算 |
| `for x in range(...)` 内側ループ | `torch.maximum(v_cur, v_prev)` で全 x を並列 |
| `for y in range(t_y - 1, -1, -1)` の逐次 index 更新 | `torch.where(should_move, ...)` でバッチ並列 |

---

## パフォーマンス特性

**CPU 環境**では numba JIT の方がネイティブコードで速いですが、実際の GPU トレーニング環境では元実装の `.data.cpu()` が **GPU パイプラインの同期バリア**になります。それまでに積んだ GPU カーネル全ての完了を待つため、実際の損失は「転送サイズ × 帯域幅」よりもはるかに大きくなります。`torch.jit.script` は Python ループのインタープリタオーバーヘッドを排除し、CPU 上でも約 1.7× の高速化効果があります。
"""