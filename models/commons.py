"""
以下に記述されている関数のコメントはリファクタリング時に GPT-4 に生成させたもので、
コードと完全に一致している保証はない。あくまで参考程度とすること。
"""

from typing import Any, Optional, Union

import torch
from torch.nn import functional as F


def make_adamw_optimizer(
    parameters: Any,
    lr: float,
    betas: tuple[float, float],
    eps: float,
) -> torch.optim.AdamW:
    """
    可能なら fused AdamW（パラメータ更新を融合した CUDA カーネルで行う実装）を
    使って AdamW オプティマイザを作成する。

    fused=True は通常の（foreach/単純ループの）AdamW と数学的に同一の更新式を
    使う PyTorch 公式の実装であり、学習結果を変えるものではない。パラメータ数
    だけ小さな更新カーネルを個別に起動する代わりに融合したカーネルで処理する
    ため、特にパラメータ数の多いネットワーク（net_g 等）でカーネル起動
    オーバーヘッドを削減できる（＝GPU 使用率の向上に寄与する）。GradScaler は
    fused optimizer を認識して正しく扱えるため、`scaler.unscale_()` を明示的に
    呼んでから `scaler.step()` する既存フローともそのまま組み合わせられる。

    PyTorch バージョンや環境によっては fused がサポートされない可能性がある
    ため、実際に小さなダミーパラメータで動作確認できた場合のみ使用し、
    だめであれば黙って通常の実装にフォールバックする（学習が落ちないことを
    優先する）。
    """
    parameters = list(parameters)
    fused_available = False
    if len(parameters) > 0 and parameters[0].is_cuda:
        try:
            probe = torch.nn.Parameter(
                torch.zeros(1, device=parameters[0].device, dtype=torch.float32)
            )
            probe.grad = torch.zeros_like(probe)
            torch.optim.AdamW(
                [probe], lr=lr, betas=betas, eps=eps, fused=True
            ).step()
            fused_available = True
        except Exception:
            fused_available = False
    if fused_available:
        try:
            return torch.optim.AdamW(parameters, lr, betas=betas, eps=eps, fused=True)
        except Exception:
            pass
    return torch.optim.AdamW(parameters, lr, betas=betas, eps=eps)


def init_weights(m: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """
    モジュールの重みを初期化する

    Args:
        m (torch.nn.Module): 重みを初期化する対象のモジュール
        mean (float): 正規分布の平均
        std (float): 正規分布の標準偏差
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """
    カーネルサイズと膨張率からパディングの大きさを計算する

    Args:
        kernel_size (int): カーネルのサイズ
        dilation (int): 膨張率

    Returns:
        int: 計算されたパディングの大きさ
    """
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape: list[list[Any]]) -> list[Any]:
    """
    パディングの形状を変換する

    Args:
        pad_shape (list[list[Any]]): 変換前のパディングの形状

    Returns:
        list[Any]: 変換後のパディングの形状
    """
    layer = pad_shape[::-1]
    new_pad_shape = [item for sublist in layer for item in sublist]
    return new_pad_shape


def intersperse(lst: list[Any], item: Any) -> list[Any]:
    """
    リストの要素の間に特定のアイテムを挿入する

    Args:
        lst (list[Any]): 元のリスト
        item (Any): 挿入するアイテム

    Returns:
        list[Any]: 新しいリスト
    """
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def slice_segments(
    x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4
) -> torch.Tensor:
    """
    テンソルからセグメントをスライスする

    Args:
        x (torch.Tensor): 入力テンソル
        ids_str (torch.Tensor): スライスを開始するインデックス
        segment_size (int, optional): スライスのサイズ (デフォルト: 4)

    Returns:
        torch.Tensor: スライスされたセグメント
    """
    gather_indices = ids_str.view(x.size(0), 1, 1).repeat(
        1, x.size(1), 1
    ) + torch.arange(segment_size, device=x.device)
    return torch.gather(x, 2, gather_indices)


def rand_slice_segments(
    x: torch.Tensor, x_lengths: Optional[torch.Tensor] = None, segment_size: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ランダムなセグメントをスライスする

    Args:
        x (torch.Tensor): 入力テンソル
        x_lengths (Optional[torch.Tensor], optional): 各バッチの長さ (デフォルト: None)
        segment_size (int, optional): スライスのサイズ (デフォルト: 4)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: スライスされたセグメントと開始インデックス
    """
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t  # type: ignore
    ids_str_max = torch.clamp(x_lengths - segment_size + 1, min=0)  # type: ignore
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def subsequent_mask(length: int) -> torch.Tensor:
    """
    後続のマスクを生成する

    Args:
        length (int): マスクのサイズ

    Returns:
        torch.Tensor: 生成されたマスク
    """
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


@torch.jit.script  # type: ignore
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor, input_b: torch.Tensor, n_channels: torch.Tensor
) -> torch.Tensor:
    """
    加算、tanh、sigmoid の活性化関数を組み合わせた演算を行う

    Args:
        input_a (torch.Tensor): 入力テンソル A
        input_b (torch.Tensor): 入力テンソル B
        n_channels (torch.Tensor): チャネル数

    Returns:
        torch.Tensor: 演算結果
    """
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def sequence_mask(
    length: torch.Tensor, max_length: Optional[int] = None
) -> torch.Tensor:
    """
    シーケンスマスクを生成する

    Args:
        length (torch.Tensor): 各シーケンスの長さ
        max_length (Optional[int]): 最大のシーケンス長さ。指定されていない場合は length の最大値を使用

    Returns:
        torch.Tensor: 生成されたシーケンスマスク
    """
    if max_length is None:
        max_length = length.max()  # type: ignore
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)  # type: ignore
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    パスを生成する

    Args:
        duration (torch.Tensor): 各時間ステップの持続時間
        mask (torch.Tensor): マスクテンソル

    Returns:
        torch.Tensor: 生成されたパス
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def clip_grad_value_(
    parameters: Union[torch.Tensor, list[torch.Tensor]],
    clip_value: Optional[float],
    norm_type: float = 2.0,
) -> float:
    """
    勾配の値をクリップする

    Args:
        parameters (Union[torch.Tensor, list[torch.Tensor]]): クリップするパラメータ
        clip_value (Optional[float]): クリップする値。None の場合はクリップしない
        norm_type (float): ノルムの種類

    Returns:
        float: 総ノルム

    Note:
        元実装はパラメータ数だけ Python ループを回し、`.item()` を1テンソルずつ
        呼び出していたため、そのたびに GPU-CPU 間の同期が発生していた（ネットワーク
        1つあたり数百回になることもある）。特にステップ時間の短い高速な GPU では、
        この同期待ちがボトルネックになりやすい。
        `torch._foreach_norm` / `torch._foreach_clamp_*` を使い、同期を関数全体で
        高々1回（`.item()` 1回）に削減する。値は元実装と数学的に同一（クランプ処理は
        bit-exact、ノルムは元実装が Python の float（=倍精度）で累積していたのに
        合わせて倍精度で合算するため、実用上 bit-exact 相当）。
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return 0.0

    grads = [p.grad for p in parameters]

    if clip_value is not None:
        clip_value = float(clip_value)
        # p.grad.data.clamp_(min=-clip_value, max=clip_value) と bit-exact
        torch._foreach_clamp_min_(grads, -clip_value)
        torch._foreach_clamp_max_(grads, clip_value)

    if norm_type == float("inf"):
        per_tensor_norms = [g.detach().abs().max() for g in grads]
        total_norm = torch.stack(per_tensor_norms).max().double()
    else:
        # torch._foreach_norm は device/dtype が揃ったテンソル群でのみ動作するため、
        # 念のためグルーピングしてから呼び出す（通常このリポジトリでは全て同一
        # device/dtype の float32 なので、実際には1グループにまとまる）。
        groups: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
        for g in grads:
            groups.setdefault((g.device, g.dtype), []).append(g)
        per_tensor_norms = []
        for group_grads in groups.values():
            per_tensor_norms.extend(torch._foreach_norm(group_grads, norm_type))
        # 倍精度で合算することで、元実装（Python float = 倍精度で累積）と
        # 同じ丸め誤差になるようにする。
        stacked = torch.stack([n.double() for n in per_tensor_norms])
        total_norm = stacked.pow(norm_type).sum().pow(1.0 / norm_type)

    # ここで初めて（関数全体で1回だけ）GPU-CPU 同期が発生する。
    return total_norm.item()
