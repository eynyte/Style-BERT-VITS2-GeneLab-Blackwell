"""
TPU (torch_xla) 推論を有効にするための小さなユーティリティ集。

device 文字列として "tpu" または "xla" が指定されたときにのみ関与し、
"cpu" / "cuda" / "cuda:0" / "mps" などの既存の挙動には一切影響を与えない
(=このモジュールを import しただけでは何も変わらない)。

## TPU 推論を使う上で最も重要な注意点

Style-Bert-VITS2 (VITS2 系モデル) の推論は、入力テキストの音素数や、
Duration Predictor が予測する発話の長さによって、内部テンソルの形状
(音素方向の長さ T_x、生成フレーム数 T_y) が呼び出しごとに変わる。

XLA はテンソル形状ごとに計算グラフをコンパイルするため、何も対策しないと
「一文合成するたびに新しい形状 → 毎回 XLA の再コンパイルが走る」状態になり、
GPU/CPU よりもむしろ遅くなることがある (TPU 実機ではコンパイルに数十秒〜
数分かかることもある)。

これを避けるため、本モジュールは形状を「バケツ」(bucket) と呼ばれる
あらかじめ決められた長さの集合に丸め込むための関数を提供する。
同じバケツに収まる入力同士は同じ形状のグラフを再利用できるため、
2 回目以降の呼び出しはコンパイル待ちなしで実行できる。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Optional, Union

import torch

from style_bert_vits2.logging import logger


# ユーザー向けのエイリアスとして "tpu" を受け付けつつ、torch/torch_xla 側の
# 実際のデバイス種別名である "xla" もそのまま受け付ける。
_XLA_DEVICE_STRINGS = ("tpu", "xla")


def is_xla_device(device: Union[str, torch.device, None]) -> bool:
    """device 指定が TPU (torch_xla) を指しているかどうかを判定する。"""

    if device is None:
        return False
    if isinstance(device, torch.device):
        return device.type == "xla"
    return str(device).lower() in _XLA_DEVICE_STRINGS


def resolve_device(device: Union[str, torch.device]) -> Union[str, torch.device]:
    """
    device 指定を、実際に Tensor.to() / Module.to() / torch.device() に渡せる値に変換する。

    - "tpu" / "xla" が指定された場合: torch_xla を import し、xm.xla_device() で
      取得した実デバイスオブジェクトを返す。
    - それ以外 ("cpu" / "cuda" / "cuda:0" / "mps" など): 何もせずそのまま返す
      (既存の CPU/GPU 向けの挙動を一切変えないため)。

    Raises:
        ImportError: "tpu"/"xla" が指定されたが torch_xla がインストールされていない場合。
    """

    if not is_xla_device(device):
        return device
    try:
        import torch_xla.core.xla_model as xm
    except ImportError as e:
        raise ImportError(
            "device='tpu' (または 'xla') で推論するには torch_xla が必要です。\n"
            "  pip install torch_xla\n"
            "でインストールしてください。torch_xla のバージョンは、利用している "
            "PyTorch のバージョンと一致させる必要があります "
            "(例: torch==2.9.0 を使っているなら torch_xla==2.9.0)。\n"
            "また、TPU VM 上で実行する際は環境変数 PJRT_DEVICE=TPU の設定が必要です。"
        ) from e
    return xm.xla_device()


def mark_step(device: Union[str, torch.device, None] = None) -> None:
    """
    torch_xla の xm.mark_step() の薄いラッパー。

    torch_xla は演算を即座に実行せず、計算グラフとして溜め込んでから
    まとめて実行する (lazy tensor)。mark_step() を呼ぶとその時点までの
    グラフが確定・実行される。`.cpu()` や `.item()` を呼んだ時点でも
    暗黙的に実行されるため必須ではないが、計測やメモリ管理の目的で
    明示的に区切りたい場合に使う。

    device が TPU/XLA でない場合、または torch_xla がロードされていない場合は
    何もしない (no-op) ので、CPU/GPU 環境でも安全に呼び出せる。
    """

    if device is not None and not is_xla_device(device):
        return
    try:
        import torch_xla.core.xla_model as xm
    except ImportError:
        return
    xm.mark_step()


def pick_bucket(true_length: int, buckets: Optional[Sequence[int]]) -> Optional[int]:
    """
    buckets の中から true_length 以上で最小のものを選ぶ。

    Args:
        true_length (int): 実際に必要な長さ
        buckets (Optional[Sequence[int]]): 候補となるバケツ長のリスト (順不同で可)

    Returns:
        Optional[int]: 選ばれたバケツ長。buckets が空、または true_length が
            最大のバケツより大きい場合は None
            (=バケツ化を諦めて実測値をそのまま使うべき、というシグナル)。
    """

    if not buckets:
        return None
    candidates = [b for b in buckets if b >= true_length]
    if not candidates:
        return None
    return min(candidates)


def pad_last_dim(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    """tensor の最後の次元 (時間軸) を target_length まで 0 埋めする。既に長ければ何もしない。"""

    current_length = tensor.shape[-1]
    if current_length >= target_length:
        return tensor
    pad_amount = target_length - current_length
    return torch.nn.functional.pad(tensor, (0, pad_amount))


def _round_up(value: int, multiple: int) -> int:
    """value を multiple の倍数に切り上げる (TPU のタイル幅に形状を合わせるため)。"""

    if multiple <= 1 or value <= 0:
        return value
    remainder = value % multiple
    if remainder == 0:
        return value
    return value + (multiple - remainder)


def default_input_length_buckets() -> list[int]:
    """
    音素(テキスト)側の入力長バケツのデフォルト値 (8 の倍数に丸め済み)。
    line_split 等で 1 文単位に区切って合成する運用を想定し、
    短い相槌から ~400 音素程度の長い文章までをカバーする、対数的な間隔の目安値。
    テキストの長さの分布が偏っている場合は、自前のリストを渡す方がよい。
    """

    raw = [16, 24, 32, 48, 64, 96, 128, 176, 240, 320, 400]
    return sorted({_round_up(v, 8) for v in raw})


def default_output_length_buckets(hps: Any) -> list[int]:
    """
    出力フレーム (mel 相当の時間軸) 側の長さバケツのデフォルト値 (8 の倍数に丸め済み)。
    hps.data.sampling_rate / hps.data.hop_length から、およそ 1〜20 秒の音声に
    相当するフレーム数を逆算する。極端に長い合成 (20 秒超) を行う場合は
    自前でより大きな値を含むリストを渡すこと。
    """

    sampling_rate = hps.data.sampling_rate
    hop_length = hps.data.hop_length
    seconds = [1, 2, 3, 5, 8, 12, 16, 20]
    frames = [
        _round_up(int(math.ceil(s * sampling_rate / hop_length)), 8) for s in seconds
    ]
    return sorted(set(frames))


def warn_bucket_overflow(kind: str, true_length: int, buckets: Sequence[int]) -> None:
    """指定した長さがどのバケツにも収まらなかった場合の警告ログを出す共通ヘルパー。"""

    logger.warning(
        f"[TPU] {kind}の実測長 {true_length} が用意されたバケツの最大値 "
        f"{max(buckets)} を超えています。この呼び出しはバケツ化されず実測値の "
        "形状で実行されるため、XLA の新規コンパイルが発生し遅くなります。"
        "頻発する場合は、より大きな値を含む長さバケツを指定してください。"
    )
