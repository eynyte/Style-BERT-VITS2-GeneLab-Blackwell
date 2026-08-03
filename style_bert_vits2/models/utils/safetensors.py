from pathlib import Path
from typing import Any, Optional, Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from style_bert_vits2.logging import logger
from style_bert_vits2.xla import is_xla_device


def load_safetensors(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    for_infer: bool = False,
    device: Union[str, torch.device] = "cpu",
) -> tuple[torch.nn.Module, Optional[int]]:
    """
    指定されたパスから safetensors モデルを読み込み、モデルとイテレーションを返す。

    Args:
        checkpoint_path (Union[str, Path]): モデルのチェックポイントファイルのパス
        model (torch.nn.Module): 読み込む対象のモデル
        for_infer (bool): 推論用に読み込むかどうかのフラグ

    Returns:
        tuple[torch.nn.Module, Optional[int]]: 読み込まれたモデルとイテレーション回数（存在する場合）
    """

    # safetensors ライブラリ (Rust 実装) は "tpu"/"xla" というデバイス種別を
    # 認識できないため、TPU/XLA 向けには常に CPU 上へロードする。model 自体は
    # 呼び出し側で既に目的のデバイスへ .to() 済みなので、後段の
    # load_state_dict() の Tensor.copy_() が CPU→XLA への転送を正しく行う。
    # それ以外のデバイス (cpu/cuda/mps 等) では従来通り device をそのまま渡す
    # (cuda の場合は safetensors が直接 GPU へロードする最適化を維持できる)。
    open_device = "cpu" if is_xla_device(device) else device

    tensors: dict[str, Any] = {}
    iteration: Optional[int] = None
    with safe_open(str(checkpoint_path), framework="pt", device=open_device) as f:  # type: ignore
        for key in f.keys():
            if key == "iteration":
                iteration = f.get_tensor(key).item()
            tensors[key] = f.get_tensor(key)
    if hasattr(model, "module"):
        result = model.module.load_state_dict(tensors, strict=False)
    else:
        result = model.load_state_dict(tensors, strict=False)
    for key in result.missing_keys:
        if key.startswith("enc_q") and for_infer:
            continue
        logger.warning(f"Missing key: {key}")
    for key in result.unexpected_keys:
        if key == "iteration":
            continue
        logger.warning(f"Unexpected key: {key}")
    if iteration is None:
        logger.info(f"Loaded '{checkpoint_path}'")
    else:
        logger.info(f"Loaded '{checkpoint_path}' (iteration {iteration})")

    return model, iteration


def save_safetensors(
    model: torch.nn.Module,
    iteration: int,
    checkpoint_path: Union[str, Path],
    is_half: bool = False,
    for_infer: bool = False,
) -> None:
    """
    モデルを safetensors 形式で保存する。

    Args:
        model (torch.nn.Module): 保存するモデル
        iteration (int): イテレーション回数
        checkpoint_path (Union[str, Path]): 保存先のパス
        is_half (bool): モデルを半精度で保存するかどうかのフラグ
        for_infer (bool): 推論用に保存するかどうかのフラグ
    """

    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    keys = []
    for k in state_dict:
        if "enc_q" in k and for_infer:
            continue
        keys.append(k)

    new_dict = (
        {k: state_dict[k].half() for k in keys}
        if is_half
        else {k: state_dict[k] for k in keys}
    )
    new_dict["iteration"] = torch.LongTensor([iteration])
    logger.info(f"Saved safetensors to {checkpoint_path}")

    save_file(new_dict, checkpoint_path)
