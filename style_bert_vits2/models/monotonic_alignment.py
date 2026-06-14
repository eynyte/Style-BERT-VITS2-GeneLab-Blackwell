"""
style_bert_vits2/models/monotonic_alignment.py

Numba JIT 実装を CUDA カーネルに置き換えたバージョン。

動作優先順位:
  1. monotonic_align_cuda_core (CUDAカーネル) ← GPU上で完結、転送ゼロ
  2. Numba JIT (元の実装)                     ← GPU未使用 or ビルド未実施時のフォールバック

置き換え手順:
  1. monotonic_align_cuda/ ディレクトリで以下を実行
       python setup.py build_ext --inplace
  2. 生成された .so を Python のパスが通る場所に置く
     (プロジェクトルートか、このファイルと同じディレクトリが推奨)
  3. このファイルで元の monotonic_alignment.py を上書きする
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import logging
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from numpy import float32, int32, zeros

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CUDAカーネルのロード試行
# ---------------------------------------------------------------------------
_cuda_ext = None
_jit_compile_attempted = False
_cpu_fallback_warned = False

# .so が置かれているディレクトリ候補（このファイルと同じ場所 → プロジェクトルートの順）
_SO_SEARCH_DIRS: list[Path] = [
    Path(__file__).parent,
    Path(__file__).parent.parent,
    Path(__file__).parent.parent.parent,
]
_MODULE_NAME = "monotonic_align_cuda_core"


def _so_filename_for_current_python() -> str:
    """
    現在の Python インタープリタが要求する .so ファイル名を返す。

    例:
        CPython 3.12, Linux x86_64
            → monotonic_align_cuda_core.cpython-312-x86_64-linux-gnu.so
        CPython 3.11, Linux x86_64
            → monotonic_align_cuda_core.cpython-311-x86_64-linux-gnu.so
    """
    vi = sys.version_info
    pyver = f"{vi.major}{vi.minor}"                   # "312", "311", ...
    machine = platform.machine()                       # "x86_64", "aarch64", ...
    # Linux: cpython-312-x86_64-linux-gnu
    # macOS: cpython-312-darwin  (CUDAは非対応だが念のため)
    system = platform.system().lower()
    if system == "linux":
        tag = f"cpython-{pyver}-{machine}-linux-gnu"
    elif system == "darwin":
        tag = f"cpython-{pyver}-darwin"
    else:
        tag = f"cpython-{pyver}"
    return f"{_MODULE_NAME}.{tag}.so"


def _find_and_rename_so() -> Path | None:
    """
    .so ファイルを探し、現在の Python バージョンに合った名前に
    リネーム（コピー）して返す。

    探索ロジック:
      1. 現在の Python に合致する .so が既に存在する → そのまま返す
      2. 同名モジュール・別バージョンの .so が存在する → コピーして返す
      3. 見つからない → None を返す
    """
    target_name = _so_filename_for_current_python()

    for search_dir in _SO_SEARCH_DIRS:
        # ① 既に正しい名前のファイルがある
        target_path = search_dir / target_name
        if target_path.exists():
            return target_path

        # ② 別バージョン向けの .so が存在するか探す
        candidates = sorted(search_dir.glob(f"{_MODULE_NAME}.cpython-*.so"))
        if candidates:
            src = candidates[0]   # 複数あれば最初の1つを使用
            logger.info(
                f"[monotonic_alignment] {src.name} を検出しました。\n"
                f"  現在の Python ({sys.version_info.major}.{sys.version_info.minor}) "
                f"向けに {target_name} としてコピーします。"
            )
            try:
                shutil.copy2(src, search_dir / target_name)
                return search_dir / target_name
            except OSError as e:
                logger.warning(
                    f"[monotonic_alignment] コピーに失敗しました: {e}\n"
                    f"  元ファイルをそのままロードします。"
                )
                return src   # コピー失敗時は元ファイルで試みる

    return None


def _try_load_cuda_ext() -> Any:
    """
    ビルド済みCUDA拡張をロードする。
    現在の Python バージョンに合う .so がなければ自動リネームを試みる。
    失敗した場合は None を返してフォールバックに委ねる。
    """
    # まず通常の import を試みる（パスが通っていれば最速）
    try:
        ext = importlib.import_module(_MODULE_NAME)
        print(f"[monotonic_alignment] CUDAカーネル版をロードしました（通常import）。")
        return ext
    except ModuleNotFoundError:
        pass

    # 通常 import が失敗した場合、.so を手動で探してロードする
    so_path = _find_and_rename_so()
    if so_path is None:
        print(
            f"[monotonic_alignment] {_MODULE_NAME} の .so が見つかりません。\n"
            f"Numba JIT にフォールバックします。\n"
            f"ビルドするには: cd monotonic_align_cuda && python setup.py build_ext --inplace"
        )
        return None

    try:
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, so_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"spec の作成に失敗: {so_path}")
        ext = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = ext
        spec.loader.exec_module(ext)  # type: ignore[union-attr]
        print(f"[monotonic_alignment] CUDAカーネル版をロードしました: {so_path.name}")
        return ext
    except Exception as e:
        sys.modules.pop(_MODULE_NAME, None)
        print(f"[monotonic_alignment] .so のロードに失敗しました ({e})。Numba JIT にフォールバックします。")
        return None


def _try_jit_build_cuda_ext() -> Any:
    """
    ビルド済み .so が見つからない場合、core.cu を PyTorch の拡張キャッシュへ
    JIT ビルドしてロードする。Linux + CUDA Toolkit がある環境向け。
    """
    global _jit_compile_attempted
    if _jit_compile_attempted:
        return None
    _jit_compile_attempted = True

    core_path = Path(__file__).with_name("core.cu")
    if not core_path.exists():
        print(
            f"[monotonic_alignment] {core_path} が見つかりません。"
            "Numba JIT にフォールバックします。"
        )
        return None

    try:
        from torch.utils.cpp_extension import load

        sys.modules.pop(_MODULE_NAME, None)
        print(
            "[monotonic_alignment] CUDAカーネル版が未ビルドのため、"
            "core.cu をJITビルドします。初回のみ時間がかかります。"
        )
        ext = load(
            name=_MODULE_NAME,
            sources=[str(core_path)],
            extra_cuda_cflags=["--use_fast_math", "-O3"],
            verbose=False,
        )
        print("[monotonic_alignment] CUDAカーネル版をJITロードしました。")
        return ext
    except Exception as e:
        print(
            f"[monotonic_alignment] CUDAカーネル版のJITビルドに失敗しました ({e})。"
            "Numba JIT にフォールバックします。"
        )
        return None

_cuda_ext = _try_load_cuda_ext()


# ---------------------------------------------------------------------------
# 公開API: maximum_path
# ---------------------------------------------------------------------------

def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    与えられた負の中心とマスクを使用して最大パスを計算する。

    Args:
        neg_cent : [B, T_y, T_x]  float, GPU or CPU
        mask     : [B, T_y, T_x]  float, GPU or CPU

    Returns:
        path     : [B, T_y, T_x]  neg_cent と同じ dtype / device
    """
    global _cuda_ext, _cpu_fallback_warned
    if neg_cent.is_cuda:
        if _cuda_ext is None:
            _cuda_ext = _try_jit_build_cuda_ext()
        if _cuda_ext is not None:
            return _maximum_path_cuda(neg_cent, mask)
        if not _cpu_fallback_warned:
            _cpu_fallback_warned = True
            print(
                "[monotonic_alignment] 警告: CUDAカーネルが使えないため、"
                "MASでGPU↔CPU転送が発生します。H100/H200/B200では大きな"
                "ボトルネックになります。"
            )
    return _maximum_path_numba(neg_cent, mask)


# ---------------------------------------------------------------------------
# CUDA実装
# ---------------------------------------------------------------------------

def _maximum_path_cuda(
    neg_cent: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    CUDAカーネルを使ってGPU上でパスを計算する。
    GPU↔CPU転送は一切発生しない。
    """
    assert _cuda_ext is not None

    device = neg_cent.device
    dtype  = neg_cent.dtype
    B, T_y, T_x = neg_cent.shape

    # カーネルは float32 / int32 を要求する
    neg_cent_f32 = neg_cent.float().contiguous()

    # マスクから有効長を取得 (GPU上で完結)
    # mask: [B, T_y, T_x]
    # t_y_max[b] = mask[b, :, 0].sum()  → 有効フレーム数
    # t_x_max[b] = mask[b, 0, :].sum()  → 有効音素数
    t_y_max = mask[:, :, 0].sum(dim=1).to(torch.int32).contiguous()
    t_x_max = mask[:, 0, :].sum(dim=1).to(torch.int32).contiguous()

    # 出力バッファ (ゼロ初期化)
    path = torch.zeros(B, T_y, T_x, dtype=torch.int32, device=device)

    # CUDAカーネル呼び出し
    _cuda_ext.maximum_path_cuda(path, neg_cent_f32, t_y_max, t_x_max)

    # 呼び出し元が期待する dtype に戻す
    return path.to(dtype=dtype)


# ---------------------------------------------------------------------------
# Numba JITフォールバック (元の実装をそのまま保持)
# ---------------------------------------------------------------------------

try:
    import numba
    _numba_available = True
except ImportError:
    _numba_available = False


def _maximum_path_numba(
    neg_cent: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    元のNumba JIT実装。CUDAカーネルが使えないときのフォールバック。
    GPU↔CPU転送が発生する。
    """
    if not _numba_available:
        raise RuntimeError(
            "monotonic_align_cuda_core もNumbaも利用できません。\n"
            "どちらかをインストールしてください。"
        )

    device = neg_cent.device
    dtype  = neg_cent.dtype

    neg_cent_np = neg_cent.data.cpu().numpy().astype(float32)
    path_np     = zeros(neg_cent_np.shape, dtype=int32)

    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(int32)

    __maximum_path_jit(path_np, neg_cent_np, t_t_max, t_s_max)

    return torch.from_numpy(path_np).to(device=device, dtype=dtype)


if _numba_available:
    import numba as _numba

    @_numba.jit(
        _numba.void(
            _numba.int32[:, :, ::1],
            _numba.float32[:, :, ::1],
            _numba.int32[::1],
            _numba.int32[::1],
        ),
        nopython=True,
        nogil=True,
    )
    def __maximum_path_jit(
        paths: Any, values: Any, t_ys: Any, t_xs: Any
    ) -> None:
        b = paths.shape[0]
        max_neg_val = -1e9
        for i in range(int(b)):
            path  = paths[i]
            value = values[i]
            t_y   = t_ys[i]
            t_x   = t_xs[i]

            v_prev = v_cur = 0.0
            index  = t_x - 1

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
