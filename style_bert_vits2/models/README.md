# monotonic_align_cuda

Numba JIT 実装の MAS (Monotonic Alignment Search) を  
CUDA カーネルに置き換えるモジュールです。

## 何が変わるか

| 項目 | 元の実装 (Numba) | CUDA カーネル版 |
|------|-----------------|----------------|
| 実行場所 | CPU | GPU |
| GPU↔CPU転送 | 毎ステップ発生 | **ゼロ** |
| バッチ並列性 | なし（逐次ループ） | あり（ブロック並列）|
| JITコンパイル遅延 | 初回起動時に発生 | なし |

## ビルド手順

```bash
cd monotonic_align_cuda
python setup.py build_ext --inplace
```

ビルド後、以下のファイルが生成されます：

```
monotonic_align_cuda_core.cpython-3XX-linux-gnu.so
```

## 配置

生成された `.so` と `monotonic_alignment.py` を  
元の `style_bert_vits2/models/` に配置します。

```
style_bert_vits2/
└── models/
    ├── monotonic_alignment.py          ← このファイルで上書き
    └── monotonic_align_cuda_core.so    ← ビルドした .so を配置
```

または `.so` をプロジェクトルートに置いても動作します  
（Python パスが通っていれば `importlib.import_module` で発見されます）。

## 動作確認

```bash
python test_monotonic_alignment.py
```

以下が出力されれば成功です：

```
正当性テスト
  出力一致: True
  → OK

速度比較  (B=16, T_y=300, T_x=100)
  Numba JIT (元実装)          : XX.XXX ms/iter
  CUDA カーネル               : XX.XXX ms/iter
  速度向上: X.XXx
```

## フォールバック動作

`.so` が見つからない場合、自動的に Numba JIT にフォールバックします。  
ビルドしていない環境でも学習は継続できます。

## GPUアーキテクチャの設定

`setup.py` の `extra_compile_args` に対象 GPU の sm コードを追記してください：

| GPU世代 | コード |
|---------|--------|
| RTX 20xx (Turing) | `sm_75` |
| RTX 30xx / A100 (Ampere) | `sm_80`, `sm_86` |
| RTX 40xx (Ada) | `sm_89` |
| H100 (Hopper) | `sm_90` |

デフォルトでは sm_75〜sm_90 がすべて含まれています。  
不要なものを削除するとビルドが速くなります。
