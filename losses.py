import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel


# ── ① Python ループを sum() ジェネレータに統一し、F.l1_loss で CUDA カーネル起動を削減 ──
def feature_loss(fmap_r, fmap_g):
    loss = sum(
        F.l1_loss(gl.float(), rl.float().detach())
        for dr, dg in zip(fmap_r, fmap_g)
        for rl, gl in zip(dr, dg)
    )
    return loss * 2


# ── ② += による累算をテンソル加算に統一（Python int との型変換を排除） ──
def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss = loss + r_loss + g_loss   # += は Python int に戻る場合があるので明示的に
        r_losses.append(r_loss.detach())
        g_losses.append(g_loss.detach())

    return loss, r_losses, g_losses


# ── ③ リスト内包表記で GPU カーネルをまとめ、sum() でスカラー削減 ──
def generator_loss(disc_outputs):
    gen_losses = [torch.mean((1 - dg.float()) ** 2) for dg in disc_outputs]
    loss = sum(gen_losses)
    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    return kl / torch.sum(z_mask)


class WavLMLoss(torch.nn.Module):
    def __init__(self, model, wd, model_sr, slm_sr=16000):
        super(WavLMLoss, self).__init__()
        self.wavlm = AutoModel.from_pretrained(model)
        self.wd = wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)
        self.wavlm.eval()
        for param in self.wavlm.parameters():
            param.requires_grad = False

    def forward(self, wav, y_rec):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states

        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16, output_hidden_states=True
        ).hidden_states

        # ── ④ Python ループを排除: 全レイヤーを一括スタックして 1 カーネルで L1 計算 ──
        # [L, B, T, H] にスタック → dim=(1,2,3) で各レイヤーの平均 → レイヤー方向に sum
        wav_stack = torch.stack(wav_embeddings).detach()  # 勾配不要
        rec_stack = torch.stack(y_rec_embeddings)
        floss = torch.abs(wav_stack - rec_stack).mean(dim=(1, 2, 3)).sum()

        return floss

    def generator(self, y_rec):
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16, output_hidden_states=True
        ).hidden_states
        y_rec_embeddings = (
            torch.stack(y_rec_embeddings, dim=1)
            .transpose(-1, -2)
            .flatten(start_dim=1, end_dim=2)
        )
        y_df_hat_g = self.wd(y_rec_embeddings)
        return torch.mean((1 - y_df_hat_g) ** 2)

    def discriminator(self, wav, y_rec):
        with torch.no_grad():
            batch_size = wav.shape[0]

            # ── ⑤ リサンプルを 1 回の GPU オペレーションにバッチ処理 ──
            # wav と y_rec をバッチ次元で結合 → 1 カーネルで両方リサンプル
            combined_16 = self.resample(torch.cat([wav, y_rec], dim=0))

            # ── ⑥ WavLM フォワードパスを 1 回に統合 (カーネル起動を半減) ──
            # ※ メモリが厳しい場合は元の 2 回呼び出しに戻すこと
            combined_embeddings = self.wavlm(
                input_values=combined_16, output_hidden_states=True
            ).hidden_states

            # バッチ次元でスライスして分離 → stack/transpose/flatten を 1 度に実行
            y_embeddings = (
                torch.stack([e[:batch_size] for e in combined_embeddings], dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )
            y_rec_embeddings = (
                torch.stack([e[batch_size:] for e in combined_embeddings], dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        y_df_hat_r = self.wd(y_embeddings)
        y_df_hat_g = self.wd(y_rec_embeddings)

        r_loss = torch.mean((1 - y_df_hat_r) ** 2)
        g_loss = torch.mean(y_df_hat_g ** 2)

        return (r_loss + g_loss)

    def discriminator_forward(self, wav):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            y_embeddings = (
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        return self.wd(y_embeddings)
"""
コードを詳しく分析して、最適化バージョンを作成します。最適化のポイントを番号ごとに説明します。

---

### ① `feature_loss` — `F.l1_loss` + ジェネレータ `sum()`
```python
# Before
loss += torch.mean(torch.abs(rl - gl))

# After
loss = sum(F.l1_loss(gl.float(), rl.float().detach()) ...)
```
`F.l1_loss` は PyTorch 内部で最適化済みの CUDA カーネルを使用します。Python の `+=` を `sum()` ジェネレータに変えることで、各ステップで `int → Tensor` の暗黙変換が起きるオーバーヘッドも除去されます。

---

### ② `discriminator_loss` — 明示的なテンソル加算
```python
loss = loss + r_loss + g_loss  # ← loss += ... より安全
```
`loss = 0` (Python int) に `+=` すると型変換が発生する場合があります。明示的に書くことでこれを回避します。

---

### ③ `generator_loss` — リスト内包表記
Python ループより内包表記の方が若干オーバーヘッドが少なく、コードも読みやすくなります。

---

### ④ `WavLMLoss.forward` — 最大の効果: Python ループ撤廃
```python
# Before: L 回のカーネル起動（L = WavLM のレイヤー数、通常 13〜25）
for er, eg in zip(wav_embeddings, y_rec_embeddings):
    floss += torch.mean(torch.abs(er - eg))

# After: 1 回のカーネル起動
wav_stack = torch.stack(wav_embeddings).detach()  # [L, B, T, H]
rec_stack = torch.stack(y_rec_embeddings)
floss = torch.abs(wav_stack - rec_stack).mean(dim=(1, 2, 3)).sum()
```
L 回の小さいカーネルを 1 回の大きいカーネルに統合することで、GPU を効率よく利用できます。

---

### ⑤⑥ `WavLMLoss.discriminator` — バッチ化で WavLM 呼び出し回数を半減
```python
# Before: resample ×2 回, WavLM forward ×2 回
wav_16 = self.resample(wav)
y_rec_16 = self.resample(y_rec)
wav_embeddings = self.wavlm(wav_16, ...)
y_rec_embeddings = self.wavlm(y_rec_16, ...)

# After: resample ×1 回, WavLM forward ×1 回
combined_16 = self.resample(torch.cat([wav, y_rec], dim=0))
combined_embeddings = self.wavlm(combined_16, ...)
```
`discriminator` は全て `no_grad` ブロック内なので、`wav` と `y_rec` をバッチ次元で結合して一括処理できます。これにより、最もコストの高い WavLM のフォワードパスが **1 回で済みます**。

> **⚠️ 注意**: バッチサイズが実質 2 倍になるため、VRAM が厳しい場合は元の 2 回呼び出しに戻してください。その場合でも、リサンプルのバッチ化（⑤）だけは効果があります。
"""