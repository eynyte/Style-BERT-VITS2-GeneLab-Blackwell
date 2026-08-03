from collections.abc import Sequence
from typing import Any, Optional, Union, cast

import torch
from numpy.typing import NDArray

from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons, utils
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.models import SynthesizerTrn
from style_bert_vits2.models.models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)
from style_bert_vits2.nlp import (
    clean_text_with_given_phone_tone,
    cleaned_text_to_sequence,
    extract_bert_feature,
)
from style_bert_vits2.nlp.symbols import SYMBOLS
from style_bert_vits2.xla import (
    default_input_length_buckets,
    default_output_length_buckets,
    is_xla_device,
    mark_step,
    pad_last_dim,
    pick_bucket,
    resolve_device,
    warn_bucket_overflow,
)


def get_net_g(
    model_path: str, version: str, device: str, hps: HyperParameters
) -> Union[SynthesizerTrn, SynthesizerTrnJPExtra]:
    # "tpu"/"xla" が指定された場合、ここで実際の torch_xla デバイスオブジェクトに変換する。
    # "cpu"/"cuda"/"mps" 等はそのまま返るため、以降の挙動は一切変わらない。
    resolved_device = resolve_device(device)
    if version.endswith("JP-Extra"):
        logger.info("Using JP-Extra model")
        net_g = SynthesizerTrnJPExtra(
            n_vocab=len(SYMBOLS),
            spec_channels=hps.data.filter_length // 2 + 1,
            segment_size=hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            # hps.model 以下のすべての値を引数に渡す
            use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
            use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
            use_mel_posterior_encoder=hps.model.use_mel_posterior_encoder,
            use_duration_discriminator=hps.model.use_duration_discriminator,
            use_wavlm_discriminator=hps.model.use_wavlm_discriminator,
            inter_channels=hps.model.inter_channels,
            hidden_channels=hps.model.hidden_channels,
            filter_channels=hps.model.filter_channels,
            n_heads=hps.model.n_heads,
            n_layers=hps.model.n_layers,
            kernel_size=hps.model.kernel_size,
            p_dropout=hps.model.p_dropout,
            resblock=hps.model.resblock,
            resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
            resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
            upsample_rates=hps.model.upsample_rates,
            upsample_initial_channel=hps.model.upsample_initial_channel,
            upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
            n_layers_q=hps.model.n_layers_q,
            use_spectral_norm=hps.model.use_spectral_norm,
            gin_channels=hps.model.gin_channels,
            slm=hps.model.slm,
        ).to(resolved_device)
    else:
        logger.info("Using normal model")
        net_g = SynthesizerTrn(
            n_vocab=len(SYMBOLS),
            spec_channels=hps.data.filter_length // 2 + 1,
            segment_size=hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            # hps.model 以下のすべての値を引数に渡す
            use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
            use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
            use_mel_posterior_encoder=hps.model.use_mel_posterior_encoder,
            use_duration_discriminator=hps.model.use_duration_discriminator,
            use_wavlm_discriminator=hps.model.use_wavlm_discriminator,
            inter_channels=hps.model.inter_channels,
            hidden_channels=hps.model.hidden_channels,
            filter_channels=hps.model.filter_channels,
            n_heads=hps.model.n_heads,
            n_layers=hps.model.n_layers,
            kernel_size=hps.model.kernel_size,
            p_dropout=hps.model.p_dropout,
            resblock=hps.model.resblock,
            resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
            resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
            upsample_rates=hps.model.upsample_rates,
            upsample_initial_channel=hps.model.upsample_initial_channel,
            upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
            n_layers_q=hps.model.n_layers_q,
            use_spectral_norm=hps.model.use_spectral_norm,
            gin_channels=hps.model.gin_channels,
            slm=hps.model.slm,
        ).to(resolved_device)
    net_g.state_dict()
    _ = net_g.eval()
    if model_path.endswith(".pth") or model_path.endswith(".pt"):
        _ = utils.checkpoints.load_checkpoint(
            model_path, net_g, None, skip_optimizer=True, device=resolved_device
        )
    elif model_path.endswith(".safetensors"):
        _ = utils.safetensors.load_safetensors(
            model_path, net_g, True, device=resolved_device
        )
    else:
        raise ValueError(f"Unknown model format: {model_path}")
    return net_g


def get_text(
    text: str,
    language_str: Languages,
    hps: HyperParameters,
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    use_jp_extra = hps.version.endswith("JP-Extra")
    norm_text, phone, tone, word2ph = clean_text_with_given_phone_tone(
        text,
        language_str,
        given_phone=given_phone,
        given_tone=given_tone,
        use_jp_extra=use_jp_extra,
        # 推論時のみ呼び出されるので、raise_yomi_error は False に設定
        raise_yomi_error=False,
    )
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert_ori = extract_bert_feature(
        norm_text,
        word2ph,
        language_str,
        device,
        assist_text,
        assist_text_weight,
    )
    del word2ph
    assert bert_ori.shape[-1] == len(phone), phone

    if language_str == Languages.ZH:
        bert = bert_ori
        ja_bert = torch.zeros(1024, len(phone))
        en_bert = torch.zeros(1024, len(phone))
    elif language_str == Languages.JP:
        bert = torch.zeros(1024, len(phone))
        ja_bert = bert_ori
        en_bert = torch.zeros(1024, len(phone))
    elif language_str == Languages.EN:
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(1024, len(phone))
        en_bert = bert_ori
    else:
        raise ValueError("language_str should be ZH, JP or EN")

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, en_bert, phone, tone, language


def infer(
    text: str,
    style_vec: NDArray[Any],
    sdp_ratio: float,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
    sid: int,  # In the original Bert-VITS2, its speaker_name: str, but here it's id
    language: Languages,
    hps: HyperParameters,
    net_g: Union[SynthesizerTrn, SynthesizerTrnJPExtra],
    device: str,
    skip_start: bool = False,
    skip_end: bool = False,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
    xla_input_buckets: Optional[Sequence[int]] = None,
    xla_output_buckets: Optional[Sequence[int]] = None,
) -> NDArray[Any]:
    """
    Args (TPU/XLA 関連のみ抜粋):
        xla_input_buckets (Optional[Sequence[int]]): device が "tpu"/"xla" のとき、
            入力音素列をこの中から実際の長さ以上で最小の値までゼロ埋めしてから
            推論する。同じバケツに収まる文同士は XLA のコンパイル結果を使い回せる
            ため、2 回目以降が高速になる。省略時 (None) は device が TPU/XLA なら
            style_bert_vits2.xla.default_input_length_buckets() の値が自動的に
            使われる。device が TPU/XLA でない場合は常に無視される
            (=CPU/GPU では以前と全く同じ挙動になる)。
        xla_output_buckets (Optional[Sequence[int]]): 生成する音声のフレーム数
            (Duration Predictor が予測する長さ) を丸め込むバケツ。使い方・省略時の
            挙動は xla_input_buckets と同様
            (style_bert_vits2.xla.default_output_length_buckets(hps) が使われる)。
    """

    is_jp_extra = hps.version.endswith("JP-Extra")
    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
        text,
        language,
        hps,
        device,
        assist_text=assist_text,
        assist_text_weight=assist_text_weight,
        given_phone=given_phone,
        given_tone=given_tone,
    )
    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        bert = bert[:, 3:]
        ja_bert = ja_bert[:, 3:]
        en_bert = en_bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        bert = bert[:, :-2]
        ja_bert = ja_bert[:, :-2]
        en_bert = en_bert[:, :-2]

    # "tpu"/"xla" が指定された場合、ここで実際の torch_xla デバイスオブジェクトに変換する。
    # それ以外のデバイスでは resolved_device は device (元の文字列) と同じ値になるため、
    # 以降の .to() 呼び出しの挙動は一切変わらない。
    resolved_device = resolve_device(device)
    use_xla = is_xla_device(device)

    # TPU/XLA 使用時のみ、明示的な指定がなければデフォルトの長さバケツを適用する。
    # CPU/GPU では xla_input_buckets/xla_output_buckets は常に None のままなので、
    # 以降のパディング/バケツ化処理は丸ごとスキップされ、元の実装と完全に同じ
    # 形状・同じ結果になる (=このパッチによる CPU/GPU 推論への副作用はない)。
    if use_xla and xla_input_buckets is None:
        xla_input_buckets = default_input_length_buckets()
    if use_xla and xla_output_buckets is None:
        xla_output_buckets = default_output_length_buckets(hps)

    with torch.no_grad():
        true_x_len = phones.size(0)  # ゼロ埋めする前の「真の」音素列長を控えておく

        input_bucket_len: Optional[int] = None
        if xla_input_buckets:
            input_bucket_len = pick_bucket(true_x_len, xla_input_buckets)
            if input_bucket_len is None:
                warn_bucket_overflow("入力音素長", true_x_len, xla_input_buckets)
                # バケツに収まらない場合は諦めて実測の長さのまま進む (切り詰めない)

        def _to_device_and_pad(t: torch.Tensor) -> torch.Tensor:
            t = t.to(resolved_device)
            if input_bucket_len is not None:
                t = pad_last_dim(t, input_bucket_len)
            return t.unsqueeze(0)

        x_tst = _to_device_and_pad(phones)
        tones = _to_device_and_pad(tones)
        lang_ids = _to_device_and_pad(lang_ids)
        bert = _to_device_and_pad(bert)
        ja_bert = _to_device_and_pad(ja_bert)
        en_bert = _to_device_and_pad(en_bert)
        # モデル側の x_mask 計算に使われるのは常にパディング前の「真の」音素長。
        # (テンソル自体の見た目の長さがバケツ長になっていても、これにより
        #  パディング部分は attention 含めて正しくマスクされ、結果は元と一致する)
        x_tst_lengths = torch.LongTensor([true_x_len]).to(resolved_device)
        style_vec_tensor = torch.from_numpy(style_vec).to(resolved_device).unsqueeze(0)
        del phones
        sid_tensor = torch.LongTensor([sid]).to(resolved_device)

        if is_jp_extra:
            output = cast(SynthesizerTrnJPExtra, net_g).infer(
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones,
                lang_ids,
                ja_bert,
                style_vec=style_vec_tensor,
                length_scale=length_scale,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                fixed_len_buckets=xla_output_buckets,
            )
        else:
            output = cast(SynthesizerTrn, net_g).infer(
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones,
                lang_ids,
                bert,
                ja_bert,
                en_bert,
                style_vec=style_vec_tensor,
                length_scale=length_scale,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                fixed_len_buckets=xla_output_buckets,
            )

        audio = output[0][0, 0].data.cpu().float().numpy()

        # 出力側もバケツ化している場合、生成された波形はバケツ長ぶん余分に
        # 長い (末尾は y_mask が 0 になっている無音相当の区間) ので、
        # 実際に有効なフレーム数を y_mask から求めて波形を切り詰める。
        if xla_output_buckets:
            true_frames = int(output[2][0, 0].sum().item())
            audio = audio[: true_frames * hps.data.hop_length]

        del (
            x_tst,
            tones,
            lang_ids,
            bert,
            x_tst_lengths,
            sid_tensor,
            ja_bert,
            en_bert,
            style_vec,
        )  # , emo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # torch_xla 使用時、ここまでで .cpu()/.item() により計算グラフは実行済みだが、
        # 念のため明示的に区切っておく (TPU/XLA 以外では no-op)。
        mark_step(resolved_device)

        return audio
