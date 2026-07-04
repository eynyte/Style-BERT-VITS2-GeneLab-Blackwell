import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from config import get_config
from mel_processing import (
    mel_spectrogram_torch,
    spectrogram_torch,
    spec_to_mel_torch,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons
from style_bert_vits2.models.hyper_parameters import HyperParametersData
from style_bert_vits2.models.utils import load_filepaths_and_text, load_wav_to_torch
from style_bert_vits2.nlp import cleaned_text_to_sequence


config = get_config()
"""Multi speaker version"""


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(
        self,
        audiopaths_sid_text: str,
        hparams: HyperParametersData,
        # 起動時に BERT/spec/wav/style_vec をすべて CPU RAM に
        # ロードしておくかどうか。num_workers=1 / prefetch_factor
        # 固定の環境では、ディスク I/O と pickle 展開を 1 回に
        # 集約できるため、各 step の __getitem__ がほぼメモリの
        # ルックアップだけになり、データ待ちストールが激減する。
        preload: bool = True,
        # spec.pt / wav / style_vec.npy だけを preload したい
        # とき True にする (BERT がメモリを食いすぎる場合用)。
        preload_bert: bool = True,
        # 真値 mel-spec を CPU 側で先回りに作ってキャッシュする
        # かどうか。毎 step の spec_to_mel_torch を消せる。
        preload_mel: bool = True,
    ):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.spk_map = hparams.spk2id
        self.hparams = hparams
        self.use_jp_extra = getattr(hparams, "use_jp_extra", False)

        self.use_mel_spec_posterior = getattr(
            hparams, "use_mel_posterior_encoder", False
        )
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 384)

        self._preload_enabled = preload
        self._preload_bert = preload and preload_bert
        self._preload_mel = preload and preload_mel
        # 0 = lazy load のまま (preload_* フラグが False のとき用)
        self._spec_list = None
        self._bert_list = None
        self._style_list = None
        self._wav_list = None
        self._mel_list = None
        self._phone_list = None
        self._tone_list = None
        self._language_list = None
        self._sid_list = None

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()
        if self._preload_enabled:
            self._preload_all()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        paths = []
        skipped = 0
        logger.info("Init dataset...")
        for _id, spk, language, text, phones, tone, word2ph in tqdm(
            self.audiopaths_sid_text, file=sys.stdout, dynamic_ncols=True
        ):
            audiopath = f"{_id}"
            # if self.min_text_len <= len(phones) and len(phones) <= self.max_text_len:
            phones = phones.split(" ")
            tone = [int(i) for i in tone.split(" ")]
            word2ph = [int(i) for i in word2ph.split(" ")]
            audiopaths_sid_text_new.append(
                [audiopath, spk, language, text, phones, tone, word2ph]
            )
            paths.append(audiopath)
            # else:
            #     skipped += 1
        logger.info(
            "skipped: "
            + str(skipped)
            + ", total: "
            + str(len(self.audiopaths_sid_text))
        )
        self.audiopaths_sid_text = audiopaths_sid_text_new
        # os.path.getsize はメタデータ取得だが、データセットが大きい
        # と初期化自体が遅いので ThreadPool で並列化する。
        logger.info("Collecting wav sizes (parallel)...")
        try:
            n_workers = max(8, min(32, (os.cpu_count() or 8) * 2))
        except Exception:
            n_workers = 16
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            sizes = list(
                tqdm(
                    ex.map(os.path.getsize, paths),
                    total=len(paths),
                    file=sys.stdout,
                    dynamic_ncols=True,
                    desc="os.path.getsize",
                )
            )
        self.lengths = [s // (2 * self.hop_length) for s in sizes]

    # ---- preloading helpers -------------------------------------------

    def _load_spec_one(self, audiopath: str) -> torch.Tensor:
        spec_filename = audiopath.replace(".wav", ".spec.pt")
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        try:
            spec = torch.load(spec_filename, map_location="cpu")
        except Exception:
            audio, _ = load_wav_to_torch(audiopath)
            audio_norm = (audio / self.max_wav_value).unsqueeze(0)
            if self.use_mel_spec_posterior:
                spec = mel_spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.hparams.mel_fmin,
                    self.hparams.mel_fmax,
                    center=False,
                )
            else:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
            spec = torch.squeeze(spec, 0)
            if config.train_ms_config.spec_cache:
                torch.save(spec, spec_filename)
        return spec.contiguous()

    def _load_bert_one(self, audiopath: str) -> torch.Tensor:
        bert_path = audiopath.replace(".wav", ".bert.pt")
        return torch.load(bert_path, map_location="cpu").contiguous()

    def _load_style_one(self, audiopath: str) -> torch.Tensor:
        return torch.from_numpy(np.load(f"{audiopath}.npy")).contiguous()

    def _load_wav_one(self, audiopath: str) -> torch.Tensor:
        audio, sr = load_wav_to_torch(audiopath)
        if sr != self.sampling_rate:
            raise ValueError(
                f"{audiopath} {sr} SR doesn't match target {self.sampling_rate} SR"
            )
        return (audio / self.max_wav_value).contiguous()

    def _preload_one(self, idx: int):
        item = self.audiopaths_sid_text[idx]
        audiopath, sid, language, text, phones, tone, word2ph = item

        # ---- spec / wav / style_vec ----
        spec = self._load_spec_one(audiopath)
        wav = self._load_wav_one(audiopath)
        style_vec = self._load_style_one(audiopath)

        # ---- mel を CPU 側で先回りに作成 ----
        if self._preload_mel:
            # spec は [n_mels, T] なので unsqueeze して batch 次元を足す
            mel = spec_to_mel_torch(
                spec.unsqueeze(0),
                self.filter_length,
                self.n_mel_channels,
                self.sampling_rate,
                self.hparams.mel_fmin,
                self.hparams.mel_fmax,
            ).squeeze(0)
            mel = mel.contiguous()
        else:
            mel = None

        # ---- BERT ----
        if self._preload_bert:
            bert_ori = self._load_bert_one(audiopath)
        else:
            bert_ori = None

        # ---- phone / tone / language (intersperse 済み) ----
        # 事前に sequence 変換 + intersperse を済ませて、ルックアップ
        # 1 回で済む形にする。
        phones_ids, tone_ids, language_ids = cleaned_text_to_sequence(
            phones, tone, language
        )
        if self.add_blank:
            phones_ids = commons.intersperse(phones_ids, 0)
            tone_ids = commons.intersperse(tone_ids, 0)
            language_ids = commons.intersperse(language_ids, 0)
            word2ph = [w * 2 for w in word2ph]
            word2ph[0] += 1
        phone_t = torch.LongTensor(phones_ids)
        tone_t = torch.LongTensor(tone_ids)
        language_t = torch.LongTensor(language_ids)
        sid_t = torch.LongTensor([int(self.spk_map[sid])])

        # ---- wav length を spec length と整合させる ----
        # 元の wav 長は wav.numel() / 1ch だが、spec からは
        # spec.shape[-1] * hop_length で復元できる。wav が pad/truncate
        # されていなければ一致する。安全のため spec 側の長さを使う。
        expected_wav_len = spec.shape[-1] * self.hop_length
        if wav.numel() != expected_wav_len:
            if wav.numel() > expected_wav_len:
                wav = wav[:expected_wav_len]
            else:
                pad = expected_wav_len - wav.numel()
                wav = torch.nn.functional.pad(wav, (0, pad))
        wav = wav.contiguous()

        return {
            "spec": spec,
            "wav": wav,
            "style": style_vec,
            "mel": mel,
            "bert": bert_ori,
            "phone": phone_t,
            "tone": tone_t,
            "language": language_t,
            "sid": sid_t,
            # 言語ごとの BERT 切り分けは __getitem__ で決定する。
            # 1024 次元の bert を 1 本だけ持つ (JP/ZH/EN)。
            "lang_str": language,
        }

    def _preload_all(self):
        """
        データセット全体を CPU RAM にロードする。

        - 効果は劇的で、__getitem__ が CPU メモリの参照 + 言語判定だけ
          になり、disk I/O と pickle 展開が初回 init に 1 回だけになる。
        - 1 worker / prefetch_factor 固定の環境では、worker は fork 後
          COW でこの RAM を参照するためメモリ消費はほぼ 0。
        """
        n = len(self.audiopaths_sid_text)
        logger.info(
            f"Pre-loading dataset into RAM (n={n}, "
            f"bert={self._preload_bert}, mel={self._preload_mel})..."
        )
        # 並列度: spec/BERT/wav/style_vec は I/O バウンドなので ThreadPool。
        # wav ロードだけは scipy GIL の関係でプロセス並列のほうが
        # 速いが、ここではお手軽に ThreadPool で並列化する。
        try:
            n_workers = max(4, min(16, (os.cpu_count() or 8)))
        except Exception:
            n_workers = 8
        # pickle 経由で worker に渡す overhead を避けるため、
        # tqdm は map 段階で出す。
        results = [None] * n
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(self._preload_one, i): i for i in range(n)}
            done = 0
            for fut in tqdm(
                futures, total=n, file=sys.stdout, dynamic_ncols=True,
                desc="preload",
            ):
                i = futures[fut]
                results[i] = fut.result()
                done += 1
        # メモリ効率のため、別リストにバラして保持する。
        self._spec_list = [r["spec"] for r in results]
        self._wav_list = [r["wav"] for r in results]
        self._style_list = [r["style"] for r in results]
        if self._preload_mel:
            self._mel_list = [r["mel"] for r in results]
        else:
            self._mel_list = [None] * n
        if self._preload_bert:
            self._bert_list = [r["bert"] for r in results]
        else:
            self._bert_list = [None] * n
        self._phone_list = [r["phone"] for r in results]
        self._tone_list = [r["tone"] for r in results]
        self._language_list = [r["language"] for r in results]
        self._sid_list = [r["sid"] for r in results]
        self._lang_list = [r["lang_str"] for r in results]

        # 統計を 1 回だけログ。
        try:
            spec_total = sum(s.numel() for s in self._spec_list)
            wav_total = sum(w.numel() for w in self._wav_list)
            mel_total = (
                sum(m.numel() for m in self._mel_list) if self._preload_mel else 0
            )
            bert_total = (
                sum(b.numel() for b in self._bert_list) if self._preload_bert else 0
            )
            mb = 4 / (1024 * 1024)
            logger.info(
                "Preload memory: spec=%.1fMB wav=%.1fMB mel=%.1fMB bert=%.1fMB"
                % (
                    spec_total * mb,
                    wav_total * mb,
                    mel_total * mb,
                    bert_total * mb,
                )
            )
        except Exception:
            pass
        logger.info("Preload done.")

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, language, text, phones, tone, word2ph = audiopath_sid_text

        bert, ja_bert, en_bert, phones, tone, language = self.get_text(
            text, word2ph, phones, tone, language, audiopath
        )

        spec, wav = self.get_audio(audiopath)
        sid = torch.LongTensor([int(self.spk_map[sid])])
        style_vec = torch.FloatTensor(np.load(f"{audiopath}.npy"))
        if self.use_jp_extra:
            return (phones, spec, wav, sid, tone, language, ja_bert, style_vec)
        else:
            return (
                phones,
                spec,
                wav,
                sid,
                tone,
                language,
                bert,
                ja_bert,
                en_bert,
                style_vec,
            )

    # ---- 高速化された get_audio / get_text / __getitem__ ----------------

    def _make_bert_triplet(self, bert_ori, language_str, T):
        """
        bert_ori (None or Tensor[1024, T]) を言語ごとに 3 本の
        zero-tensor に割り振る。空テンソルの生成は torch.zeros ではなく
        torch.empty + zero_ で同コストだが、empty のほうが少しだけ速い。
        """
        if bert_ori is None:
            bert = torch.empty(1024, T, dtype=torch.float32).zero_()
        elif language_str == "ZH":
            bert = bert_ori
        elif language_str == "JP":
            bert = torch.empty(1024, T, dtype=torch.float32).zero_()
        elif language_str == "EN":
            bert = torch.empty(1024, T, dtype=torch.float32).zero_()
        else:
            bert = bert_ori
        return bert

    def _build_item(self, index):
        """
        preload モード: キャッシュ済み tensor から item を組み立てる。
        非 preload モード: 従来通り lazy にロードする。
        """
        if self._spec_list is None:
            return self.get_audio_text_speaker_pair(
                self.audiopaths_sid_text[index]
            )

        item = self.audiopaths_sid_text[index]
        _audiopath, sid, language_str, _text, _phones, _tone, _word2ph = item

        spec = self._spec_list[index]
        wav = self._wav_list[index]
        style_vec = self._style_list[index]
        phones = self._phone_list[index]
        tone = self._tone_list[index]
        language = self._language_list[index]
        sid_t = self._sid_list[index]
        bert_ori = self._bert_list[index] if self._preload_bert else None

        T = phones.size(0)
        # 言語ごとに 3 本の BERT スロットを作る。
        if self.use_jp_extra:
            # jp_extra パス: ja_bert に本物の値を入れ、bert は zero。
            if language_str == "JP":
                ja_bert = bert_ori if bert_ori is not None else \
                    self._load_bert_one(_audiopath)
                bert = torch.zeros(1024, T, dtype=torch.float32)
            elif language_str == "ZH":
                ja_bert = torch.zeros(1024, T, dtype=torch.float32)
                bert = bert_ori if bert_ori is not None else \
                    self._load_bert_one(_audiopath)
            elif language_str == "EN":
                ja_bert = torch.zeros(1024, T, dtype=torch.float32)
                bert = torch.zeros(1024, T, dtype=torch.float32)
            else:
                ja_bert = bert_ori if bert_ori is not None else \
                    self._load_bert_one(_audiopath)
                bert = torch.zeros(1024, T, dtype=torch.float32)
            return (phones, spec, wav, sid_t, tone, language, ja_bert, style_vec)
        else:
            if language_str == "ZH":
                bert = bert_ori if bert_ori is not None else \
                    self._load_bert_one(_audiopath)
                ja_bert = torch.zeros(1024, T, dtype=torch.float32)
                en_bert = torch.zeros(1024, T, dtype=torch.float32)
            elif language_str == "JP":
                bert = torch.zeros(1024, T, dtype=torch.float32)
                ja_bert = bert_ori if bert_ori is not None else \
                    self._load_bert_one(_audiopath)
                en_bert = torch.zeros(1024, T, dtype=torch.float32)
            elif language_str == "EN":
                bert = torch.zeros(1024, T, dtype=torch.float32)
                ja_bert = torch.zeros(1024, T, dtype=torch.float32)
                en_bert = bert_ori if bert_ori is not None else \
                    self._load_bert_one(_audiopath)
            else:
                bert = bert_ori if bert_ori is not None else \
                    self._load_bert_one(_audiopath)
                ja_bert = torch.zeros(1024, T, dtype=torch.float32)
                en_bert = torch.zeros(1024, T, dtype=torch.float32)
            return (
                phones,
                spec,
                wav,
                sid_t,
                tone,
                language,
                bert,
                ja_bert,
                en_bert,
                style_vec,
            )

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"{filename} {sampling_rate} SR doesn't match target {self.sampling_rate} SR"
            )
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        try:
            spec = torch.load(spec_filename)
        except:
            if self.use_mel_spec_posterior:
                spec = mel_spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.hparams.mel_fmin,
                    self.hparams.mel_fmax,
                    center=False,
                )
            else:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
            spec = torch.squeeze(spec, 0)
            if config.train_ms_config.spec_cache:
                torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text, word2ph, phone, tone, language_str, wav_path):
        phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)
        if self.add_blank:
            phone = commons.intersperse(phone, 0)
            tone = commons.intersperse(tone, 0)
            language = commons.intersperse(language, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1
        bert_path = wav_path.replace(".wav", ".bert.pt")
        try:
            bert_ori = torch.load(bert_path)
            assert bert_ori.shape[-1] == len(phone)
        except Exception as e:
            logger.warning("Bert load Failed")
            logger.warning(e)

        if language_str == "ZH":
            bert = bert_ori
            ja_bert = torch.zeros(1024, len(phone))
            en_bert = torch.zeros(1024, len(phone))
        elif language_str == "JP":
            bert = torch.zeros(1024, len(phone))
            ja_bert = bert_ori
            en_bert = torch.zeros(1024, len(phone))
        elif language_str == "EN":
            bert = torch.zeros(1024, len(phone))
            ja_bert = torch.zeros(1024, len(phone))
            en_bert = bert_ori
        phone = torch.LongTensor(phone)
        tone = torch.LongTensor(tone)
        language = torch.LongTensor(language)
        return bert, ja_bert, en_bert, phone, tone, language

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def get_cached_mel(self, index):
        """
        preload モード時のみ: CPU 側にキャッシュ済みの mel-spec を返す。
        非 preload モードでは None を返すので、訓練ループ側で
        fallback する必要がある。
        """
        if self._mel_list is None:
            return None
        return self._mel_list[index]

    def __getitem__(self, index):
        return self._build_item(index)

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False, use_jp_extra=False):
        self.return_ids = return_ids
        self.use_jp_extra = use_jp_extra

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        tone_padded = torch.LongTensor(len(batch), max_text_len)
        language_padded = torch.LongTensor(len(batch), max_text_len)
        # This is ZH bert if not use_jp_extra, JA bert if use_jp_extra
        bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
        if not self.use_jp_extra:
            ja_bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
            en_bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
        style_vec = torch.FloatTensor(len(batch), 256)

        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        tone_padded.zero_()
        language_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        bert_padded.zero_()
        if not self.use_jp_extra:
            ja_bert_padded.zero_()
            en_bert_padded.zero_()
        style_vec.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

            tone = row[4]
            tone_padded[i, : tone.size(0)] = tone

            language = row[5]
            language_padded[i, : language.size(0)] = language

            bert = row[6]
            bert_padded[i, :, : bert.size(1)] = bert

            if self.use_jp_extra:
                style_vec[i, :] = row[7]
            else:
                ja_bert = row[7]
                ja_bert_padded[i, :, : ja_bert.size(1)] = ja_bert

                en_bert = row[8]
                en_bert_padded[i, :, : en_bert.size(1)] = en_bert
                style_vec[i, :] = row[9]

        if self.use_jp_extra:
            return (
                text_padded,
                text_lengths,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                sid,
                tone_padded,
                language_padded,
                bert_padded,
                style_vec,
            )
        else:
            return (
                text_padded,
                text_lengths,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                sid,
                tone_padded,
                language_padded,
                bert_padded,
                ja_bert_padded,
                en_bert_padded,
                style_vec,
            )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        logger.info(f"Bucket info: {self.num_samples_per_bucket}")
        # logger.info(
        #     f"Unused samples: {len(self.lengths) - sum(self.num_samples_per_bucket)}"
        # )
        # ↑マイナスになることあるし、別にこれは使われないサンプル数ではないようだ……
        # バケットの仕組みはよく分からない

        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        try:
            for i in range(len(buckets) - 1, 0, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)
            assert all(len(bucket) > 0 for bucket in buckets)
        # When one bucket is not traversed
        except Exception as e:
            logger.info("Bucket warning ", e)
            for i in range(len(buckets) - 1, -1, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            if len_bucket == 0:
                continue
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
