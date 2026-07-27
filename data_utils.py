import os
import random
import sys

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from config import get_config
from mel_processing import mel_spectrogram_torch, spectrogram_torch
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons
from style_bert_vits2.models.hyper_parameters import HyperParametersData
from style_bert_vits2.models.utils import load_filepaths_and_text, load_wav_to_torch
from style_bert_vits2.nlp import cleaned_text_to_sequence


config = get_config()
"""Multi speaker version"""

# ------------------------------------------------------------------------
# [カスタム改変] 記号(？/！/「 等)による疑似話者切り替えのための対応
# ------------------------------------------------------------------------
# True にすると、話者ごとの本物の (発話固有の) style vector を使う代わりに、
# 「推論時に Neutral として使われるのと同じ、話者全体の平均 style vector」を
# 全サンプルに定数として与える。
#
# 通常このスクリプトは f"{audiopath}.npy" (pyannote 由来の話者性を色濃く含む
# xvector) を発話ごとにロードして TextEncoder に渡している。sid を全発話で
# 同一 (例: 全員 "mix") にしても、この per-utterance の style vector に
# 話者を区別する情報がそのまま残ってしまうと、モデルは (学習時にしか
# 存在しない) この抜け道に頼って損失を下げてしまい、テキスト内容
# (音素列・トーン・BERT特徴、たとえば文頭の ？/！/「 のような記号) を
# 手がかりにする経路を学習しない可能性が高い。
#
# 推論時は style_vectors.npy の Neutral (=話者全体の平均) 1本しか
# 存在しないため、学習時もそれと同じ「全発話平均」の定数ベクトルに
# 差し替えることで、この抜け道を塞ぎ、テキスト内容だけが話者を
# 区別できる唯一の手がかりになるようにする。
#
# 通常通り「話者ごとに意味のある個別 style vector を使いたい」学習に
# 戻す場合は False にすること。
USE_CONSTANT_STYLE_VEC = True


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_text: str, hparams: HyperParametersData):
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

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

        self._const_style_vec = None
        if USE_CONSTANT_STYLE_VEC:
            self._const_style_vec = self._load_or_compute_const_style_vec()

    def _load_or_compute_const_style_vec(self) -> "torch.Tensor":
        """
        [カスタム改変] 学習時に全サンプルへ与える、定数の style vector を用意する。

        train_ms_jp_extra.py は学習ループに入る前に default_style.save_styles_by_dirs()
        を呼び、config.out_dir/style_vectors.npy の 0 行目に「全発話の平均 (Neutral)」を
        保存している (これは推論時に実際に使われるものと同一)。これが存在すればそれを
        最優先で使い、存在しない場合 (--skip_default_style 指定時や、このローダーを
        単体で使う場合など) は、手元のデータセットから同じ計算をフォールバックとして
        行う。
        """
        style_vectors_path = os.path.join(str(config.out_dir), "style_vectors.npy")
        if os.path.exists(style_vectors_path):
            mean_vec = np.load(style_vectors_path)[0]
            logger.info(
                f"[USE_CONSTANT_STYLE_VEC] {style_vectors_path} の Neutral (0行目) を"
                "学習用の固定 style vector として使用します。"
            )
        else:
            logger.warning(
                f"[USE_CONSTANT_STYLE_VEC] {style_vectors_path} が見つからないため、"
                "データセット内の *.npy から平均 style vector をその場で計算します。"
            )
            all_vecs = [
                np.load(f"{item[0]}.npy") for item in self.audiopaths_sid_text
            ]
            mean_vec = np.mean(np.stack(all_vecs, axis=0), axis=0)
        return torch.FloatTensor(mean_vec)

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
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
            lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
            # else:
            #     skipped += 1
        logger.info(
            "skipped: "
            + str(skipped)
            + ", total: "
            + str(len(self.audiopaths_sid_text))
        )
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, language, text, phones, tone, word2ph = audiopath_sid_text

        bert, ja_bert, en_bert, phones, tone, language = self.get_text(
            text, word2ph, phones, tone, language, audiopath
        )

        spec, wav = self.get_audio(audiopath)
        sid = torch.LongTensor([int(self.spk_map[sid])])
        if USE_CONSTANT_STYLE_VEC:
            # [カスタム改変] 発話固有の (話者性が漏れる) xvector ではなく、
            # __init__ で用意した固定の平均 style vector を使う。
            style_vec = self._const_style_vec
        else:
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

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

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
