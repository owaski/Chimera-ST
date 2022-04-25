# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import logging
import os.path as op
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)
from fairseq.data.audio.feature_transforms import \
    CompositeAudioFeatureTransform
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    get_features_or_waveform,
    _collate_frames,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
)
import numpy as np

import torch.nn.functional as F
from fairseq.data.joint_dataset import JointDataset

from fairseq.data.multi_corpus_sampled_dataset import MultiCorpusSampledDataset


logger = logging.getLogger(__name__)

MAX_FRAME = 320000

class MultilingualTripletDataConfig(S2TDataConfig):
    """Wrapper class for data config YAML"""

    @property
    def src_bpe_tokenizer(self) -> Dict:
        """Subword tokenizer to apply after pre-tokenization. Returning
        a dictionary with `bpe` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("src_bpe_tokenizer", {"bpe": None})

    @property
    def src_vocab_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("src_vocab_filename", "dict.txt")

    @property
    def phone_dict(self):
        return self.config.get("phone_dict", None)

    @property
    def lang_list_filename(self):
        return self.config.get("lang_list_filename", None)

    @property
    def prepend_src_lang_tag(self):
        return self.config.get("prepend_src_lang_tag", False)

    @property
    def voxpopuli_root(self):
        return self.config.get("voxpopuli_root", None)

    @property
    def unlabeled_sampling_ratio(self):
        return self.config.get("unlabeled_sampling_ratio", 1.0)


class MultilingualTripletDataset(SpeechToTextDataset):
    LANG_TAG_TEMPLATE = "<lang:{}>"

    def __init__(
        self,
        split: str,
        is_train_split: bool,
        data_cfg: MultilingualTripletDataConfig,
        audio_paths: List[str],
        n_frames: List[int],
        src_texts: Optional[List[str]] = None,
        tgt_texts: Optional[List[str]] = None,
        phones: Optional[List[str]] = None,
        speakers: Optional[List[str]] = None,
        src_langs: Optional[List[str]] = None,
        tgt_langs: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        tgt_dict: Optional[Dictionary] = None,
        src_dict: Optional[Dictionary] = None,
        phone_dict: Optional[Dict] = None,
        pre_tokenizer=None,
        bpe_tokenizer=None,
        src_bpe_tokenizer=None,
        normalize=False,
        mask=True,
        sample_rate=16000,
        src_lang2idx=None,
        labeled=True,
    ):
        self.split, self.is_train_split = split, is_train_split
        self.data_cfg = data_cfg
        self.audio_paths = audio_paths
        self.n_frames = [min(n_f, MAX_FRAME) for n_f in n_frames]
        self.n_samples = len(audio_paths)
        assert len(n_frames) == self.n_samples > 0
        assert src_texts is None or len(src_texts) == self.n_samples
        assert tgt_texts is None or len(tgt_texts) == self.n_samples
        assert speakers is None or len(speakers) == self.n_samples
        assert src_langs is None or len(src_langs) == self.n_samples
        assert tgt_langs is None or len(tgt_langs) == self.n_samples
        assert ids is None or len(ids) == self.n_samples
        assert (tgt_texts is None) or (
            tgt_dict is not None and tgt_texts is not None
        )
        assert (src_texts is None) or (
            src_dict is not None and src_texts is not None
        )
        self.src_texts, self.tgt_texts = src_texts, tgt_texts
        self.phones = phones
        self.src_langs, self.tgt_langs = src_langs, tgt_langs
        self.src_lang2idx = src_lang2idx
        self.lang_codes = self.get_lang_codes(data_cfg.lang_list_filename)
        self.lang2langcode = {code[:2] : code for code in self.lang_codes}
        self.tgt_dict = tgt_dict
        self.src_dict = src_dict
        self.phone_dict = phone_dict
        self.labeled = labeled
        if labeled:
            self.check_lang_tag()
        self.ids = ids
        self.shuffle = data_cfg.shuffle if is_train_split else False

        self.feature_transforms = CompositeAudioFeatureTransform.\
            from_config_dict(
                self.data_cfg.get_feature_transforms(split, is_train_split)
            )

        self.pre_tokenizer = pre_tokenizer
        self.bpe_tokenizer = bpe_tokenizer
        self.src_bpe_tokenizer = src_bpe_tokenizer
        self.normalize = normalize
        self.mask = mask
        self.sample_rate = sample_rate

        logger.info(self.__repr__())

    @classmethod
    def get_lang_codes(cls, language_list_filename):
        assert language_list_filename is not None
        with open(language_list_filename, 'r') as r:
            codes = [line.strip() for line in r.readlines() if line.strip() != ""]
        return codes

    def check_lang_tag(self):
        if self.data_cfg.prepend_tgt_lang_tag:
            assert self.tgt_langs is not None and self.tgt_dict is not None
            tgt_lang_tags = [
                self.LANG_TAG_TEMPLATE.format(self.lang2langcode[t[:2]]) for t in set(self.tgt_langs)
            ]
            assert all(t in self.tgt_dict for t in tgt_lang_tags)
        if self.data_cfg.prepend_src_lang_tag:
            assert self.src_langs is not None and self.src_dict is not None
            src_lang_tags = [
                self.LANG_TAG_TEMPLATE.format(self.lang2langcode[s[:2]]) for s in set(self.src_langs)
            ]
            assert all(s in self.src_dict for s in src_lang_tags)

    def tokenize_text(self, text: str, side='target'):
        if self.pre_tokenizer is not None:
            text = self.pre_tokenizer.encode(text)
        if side == 'target':
            if self.bpe_tokenizer is not None:
                text = self.bpe_tokenizer.encode(text)
        elif side == 'source':
            if self.src_bpe_tokenizer is not None:
                text = self.src_bpe_tokenizer.encode(text)
        return text

    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
        source = get_features_or_waveform(
            self.audio_paths[index],
            need_waveform=self.data_cfg.use_audio_input,
            sample_rate=self.sample_rate,
        )
        if self.feature_transforms is not None:
            assert not self.data_cfg.use_audio_input
            source = self.feature_transforms(source)
        source = torch.from_numpy(source).float()
        if self.normalize:
            with torch.no_grad():
                source = F.layer_norm(source, source.shape)

        target = None
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index], 'target')
            target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.data_cfg.prepend_tgt_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format( \
                    self.lang2langcode[self.tgt_langs[index][:2]])
                lang_tag_idx = self.tgt_dict.index(lang_tag)
                target = torch.cat(
                    (torch.LongTensor([lang_tag_idx]), target),
                    0
                )

        src_text = None
        src_lang_tag_idx = None
        if self.src_texts is not None:
            src_tokenized = self.tokenize_text(self.src_texts[index], 'source')
            src_text = self.src_dict.encode_line(
                src_tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.data_cfg.prepend_src_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format( \
                    self.lang2langcode[self.src_langs[index][:2]])
                src_lang_tag_idx = self.src_dict.index(lang_tag)
                src_text = torch.cat(
                    (torch.LongTensor([src_lang_tag_idx]), src_text),
                    0
                )

        phones = None
        if self.phones is not None:
            phones = [
                self.phone_dict[phone] for phone in self.phones[index].split(' ')
            ]

        # src_lang_idx = None
        # if self.src_langs[index] in self.src_lang2idx:
        #     src_lang_idx = self.src_lang2idx[self.src_langs[index]]

        return index, source, target, src_text, src_lang_tag_idx, phones

    def collater(
        self, samples: List[Tuple[int, torch.Tensor, torch.Tensor]]
    ) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([i for i, _, _, _, _, _ in samples], dtype=torch.long)
        frames = _collate_frames(
            [s for _, s, _, _, _, _ in samples], self.data_cfg.use_audio_input
        )
        # sort samples by descending number of frames
        n_frames = torch.tensor([s.size(0) for _, s, _, _, _, _ in samples],
                                dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        if self.labeled:
            target = fairseq_data_utils.collate_tokens(
                [t for _, _, t, _, _, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [t.size(0) - 1 for _, _, t, _, _, _ in samples], dtype=torch.long
            ).index_select(0, order)
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [t for _, _, t, _, _, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(t.size(0) for _, _, t, _, _, _ in samples)

        src_text, src_text_lengths = None, None
        asr_target, asr_target_lengths, asr_prev_output_tokens = None, None, None
        if self.labeled:
            src_text = fairseq_data_utils.collate_tokens(
                [s for _, _, _, s, _, _ in samples],
                self.src_dict.pad(),
                self.src_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            src_text = src_text.index_select(0, order)
            src_text_lengths = torch.tensor(
                [s.size(0) for _, _, _, s, _, _ in samples], dtype=torch.long
            ).index_select(0, order)

            asr_target = fairseq_data_utils.collate_tokens(
                [s for _, _, _, s, _, _ in samples],
                self.src_dict.pad(),
                self.src_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            ).index_select(0, order)
            asr_target_lengths = torch.tensor(
                [s.size(0) - 1 for _, _, _, s, _, _ in samples], dtype=torch.long
            ).index_select(0, order)
            asr_prev_output_tokens = fairseq_data_utils.collate_tokens(
                [s for _, _, _, s, _, _ in samples],
                self.src_dict.pad(),
                self.src_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            ).index_select(0, order)

        src_lang_tag_indices = torch.tensor(
            [src_lang_tag_idx for _, _, _, _, src_lang_tag_idx, _ in samples], dtype=torch.long
        ).index_select(0, order).unsqueeze(-1)

        phones = []
        phone_lengths = []
        for o in order:
            phones.extend(samples[o][5])
            phone_lengths.append(len(samples[o][5]))
        phones = torch.LongTensor(phones)
        phone_lengths = torch.LongTensor(phone_lengths)

        out = {
            "id": indices,
            "net_input": {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": prev_output_tokens,
                "mask": self.mask,
                "src_lang_tag_indices": src_lang_tag_indices
            },
            "target": target,
            "target_lengths": target_lengths,
            "src_text": src_text,
            "src_text_lengths": src_text_lengths,
            "asr_target": asr_target,
            "asr_target_lengths": asr_target_lengths,
            "asr_prev_output_tokens": asr_prev_output_tokens,
            "phones": phones,
            "phone_lengths": phone_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        return out

    def size(self, index):
        t_len = 0
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            t_len = len(tokenized.split(" "))
        return self.n_frames[index], t_len


class MultilingualTripletDatasetCreator(SpeechToTextDatasetCreator):

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[List[Dict]],
        data_cfg: MultilingualTripletDataConfig,
        tgt_dict,
        src_dict,
        phone_dict,
        pre_tokenizer,
        bpe_tokenizer,
        src_bpe_tokenizer,
        normalize,
        mask,
        sample_rate,
        src_lang2idx
    ) -> MultilingualTripletDataset:
        parts = split_name.split('_')
        if len(parts) == 1: # unlabeled data
            labeled = False
            audio_paths, n_frames, src_texts, tgt_texts, ids = [], [], None, None, []
            speakers, src_langs, tgt_langs = None, [], None
            for s in samples:
                ids.extend([ss[cls.KEY_ID] for ss in s])
                audio_paths.extend(
                    [op.join(data_cfg.voxpopuli_root, 'unlabelled_data', ss[cls.KEY_SRC_LANG], ss[cls.KEY_AUDIO]) for ss in s]
                )
                n_frames.extend([int(ss[cls.KEY_N_FRAMES]) for ss in s])
                src_langs.extend([ss.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG)
                                for ss in s])            
        else:
            labeled = True
            audio_paths, n_frames, src_texts, tgt_texts, ids = [], [], [], [], []
            speakers, src_langs, tgt_langs = [], [], []
            phones = []
            for s in samples:
                ids.extend([ss[cls.KEY_ID] for ss in s])
                audio_paths.extend(
                    [op.join(data_cfg.audio_root, ss[cls.KEY_SRC_LANG], '16kHz', ss[cls.KEY_AUDIO]) for ss in s]
                )
                n_frames.extend([int(ss[cls.KEY_N_FRAMES]) for ss in s])
                tgt_texts.extend([ss[cls.KEY_TGT_TEXT] for ss in s])
                src_texts.extend(
                    [ss.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for ss in s]
                )
                speakers.extend([ss.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER)
                                for ss in s])
                src_langs.extend([ss.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG)
                                for ss in s])
                tgt_langs.extend([ss.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG)
                                for ss in s])
                phones.extend([ss['phones'] for ss in s])

        return MultilingualTripletDataset(
            split_name,
            is_train_split,
            data_cfg,
            audio_paths,
            n_frames,
            src_texts,
            tgt_texts,
            phones,
            speakers,
            src_langs,
            tgt_langs,
            ids,
            tgt_dict,
            src_dict,
            phone_dict,
            pre_tokenizer,
            bpe_tokenizer,
            src_bpe_tokenizer,
            normalize,
            mask,
            src_lang2idx=src_lang2idx,
            labeled=labeled
        )

    @classmethod
    def from_tsv(
        cls,
        root: str,
        data_cfg: MultilingualTripletDataConfig,
        splits: str,
        tgt_dict,
        src_dict,
        phone_dict,
        pre_tokenizer,
        bpe_tokenizer,
        src_bpe_tokenizer,
        is_train_split: bool,
        epoch: int,
        seed: int,
        normalize: bool = False,
        mask: bool = True,
        sample_rate: int = 16000,
    ) -> MultilingualTripletDataset:
        samples = []
        _splits = splits.split(",") # here the splits contain codes of language pair that we use
        src_lang2idx = {}
        types = [False, False]
        for split in _splits:
            parts = split.split('_')
            if len(parts) == 1: # unlabelled data
                src_code = parts[0]
                voxp_root = data_cfg.voxpopuli_root
                assert voxp_root is not None
                tsv_path = op.join(voxp_root, 'unlabelled_data', src_code, "{}.tsv".format(src_code))
                if src_code not in src_lang2idx:
                    src_lang2idx[src_code] = len(src_lang2idx)
                types[0] = True
            else:
                src_code, tgt_code, tp, tag = split.split('_')
                tsv_path = op.join(root, src_code, "{}_st_{}_{}_{}.tsv".format(tp, src_code, tgt_code, tag))
                types[1] = True
            if not op.isfile(tsv_path):
                raise FileNotFoundError(f"Dataset not found: {tsv_path}")
            with open(tsv_path) as f:
                reader = csv.DictReader(
                    f,
                    delimiter="\t",
                    quotechar=None,
                    doublequote=False,
                    lineterminator="\n",
                    quoting=csv.QUOTE_NONE,
                )
                samples.append([dict(e) for e in reader])
                assert len(samples) > 0

        datasets = [
            cls._from_list(
                name,
                is_train_split,
                [s],
                data_cfg,
                tgt_dict,
                src_dict,
                phone_dict,
                pre_tokenizer,
                bpe_tokenizer,
                src_bpe_tokenizer,
                normalize,
                mask,
                sample_rate,
                src_lang2idx
            )
            for name, s in zip(_splits, samples)
        ]

        if is_train_split and len(_splits) > 1 and \
                data_cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls._get_size_ratios(
                _splits, [len(dataset.n_frames) for dataset in datasets],
                alpha=data_cfg.sampling_alpha
            )
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for d, r in zip(datasets, size_ratios)
            ]

        # sampling ratio for unlabeled dataset
        if all(types):
            unlabeled_r = data_cfg.unlabeled_sampling_ratio

            labeled_dataset = ConcatDataset([d for d in datasets if d.labeled])
            unlabeled_dataset = ConcatDataset([
                ResamplingDataset(
                    d, size_ratio=unlabeled_r, seed=seed, epoch=epoch, replace=(unlabeled_r >= 1.0)
                ) for d in datasets if not d.labeled
            ])

            datasets = OrderedDict({
                'labeled': labeled_dataset,
                'unlabeled': unlabeled_dataset
            })

            configs = OrderedDict({
                'labeled': data_cfg,
                'unlabeled': data_cfg
            })

            ratios = OrderedDict({
                'labeled': 1,
                'unlabeled': 1
            })

            return JointDataset(datasets, configs, ratios)

        return ConcatDataset(datasets)
