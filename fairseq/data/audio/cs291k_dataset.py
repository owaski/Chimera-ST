import os
import csv
import logging
from typing import Dict, List, Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from fairseq.data import (
    Dictionary,
    ConcatDataset,
    data_utils as fairseq_data_utils, 
    ResamplingDataset
)
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig, SpeechToTextDataset, SpeechToTextDatasetCreator, 
    get_features_or_waveform, _collate_frames
)
from fairseq.data.concat_dataset import ConcatDataset

logger = logging.getLogger(__name__)

class CS291KDataConfig(S2TDataConfig):
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

    
class CS291KDataset(SpeechToTextDataset):

    def __init__(
        self,
        split: str,
        is_train_split: bool,
        data_cfg: CS291KDataConfig,
        audio_paths: List[str],
        n_frames: List[int],
        src_texts: Optional[List[str]] = None,
        tgt_texts: Optional[List[str]] = None,
        speakers: Optional[List[str]] = None,
        src_langs: Optional[List[str]] = None,
        tgt_langs: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        src_dict: Optional[Dictionary] = None,
        tgt_dict: Optional[Dictionary] = None,
        pre_tokenizer=None,
        src_bpe_tokenizer=None,
        bpe_tokenizer=None,
        normalize=False,
        sample_rate=16000,
    ):
        self.split, self.is_train_split = split, is_train_split
        self.data_cfg = data_cfg
        self.audio_paths, self.n_frames = audio_paths, n_frames
        self.n_samples = len(audio_paths)
        assert len(n_frames) == self.n_samples > 0
        assert src_texts is None or len(src_texts) == self.n_samples
        assert tgt_texts is None or len(tgt_texts) == self.n_samples
        assert speakers is None or len(speakers) == self.n_samples
        assert src_langs is None or len(src_langs) == self.n_samples
        assert tgt_langs is None or len(tgt_langs) == self.n_samples
        assert ids is None or len(ids) == self.n_samples
        assert (tgt_dict is None and tgt_texts is None) or (
            tgt_dict is not None and tgt_texts is not None
        )
        assert (src_dict is None and src_texts is None) or (
            src_dict is not None and src_texts is not None
        )
        self.src_texts, self.tgt_texts = src_texts, tgt_texts
        self.src_langs, self.tgt_langs = src_langs, tgt_langs
        self.tgt_dict = tgt_dict
        self.src_dict = src_dict
        self.check_tgt_lang_tag()
        self.ids = ids
        self.shuffle = data_cfg.shuffle if is_train_split else False

        self.feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.data_cfg.get_feature_transforms(split, is_train_split)
        )

        self.pre_tokenizer = pre_tokenizer
        self.bpe_tokenizer = bpe_tokenizer
        self.src_bpe_tokenizer = src_bpe_tokenizer
        self.normalize = normalize
        self.sample_rate = sample_rate

        logger.info(self.__repr__())

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

    def __getitem__(self, index: int):
        source = get_features_or_waveform(
            self.audio_paths[index],
            need_waveform=self.data_cfg.use_audio_input,
            sample_rate=self.sample_rate,
        )
        if self.feature_transforms is not None:
            assert not self.data_cfg.use_audio_input
            source = self.feature_transforms(source)
        source = th.from_numpy(source).float()
        if self.normalize:
            with th.no_grad():
                source = F.layer_norm(source, source.shape)
        
        target = None
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index], 'target')
            target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.data_cfg.prepend_tgt_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format(self.tgt_langs[index])
                lang_tag_idx = self.tgt_dict.index(lang_tag)
                target = th.cat([th.tensor([lang_tag_idx]).long(), target], dim=0)
        
        src_text = None
        if self.src_texts is not None:
            src_tokenized = self.tokenize_text(self.src_texts[index], 'source')
            source = self.src_dict.encode_line(
                src_tokenized, add_if_not_exist=False, append_eos=True
            ).long()
        
        return index, source, target, src_text

    def collater(self, samples) -> Dict:
        if len(samples) == 0:
            return {}
        indices = th.tensor([i for i, _, _, _ in samples]).long()
        frames = _collate_frames([s for _, s, _, _ in samples], self.data_cfg.use_audio_input)

        # sort samples by descending number of frames
        n_frames = th.tensor([s.size(0) for _, s, _, _ in samples]).long()
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        if self.tgt_texts is not None:
            target = fairseq_data_utils.collate_tokens(
                [t for _, _, t, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False
            )
            target = target.index_select(0, order)
            target_lengths = th.tensor(
                [t.size(0) for _, _, t, _ in samples]
            ).long().index_select(0, order)
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [t for _, _, t, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            ).index_select(0, order)
            ntokens = target_lengths.sum()
        
        src_text, src_text_lengths = None, None
        if self.src_texts is not None:
            src_text = fairseq_data_utils.collate_tokens(
                [s for _, _, _, s in samples],
                self.src_dict.pad(),
                self.src_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False
            ).index_select(0, order)
            src_text_lengths = th.tensor(
                [s.size(0) for _, _, _, s in samples]
            ).long().index_select(0, order)

        out = {
            "id": indices,
            "mode": "st",
            "net_input": {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": prev_output_tokens,
                "src_text_lengths": src_text_lengths,
            },
            "target": target,
            "target_lengths": target_lengths,
            "src_text": src_text,
            "src_text_lengths": src_text_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples)
        }
        return out

    def size(self, index):
        # TODO: check if dataset.size() is used
        t_len = 0
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index], 'target')
            t_len = len(tokenized.split(' '))
        return self.n_frames[index], t_len

class CS291KDatasetCreator(SpeechToTextDatasetCreator):

    @classmethod
    def _from_list(
        cls, 
        split_name: str, 
        is_train_split, 
        samples: List[List[Dict]], 
        data_cfg: CS291KDataConfig, 
        src_dict,
        tgt_dict, 
        pre_tokenizer, 
        src_bpe_tokenizer,
        bpe_tokenizer, 
        normalize,
    ) -> CS291KDataset:
        audio_paths, n_frames, src_texts, tgt_texts, ids = [], [], [], [], []
        speakers, src_langs, tgt_langs = [], [], []
        for s in samples:
            ids.extend([ss[cls.KEY_ID] for ss in s])
            audio_paths.extend([os.path.join(data_cfg.audio_root, ss[cls.KEY_AUDIO]) for ss in s])
            n_frames.extend([int(ss[cls.KEY_N_FRAMES]) for ss in s])
            tgt_texts.extend([ss[cls.KEY_TGT_TEXT] for ss in s])
            src_texts.extend([ss.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for ss in s])
            speakers.extend([ss.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for ss in s])
            src_langs.extend([ss.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for ss in s])
            tgt_langs.extend([ss.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for ss in s])

        return CS291KDataset(
            split_name,
            is_train_split,
            data_cfg,
            audio_paths,
            n_frames,
            src_texts,
            tgt_texts,
            speakers,
            src_langs,
            tgt_langs,
            ids,
            src_dict,
            tgt_dict,
            pre_tokenizer,
            src_bpe_tokenizer,
            bpe_tokenizer,
            normalize,
        )

    @classmethod
    def from_tsv(
        cls, 
        root: str, 
        data_cfg: CS291KDataConfig, 
        splits: str, 
        src_dict,
        tgt_dict, 
        pre_tokenizer, 
        src_bpe_tokenizer, 
        bpe_tokenizer, 
        is_train_split: bool, 
        epoch: int, 
        seed: int, 
        normalize: bool = False, 
    ) -> CS291KDataset:
        samples = []
        _splits = splits.split(',')
        for split in _splits:
            tsv_path = os.path.join(root, f"{split}.tsv")
            if not os.path.isfile(tsv_path):
                raise FileNotFoundError(f"Dataset not found: {tsv_path}")
            with open(tsv_path) as f:
                reader = csv.DictReader(
                    f,
                    delimiter='\t',
                    quotechar=None,
                    doublequote=False,
                    lineterminator='\n',
                    quoting=csv.QUOTE_NONE
                )
                samples.append([dict(e) for e in reader])
                assert len(samples[-1]) > 0
        
        datasets = [
            cls._from_list(
                name,
                is_train_split,
                [s],
                data_cfg,
                src_dict,
                tgt_dict,
                pre_tokenizer,
                src_bpe_tokenizer,
                bpe_tokenizer,
                normalize,
            )
            for name, s in zip(_splits, samples)
        ]

        if is_train_split and len(_splits) > 1 and data_cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls._get_size_ratios(
                _splits, [len(s) for s in samples],
                alpha=data_cfg.sampling_alpha
            )
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for d, r in zip(datasets, size_ratios)
            ]
        
        return ConcatDataset(datasets)
        