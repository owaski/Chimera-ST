import os
import logging
from argparse import Namespace

from fairseq import criterions
from fairseq.data import encoders
from fairseq.data.audio.cs291k_dataset import CS291KDataConfig, CS291KDataset, CS291KDatasetCreator
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset, get_features_or_waveform
from fairseq.data.dictionary import Dictionary
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import LegacyFairseqTask

logger = logging.getLogger(__name__)

@register_task('cs291k_task')
class CS291KTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            'data',
            help='Manifest root path'
        )
        parser.add_argument(
            '--normalize',
            action='store_true',
            help='If set, normalizes input to have 0 mean and unit variance'
        )
        parser.add_argument(
            '--config-yaml',
            type=str,
            default='config.yaml',
            help='Configuration YAML filename (under manifest root)'
        )
        parser.add_argument(
            '--max-source-positions',
            type=int,
            default=6000,
            help='Max number of tokens in the source sequence'
        )
        parser.add_argument(
            '--max-target-positions',
            type=int,
            default=1024,
            help='Max number of tokens in the target sequence'
        )
        parser.add_argument(
            '--sample-rate',
            type=int,
            default=16000,
            help='Sample rate of audio'
        )
    
    def __init__(self, args, src_dict, tgt_dict, data_cfg):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.data_cfg = data_cfg

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = CS291KDataConfig(os.path.join(args.data, args.config_yaml))

        def load_dict(vocab_filename):
            _dict_path = os.path.join(args.data, vocab_filename)
            if not os.path.exists(_dict_path):
                raise FileNotFoundError('Dict not found: {}'.format(_dict_path))
            _dict = Dictionary.load(_dict_path)
            return _dict

        src_dict = load_dict(data_cfg.src_vocab_filename)
        tgt_dict = load_dict(data_cfg.vocab_filename)
        logger.info(
            'source dictionary size ({}): {}'.format(data_cfg.src_vocab_filename, len(src_dict))
        )
        logger.info(
            'target dictionary size ({}): {}'.format(data_cfg.vocab_filename, len(tgt_dict))
        )

        return cls(args, src_dict, tgt_dict, data_cfg)

    def build_criterion(self, args):
        if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                'target language ID token is prepended as BOS.'
            )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith('train')
        pre_tokenizer = self.build_tokenizer(self.args)
        src_bpe_tokenizer = self.build_bpe('source')
        bpe_tokenizer = self.build_bpe('target')
        self.datasets[split] = CS291KDatasetCreator.from_tsv(
            self.data,
            self.data_cfg,
            split,
            self.src_dict,
            self.tgt_dict,
            pre_tokenizer,
            src_bpe_tokenizer,
            bpe_tokenizer,
            is_train_split,
            epoch,
            self.args.seed,
            self.args.normalize,
        )

    def build_tokenizer(self, tokenizer_config):
        logger.info('pre-tokenizer: {}'.format(self.data_cfg.pre_tokenizer))
        self.tokenizer = encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))
        return self.tokenizer

    def build_bpe(self, side):
        logger.info('{} tokenizer: {}'.format(side, self.data_cfg.bpe_tokenizer))
        if side == 'target':
            self.bpe = encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))
            return self.bpe
        else:
            self.src_bpe = encoders.build_bpe(Namespace(**self.data_cfg.src_bpe_tokenizer))
            return self.src_bpe

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_generator(self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None):
        if self.data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
            raise ValueError(
                'Please set "--prefix-size 1" since '
                'target language ID token is prepended as BOS.'
            )
        lang_token_ids = {
            i for s, i in self.tgt_dict.indices.items() if SpeechToTextDataset.is_lang_tag(s)
        }
        extra_gen_cls_kwargs = {"symbols_to_strip_from_output": lang_token_ids}
        return super().build_generator(
            models, args, seq_gen_cls=seq_gen_cls, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return CS291KDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p, True).shape[0] for p in lines]
        return lines, n_frames