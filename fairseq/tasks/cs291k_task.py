import os
import logging
from argparse import Namespace

import numpy as np
import torch as th

from fairseq import utils, metrics, criterions
from fairseq.data import encoders
from fairseq.data.audio.cs291k_dataset import CS291KDataConfig, CS291KDataset, CS291KDatasetCreator
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset, get_features_or_waveform
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass.configs import GenerationConfig
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import LegacyFairseqTask

logger = logging.getLogger(__name__)

EVAL_BLEU_ORDER = 4

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

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        parser.add_argument('--eval-bleu-bpe', type=str, metavar='BPE',
                            default=None,
                            help='args for building the bpe, if needed')
        parser.add_argument('--eval-bleu-bpe-path', type=str, metavar='BPE',
                            help='args for building the bpe, if needed')
    
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
            self.args.data,
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
            self.src_bpe = None
            if getattr(self.data_cfg, 'src_bpe_tokenizer', None): # None if use phoneme src
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

    def build_model(self, args):
        model = super(CS291KTask, self).build_model(args)

        if getattr(args, "eval_bleu", False):
            import json
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args))
            if args.eval_bleu_bpe is None:
                self.bpe = None
            else:
                logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
                self.bpe = encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.sequence_generator = self.build_generator([model], GenerationConfig(**gen_args))

        return model

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

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                escape_unk=escape_unk,
            )
            if self.bpe is not None:
                s = self.bpe.decode(s)
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyp = decode(gen_out[i][0]["tokens"])
            ref = decode(
                utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            )
            # if self.args.lang_prefix_tok is not None:
            #     hyp = hyp.replace(self.args.lang_prefix_tok, "")
            #     ref = ref.replace(self.args.lang_prefix_tok, "")
            hyps.append(hyp)
            refs.append(ref)

        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:
            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            for i in range(EVAL_BLEU_ORDER):
                if type(counts[i]) is th.Tensor:
                    counts[i] = counts[i].cpu()
                if type(totals[i]) is th.Tensor:
                    totals[i] = totals[i].cpu()

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)