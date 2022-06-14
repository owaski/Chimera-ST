# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import logging
import os.path as op
from argparse import Namespace
import torch as th
import pickle
import numpy as np

from fairseq import utils, metrics, criterions
from fairseq.data import Dictionary, encoders
from fairseq.data.audio.multilingual_triplet_align_adv_v2_dataset import (
    MultilingualTripletDataConfig,
    MultilingualTripletDataset,
    MultilingualTripletDatasetCreator
)
from fairseq.data.audio.triplet_dataset import (
    get_features_or_waveform,
)
from fairseq.dataclass.configs import GenerationConfig
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset


logger = logging.getLogger(__name__)

EVAL_BLEU_ORDER = 4

@register_task("multilingual_triplet_align_adv_v2_task")
class MultilingualTripletAlignAdvV2Task(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--normalize",
            action="store_true",
            help="if set, normalizes input to have 0 mean and unit variance",
        )
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--dump-feature-to-file",
            type=str, default=None,
        )
        parser.add_argument(
            "--sample-rate", type=int, default=16000
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

    def __init__(self, args, tgt_dict, src_dict, data_cfg):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.src_dict = src_dict
        self.data_cfg = data_cfg
        self.dump_feature_to_file = args.dump_feature_to_file
        if self.dump_feature_to_file is not None:
            self.cached_features = {
                _name: [] for _name in
                ('src_text', 'audio_features', 'text_features')
            }
        else:
            self.cached_features = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = MultilingualTripletDataConfig(op.join(args.data, args.config_yaml))
        
        codes = MultilingualTripletDataset.get_lang_codes(data_cfg.lang_list_filename)

        def load_dict(vocab_filename):
            _dict_path = op.join(args.data, vocab_filename)
            if not op.isfile(_dict_path):
                raise FileNotFoundError(f"Dict not found: {_dict_path}")
            _dict = Dictionary.load(_dict_path)
            for code in codes:
                _dict.add_symbol(MultilingualTripletDataset.LANG_TAG_TEMPLATE.format(code))
            _dict.add_symbol('<mask>')
            return _dict

        tgt_dict = load_dict(data_cfg.vocab_filename)
        src_dict = load_dict(data_cfg.src_vocab_filename)
        logger.info(
            f"target dictionary size ({data_cfg.vocab_filename}): "
            f"{len(tgt_dict):,}"
        )
        logger.info(
            f"source dictionary size ({data_cfg.src_vocab_filename}): "
            f"{len(src_dict):,}"
        )

        return cls(args, tgt_dict, src_dict, data_cfg)

    def build_criterion(self, args):
        if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = 'train' in split
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        src_bpe_tokenizer = self.build_src_bpe()
        self.datasets[split] = MultilingualTripletDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            self.src_dict,
            pre_tokenizer,
            bpe_tokenizer,
            src_bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            normalize=self.args.normalize,
            sample_rate=self.args.sample_rate,
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return self.src_dict

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_model(self, args):
        model = super().build_model(args)

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

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        # if self.data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
        #     raise ValueError(
        #         'Please set "--prefix-size 1" since '
        #         "target language ID token is prepended as BOS."
        #     )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if MultilingualTripletDataset.is_lang_tag(s)
        }
        extra_gen_cls_kwargs = {"symbols_to_strip_from_output": lang_token_ids}
        return super().build_generator(
            models, args, seq_gen_cls=None,
            extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        self.tokenizer = encoders.build_tokenizer(
            Namespace(**self.data_cfg.pre_tokenizer))
        return self.tokenizer

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        self.bpe = encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))
        return self.bpe

    def build_src_bpe(self):
        logger.info(f"source tokenizer: {self.data_cfg.src_bpe_tokenizer}")
        self.src_bpe = encoders.build_bpe(
            Namespace(**self.data_cfg.src_bpe_tokenizer))
        return self.src_bpe

    def dump_features(self):
        if self.cached_features is None:
            return
        with open(self.dump_feature_to_file, 'wb') as f:
            self.cached_features['audio_features'] = np.concatenate(
                self.cached_features['audio_features']
            )
            self.cached_features['text_features'] = np.concatenate(
                self.cached_features['text_features']
            )
            pickle.dump(self.cached_features, f)

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p, True).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return MultilingualTripletDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )

    def valid_step(self, sample, model, criterion):
        if self.dump_feature_to_file is not None:
            model.eval()
            with th.no_grad():
                st_input = sample['net_input']
                mt_input = {
                    "src_tokens": sample["src_text"],
                    "src_lengths": sample["src_text_lengths"],
                    "prev_output_tokens":
                    sample["net_input"]["prev_output_tokens"],
                    "mask": sample["net_input"]["mask"],
                }
                _, audio_internal = model.forward_with_internal(**st_input)
                _, text_internal = model.forward_with_internal(**mt_input)
                self.cached_features['audio_features'].append(
                    audio_internal.detach().cpu().numpy().transpose(1, 0, 2),
                )
                self.cached_features['text_features'].append(
                    text_internal.detach().cpu().numpy().transpose(1, 0, 2),
                )
                self.cached_features['src_text'].extend([
                    self.datasets['dev_wave'].datasets[0].src_texts[i]
                    for i in sample['id']
                ])
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            sample["net_input"]["src_group_lengths"] = None
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

        gen_out = self.inference_step(generator, [model], sample, \
            prefix_tokens=sample['net_input']['prev_output_tokens'][:, 1 : 1 + int(self.data_cfg.prepend_tgt_lang_tag)])
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyp = decode(gen_out[i][0]["tokens"][1:])
            ref = decode(
                utils.strip_pad(sample["target"][i][1:], self.tgt_dict.pad()),
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

                    compute_bleu = sacrebleu.metrics.bleu.BLEU.compute_bleu

                    fn_sig = inspect.getfullargspec(compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)
