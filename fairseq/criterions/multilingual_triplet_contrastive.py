from random import choices
import torch as th
import torch.nn.functional as F
from examples.speech_recognition.data.data_utils import encoder_padding_mask_to_lengths, padding_mask_to_lengths

from fairseq import logging, metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

@register_criterion('multilingual_triplet_contrastive_criterion')
class MultilingualTripletContrastiveCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self, 
        task, 
        sentence_avg, 
        label_smoothing, 
        ignore_prefix_size=0,
        report_accuracy=False, 
        loss_ratio=[1.0, 1.0, 1.0, 1.0],
        gamma=1.0,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.loss_ratio = loss_ratio
        self.gamma = gamma

    @staticmethod
    def get_num_updates():
        return metrics.get_smoothed_values("train").get("num_updates", 0)

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--label-smoothing', 
            default=0., 
            type=float, 
            metavar='EPS',
            help='Epsilon for lable smoothing, 0 means none'
        )
        parser.add_argument(
            '--report-accuracy', 
            action='store_true',
            help='Report accuracy metric'
        )
        parser.add_argument(
            '--ignore-prefix-size', 
            default=0, 
            type=int, 
            metavar='N',
            help='Ignore first N tokens'
        )
        parser.add_argument(
            '--loss-ratio',
            default=[1.0, 1.0, 1.0, 1.0], # st, mt, asr, contrastive
            type=float,
            nargs='+'
        )
        parser.add_argument(
            '--gamma',
            default=1.0,
            type=float,
        )

    def forward(self, model, sample, reduce=True):
        st_loss = st_nll_loss = th.tensor(0.)
        mt_loss = mt_nll_loss = th.tensor(0.)
        asr_loss = asr_nll_loss = th.tensor(0.)
        const_loss = th.tensor(0.)

        if self.loss_ratio[0] > 0:
            st_net_output, st_encoder_out = model.forward_with_internal(**sample["net_input"])
            st_loss, st_nll_loss = self.compute_loss(model, st_net_output, sample["target"], reduce=reduce)

            if sample["mixed_src_tokens"] is not None:
                mixed_st_input = {
                    "src_tokens": sample["mixed_src_tokens"],
                    "src_lengths": sample["mixed_src_lengths"],
                    "prev_output_tokens": sample["net_input"]["prev_output_tokens"][sample["mixed_indices"]],
                    "mask": sample["net_input"]["mask"],
                    "src_lang_tag_indices": sample["net_input"]["src_lang_tag_indices"],
                }
                mixed_st_net_output, mixed_st_encoder_out = model.forward_with_internal(**mixed_st_input)
                mixed_st_loss, mixed_st_nll_loss = self.compute_loss(model, mixed_st_net_output, sample["target"][sample["mixed_indices"]], reduce=reduce)

                st_loss = st_loss + mixed_st_loss
                st_nll_loss = st_nll_loss + mixed_st_nll_loss

        mt_loss = mt_nll_loss = 0.
        if self.loss_ratio[1] > 0:
            mt_input = {
                "src_tokens": sample["src_text"],
                "src_lengths": sample["src_text_lengths"],
                "prev_output_tokens": sample["net_input"]["prev_output_tokens"]
            }
            mt_net_output = model(**mt_input)
            mt_loss, mt_nll_loss = self.compute_loss(model, mt_net_output, sample["target"], reduce=reduce)
        
        if self.loss_ratio[2] > 0:
            asr_input = {
                "src_tokens": sample["net_input"]["src_tokens"], 
                "src_lengths": sample["net_input"]["src_lengths"], 
                "prev_output_tokens": sample["asr_prev_output_tokens"],
                "src_lang_tag_indices": sample["net_input"]["src_lang_tag_indices"]
            }
            asr_net_output = model(**asr_input)
            asr_loss, asr_nll_loss = self.compute_loss(model, asr_net_output, sample["asr_target"], reduce=reduce)

            if sample["mixed_src_tokens"] is not None:
                mixed_asr_input = {
                    "src_tokens": sample["mixed_src_tokens"], 
                    "src_lengths": sample["mixed_src_lengths"], 
                    "prev_output_tokens": sample["asr_prev_output_tokens"][sample["mixed_indices"]],
                    "src_lang_tag_indices": sample["net_input"]["src_lang_tag_indices"]
                }
                mixed_asr_net_output = model(**mixed_asr_input)
                mixed_asr_loss, mixed_asr_nll_loss = self.compute_loss(model, mixed_asr_net_output, sample["asr_target"][sample["mixed_indices"]], reduce=reduce)

                asr_loss = asr_loss + mixed_asr_loss
                asr_nll_loss = asr_nll_loss + mixed_asr_nll_loss
        
        const_sample_size = 0
        if sample["mixed_src_tokens"] is not None and self.loss_ratio[3] > 0:
            const_loss, const_sample_size = self.contrastive(
                st_encoder_out.encoder_out, mixed_st_encoder_out.encoder_out,
                st_encoder_out.encoder_padding_mask, mixed_st_encoder_out.encoder_padding_mask, 
                sample["matches"], sample["mixed_indices"]
            )

        loss = self.loss_ratio[0] * st_loss + \
               self.loss_ratio[1] * mt_loss + \
               self.loss_ratio[2] * asr_loss + \
               self.loss_ratio[3] * const_loss
        nll_loss = self.loss_ratio[0] * st_nll_loss + \
                   self.loss_ratio[1] * mt_nll_loss + \
                   self.loss_ratio[2] * asr_nll_loss

        sample_size = asr_sample_size = 0
        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        asr_sample_size = sample["asr_target"].size(0) if self.sentence_avg else sample["asr_target_lengths"].sum()

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "st_loss": st_loss.data if self.loss_ratio[0] > 0 else 0,
            "st_nll_loss": st_nll_loss.data if self.loss_ratio[0] > 0 else 0,
            "mt_loss": mt_loss.data if self.loss_ratio[1] > 0 else 0,
            "mt_nll_loss": mt_nll_loss.data if self.loss_ratio[1] > 0 else 0,
            "asr_loss": asr_loss.data if self.loss_ratio[2] > 0 else 0,
            "asr_nll_loss": asr_nll_loss.data if self.loss_ratio[2] > 0 else 0,
            "const_loss": const_loss.data if sample["mixed_src_tokens"] is not None and self.loss_ratio[3] > 0 else 0,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "asr_sample_size": asr_sample_size,
            "const_sample_size": const_sample_size
        }

        if self.report_accuracy:
            if self.loss_ratio[0] > 0:
                st_n_correct, st_total = self.compute_accuracy(model, st_net_output, sample["target"])
                logging_output["st_n_correct"] = utils.item(st_n_correct.data)
                logging_output["st_total"] = utils.item(st_total.data)

            if self.loss_ratio[1] > 0:
                mt_n_correct, mt_total = self.compute_accuracy(model, mt_net_output, sample["target"])
                logging_output["mt_n_correct"] = utils.item(mt_n_correct.data)
                logging_output["mt_total"] = utils.item(mt_total.data)

            if self.loss_ratio[2] > 0:
                asr_n_correct, asr_total = self.compute_accuracy(model, asr_net_output, sample["asr_target"])
                logging_output["asr_n_correct"] = utils.item(asr_n_correct.data)
                logging_output["asr_total"] = utils.item(asr_total.data)

        return loss, sample_size, logging_output

    def contrastive(self, orig_x, mixed_x, orig_padding_mask, mixed_padding_mask, matches, indices):
        orig_length = padding_mask_to_lengths(orig_padding_mask)
        mixed_length = padding_mask_to_lengths(mixed_padding_mask)

        orig_x = orig_x.float()
        mixed_x = mixed_x.float()

        # assume orig_x and mixed_x be (L * B * H)
        _, bsz, _ = orig_x.size()
        const_loss = th.tensor(0.).to(orig_x.device)
        const_sample_size = 0

        for mixed_idx, orig_idx in enumerate(indices):
            orig_len = orig_length[orig_idx]
            mixed_len = mixed_length[mixed_idx]
            mix_mask, orig_norm_itv, mixed_norm_itv = matches[mixed_idx]
            if mix_mask.sum() > 0:
                orig_itv = orig_norm_itv * orig_len
                orig_itv[:, 0] = orig_itv[:, 0].floor()
                orig_itv[:, 1] = orig_itv[:, 1].ceil()
                orig_itv = orig_itv.long()

                mixed_itv = mixed_norm_itv * mixed_len
                mixed_itv[:, 0] = mixed_itv[:, 0].floor()
                mixed_itv[:, 1] = mixed_itv[:, 1].ceil()
                mixed_itv = mixed_itv.long()

                orig_ft = []
                mixed_ft = []
                for orig_rg, mixed_rg in zip(orig_itv, mixed_itv):
                    orig_ft.append(orig_x[orig_rg[0] : orig_rg[1], orig_idx].mean(dim=0, keepdim=True))
                    mixed_ft.append(mixed_x[mixed_rg[0] : mixed_rg[1], mixed_idx].mean(dim=0, keepdim=True))
                orig_ft = th.cat(orig_ft, dim=0)
                mixed_ft = th.cat(mixed_ft, dim=0)

                sim_matrix = F.cosine_similarity(
                    orig_ft.unsqueeze(0),
                    mixed_ft.unsqueeze(1),
                    dim=-1
                ) / self.gamma

                labels = th.arange(orig_ft.size(0))[mix_mask].to(orig_ft).long()
                const_loss = const_loss + F.cross_entropy(sim_matrix[mix_mask, :], labels, reduction='sum')
                const_loss = const_loss + F.cross_entropy(sim_matrix[:, mix_mask].T, labels, reduction='sum')
                const_sample_size += mix_mask.sum() * 2

        return const_loss, const_sample_size
                

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        if sample_size > 0:
            for name in ('loss', 'nll_loss', 'st_loss', 'st_nll_loss', 'mt_loss', 'mt_nll_loss'):
                _sum = sum(log.get(name, 0) for log in logging_outputs)
                metrics.log_scalar(name, _sum / sample_size, sample_size, round=3)

        asr_sample_size = sum(log.get('asr_sample_size', 0) for log in logging_outputs)
        if asr_sample_size > 0:
            for name in ('asr_loss', 'asr_nll_loss'):
                _sum = sum(log.get(name, 0) for log in logging_outputs)
                metrics.log_scalar(name, _sum / asr_sample_size, asr_sample_size, round=3)

        const_sample_size = sum(log.get('const_sample_size', 0) for log in logging_outputs)
        if const_sample_size > 0:
            _sum = sum(log.get('const_loss', 0) for log in logging_outputs)
            metrics.log_scalar('const_loss', _sum / const_sample_size, const_sample_size, round=3)

        # metrics.log_scalar('ntokens', sum(log.get('ntokens', 0) for log in logging_outputs))
        
        if sample_size > 0:
            for name in ('', 'st_', 'mt_', 'asr_'):
                _sum = sum(log.get(name + 'nll_loss', 0) for log in logging_outputs)
                metrics.log_scalar(name + 'ppl', utils.get_perplexity(_sum / sample_size, base=th.e))

        for name in ('st_', 'mt_', 'asr_'):
            total = utils.item(sum(log.get(name + 'total', 0) for log in logging_outputs))
            if total > 0:
                metrics.log_scalar(name + 'total', total)
                n_correct = utils.item(sum(log.get(name + 'n_correct', 0) for log in logging_outputs))
                metrics.log_scalar(name + 'n_correct', n_correct)
                metrics.log_scalar(name + 'accuracy', n_correct / total)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False