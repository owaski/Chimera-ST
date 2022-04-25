from random import choices
import torch as th
import torch.nn.functional as F
from examples.speech_recognition.data.data_utils import encoder_padding_mask_to_lengths, padding_mask_to_lengths

from fairseq import logging, metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

@register_criterion('multilingual_triplet_phone_criterion')
class MultilingualTripletContrastiveCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self, 
        task, 
        sentence_avg, 
        label_smoothing, 
        ignore_prefix_size=0,
        report_accuracy=False, 
        loss_ratio=[1.0, 1.0, 1.0, 1.0, 1.0],
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.loss_ratio = loss_ratio

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
            default=[1.0, 1.0, 1.0, 1.0, 1.0], # st, mt, asr, ctc on phone, ctc on text
            type=float,
            nargs='+'
        )

    def forward(self, model, sample, reduce=True):
        st_loss = st_nll_loss = th.tensor(0.)
        mt_loss = mt_nll_loss = th.tensor(0.)
        asr_loss = asr_nll_loss = th.tensor(0.)
        ctc_phone = ctc_text = th.tensor(0.)

        if self.loss_ratio[0] > 0:
            st_net_output, st_encoder_out = model.forward_with_internal(**sample["net_input"])
            st_loss, st_nll_loss = self.compute_loss(model, st_net_output, sample["target"], reduce=reduce)

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

        if self.loss_ratio[3] > 0:
            encoder_lengths = padding_mask_to_lengths(st_encoder_out.encoder_padding_mask) - 1
            ctc_phone = F.ctc_loss(
                st_encoder_out.phone_logp.transpose(0, 1).float(), sample["phones"], 
                encoder_lengths, sample["phone_lengths"],
                blank=0, reduction='sum' if reduce else 'none'
            )

        if self.loss_ratio[4] > 0:
            encoder_lengths = padding_mask_to_lengths(st_encoder_out.encoder_padding_mask)

            src_text = []
            for idx in range(sample["src_text"].size(0)):
                src_sent, src_len = sample["src_text"][idx], sample["src_text_lengths"][idx]
                src_text.append(src_sent[:src_len])
            src_text = th.cat(src_text, dim=0).long()

            ctc_text = F.ctc_loss(
                st_encoder_out.text_logp.transpose(0, 1).float(), src_text,
                encoder_lengths, sample["src_text_lengths"],
                blank=st_encoder_out.text_logp.size(-1) - 1, reduction='sum' if reduce else 'none'
            )

        loss = self.loss_ratio[0] * st_loss + \
               self.loss_ratio[1] * mt_loss + \
               self.loss_ratio[2] * asr_loss + \
               self.loss_ratio[3] * ctc_phone + \
               self.loss_ratio[4] * ctc_text
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
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "asr_sample_size": asr_sample_size,
            "adv_enc_sample_size": sample["target"].size(0),
            "adv_disc_sample_size": sample["target"].size(0),
            "ctc_phone_loss": ctc_phone if self.loss_ratio[3] > 0 else 0,
            "ctc_phone_sample_size": encoder_lengths.sum() if self.loss_ratio[3] > 0 else 0,
            "ctc_text_loss": ctc_text if self.loss_ratio[4] > 0 else 0,
            "ctc_text_sample_size": encoder_lengths.sum() if self.loss_ratio[4] > 0 else 0,
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

        # discriminator
        adv_enc_sample_size = sum(log.get('adv_enc_sample_size', 0) for log in logging_outputs)
        adv_enc_loss_sum = sum(log.get('adv_enc_loss', 0) for log in logging_outputs)
        if adv_enc_sample_size > 0:
            metrics.log_scalar('adv_enc_loss', adv_enc_loss_sum / adv_enc_sample_size, adv_enc_sample_size, round=3)

        adv_disc_sample_size = sum(log.get('adv_disc_sample_size', 0) for log in logging_outputs)
        adv_disc_loss_sum = sum(log.get('adv_disc_loss', 0) for log in logging_outputs)
        if adv_disc_sample_size > 0:
            metrics.log_scalar('adv_disc_loss', adv_disc_loss_sum / adv_disc_sample_size, adv_disc_sample_size, round=3)

        # ctc
        ctc_phone_sample_size = sum(log.get('ctc_phone_sample_size', 0) for log in logging_outputs)
        ctc_phone_loss_sum = sum(log.get('ctc_phone_loss', 0) for log in logging_outputs)
        if ctc_phone_sample_size > 0:
            metrics.log_scalar('ctc_phone_loss', ctc_phone_loss_sum / ctc_phone_sample_size, ctc_phone_sample_size, round=3)

        ctc_text_sample_size = sum(log.get('ctc_text_sample_size', 0) for log in logging_outputs)
        ctc_text_loss_sum = sum(log.get('ctc_text_loss', 0) for log in logging_outputs)
        if ctc_text_sample_size > 0:
            metrics.log_scalar('ctc_text_loss', ctc_text_loss_sum / ctc_text_sample_size, ctc_text_sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False