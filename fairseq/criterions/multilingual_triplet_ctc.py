import torch as th
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

from examples.speech_recognition.data.data_utils import padding_mask_to_lengths

@register_criterion('multilingual_triplet_ctc_criterion')
class MultilingualTripletCTCCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self, 
        task, 
        sentence_avg, 
        label_smoothing, 
        ignore_prefix_size=0,
        report_accuracy=False, 
        loss_ratio=[1.0, 1.0, 1.0],
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
            default=[1.0, 1.0, 1.0], # st, mt, ctc
            type=float,
            nargs='+'
        )

    def forward(self, model, sample, reduce=True):
        st_loss = st_nll_loss = th.tensor(0.)
        mt_loss = mt_nll_loss = th.tensor(0.)
        ctc_loss = th.tensor(0.)

        target_sample_size = 0

        if self.loss_ratio[0] > 0 and sample["st_indices"].size(0) > 0:
            st_net_output = model(**sample["net_input"])
            st_loss, st_nll_loss = self.compute_loss(model, st_net_output, sample["target"], reduce=reduce)
            target_sample_size = sample["target_lengths"].sum()

        if self.loss_ratio[1] > 0 and sample["st_indices"].size(0) > 0:
            src_text = sample["src_text"][sample["st_indices"]]
            src_text_lengths = sample["src_text_lengths"][sample["st_indices"]]
            src_text = src_text[:, :src_text_lengths.max()]
            mt_input = {
                "src_tokens": src_text,
                "src_lengths": src_text_lengths,
                "prev_output_tokens": sample["net_input"]["prev_output_tokens"]
            }
            mt_net_output = model(**mt_input)
            mt_loss, mt_nll_loss = self.compute_loss(model, mt_net_output, sample["target"], reduce=reduce)
        
        ctc_sample_size = 0
        if self.loss_ratio[2] > 0:
            t_input = {
                "src_tokens": sample["src_text"],
                "src_lengths": sample["src_text_lengths"],
            }

            s_enc_out = model.encoder(**sample["ctc_input"])
            with th.no_grad():
                t_enc_out = model.encoder(**t_input)

            ctc_loss, ctc_sample_size = self.compute_ctc(
                s_enc_out, t_enc_out, model
            )

        # print(ctc_loss, ctc_sample_size)

        loss = self.loss_ratio[0] * st_loss + \
               self.loss_ratio[1] * mt_loss + \
               self.loss_ratio[2] * ctc_loss
        nll_loss = self.loss_ratio[0] * st_nll_loss + \
                   self.loss_ratio[1] * mt_nll_loss

        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "st_loss": st_loss.data if self.loss_ratio[0] > 0 else 0,
            "st_nll_loss": st_nll_loss.data if self.loss_ratio[0] > 0 else 0,
            "mt_loss": mt_loss.data if self.loss_ratio[1] > 0 else 0,
            "mt_nll_loss": mt_nll_loss.data if self.loss_ratio[1] > 0 else 0,
            "ctc_loss": ctc_loss.data if self.loss_ratio[2] > 0 else 0,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0) if sample["target"] is not None else 0,
            "sample_size": sample_size,
            "target_sample_size": target_sample_size,
            "ctc_sample_size": ctc_sample_size
        }

        if self.report_accuracy:
            if self.loss_ratio[0] > 0 and sample["st_indices"].size(0) > 0:
                st_n_correct, st_total = self.compute_accuracy(model, st_net_output, sample["target"])
                logging_output["st_n_correct"] = utils.item(st_n_correct.data)
                logging_output["st_total"] = utils.item(st_total.data)

            if self.loss_ratio[1] > 0 and sample["st_indices"].size(0) > 0:
                mt_n_correct, mt_total = self.compute_accuracy(model, mt_net_output, sample["target"])
                logging_output["mt_n_correct"] = utils.item(mt_n_correct.data)
                logging_output["mt_total"] = utils.item(mt_total.data)

        return loss, sample_size, logging_output

    def compute_ctc(self, s_enc_out, t_enc_out, model):
        s_x, t_x = s_enc_out.encoder_out, t_enc_out.encoder_out

        s_l = padding_mask_to_lengths(s_enc_out.encoder_padding_mask)
        t_l = padding_mask_to_lengths(t_enc_out.encoder_padding_mask)

        targets = []
        for l in t_l:
            targets.append(th.arange(1, l + 1).long().to(s_x.device))
        targets = th.cat(targets, dim=0)

        sim_matrix = (s_x.unsqueeze(1) * t_x.unsqueeze(0)).sum(dim=-1).transpose(1, 2) # T * B * C

        silence_logit = (s_x * model.encoder.silence_emb.view(1, 1, -1)).sum(dim=-1).unsqueeze(-1) # T * B * 1

        sim_matrix = th.cat([silence_logit, sim_matrix], dim=-1)

        logp = F.log_softmax(sim_matrix, dim=-1)

        ctc_loss = F.ctc_loss(
            logp, targets,
            s_l, t_l, 
            blank=0,
            reduction='sum',
        )
        ctc_sample_size = targets.size(0)

        return ctc_loss, ctc_sample_size

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        if sample_size > 0:
            _sum = sum(log.get('loss', 0) for log in logging_outputs)
            metrics.log_scalar('loss', _sum / sample_size, sample_size, round=3)

        target_sample_size = sum(log.get('target_sample_size', 0) for log in logging_outputs)
        if target_sample_size > 0:
            for name in ('st_loss', 'st_nll_loss', 'mt_loss', 'mt_nll_loss'):
                _sum = sum(log.get(name, 0) for log in logging_outputs)
                metrics.log_scalar(name, _sum / target_sample_size, target_sample_size, round=3)

        # metrics.log_scalar('ntokens', sum(log.get('ntokens', 0) for log in logging_outputs))
        
        if target_sample_size > 0:
            for name in ('', 'st_', 'mt_'):
                _sum = sum(log.get(name + 'nll_loss', 0) for log in logging_outputs)
                metrics.log_scalar(name + 'ppl', utils.get_perplexity(_sum / target_sample_size, base=th.e))

        for name in ('st_', 'mt_'):
            total = utils.item(sum(log.get(name + 'total', 0) for log in logging_outputs))
            if total > 0:
                metrics.log_scalar(name + 'total', total)
                n_correct = utils.item(sum(log.get(name + 'n_correct', 0) for log in logging_outputs))
                metrics.log_scalar(name + 'n_correct', n_correct)
                metrics.log_scalar(name + 'accuracy', n_correct / total)

        # ctc
        ctc_sample_size = sum(log.get('ctc_sample_size', 0) for log in logging_outputs)
        ctc_loss_sum = sum(log.get('ctc_loss', 0) for log in logging_outputs)
        # print(ctc_loss_sum, ctc_sample_size)
        if ctc_sample_size > 0:
            metrics.log_scalar('ctc_loss', ctc_loss_sum / ctc_sample_size, ctc_sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False