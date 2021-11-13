import torch as th
import torch.nn as nn
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

@register_criterion("cs291k_criterion")
class CS291KCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, 
        task, 
        sentence_avg, 
        label_smoothing, 
        loss_ratio, 
        ignore_prefix_size,
        report_accuracy=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.loss_ratio = loss_ratio

    @staticmethod
    def get_num_updates():
        return metrics.get_smoothed_values("train").get("num_updates", 0)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0.,
                            type=float, metavar='EPS',
                            help='Epsilon for lable smoothing, 0 means none')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='Report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0,
                            type=int, metavar='N',
                            help='Ignore first N tokens')
        parser.add_argument('--loss-ratio', default=[1, 1, 1, 1, 1],
                            type=float, nargs='+',
                            help='Ratio of each loss function')

    def forward(self, model, sample, reduce=True):
        # st loss
        if sample["mode"] == "st":
            st_net_output, audio_internal = model.forward_with_internal(**sample["net_input"])
            st_loss, st_nll_loss = self.compute_loss(model, st_net_output, sample, reduce=reduce)

            # mt loss
            if self.loss_ratio[1] > 0:
                mt_input = {
                    "src_tokens": sample["src_text"],
                    "src_lengths": sample["src_text_lengths"],
                    "prev_output_tokens": sample["net_input"]["prev_output_tokens"]
                }
                mt_net_output, text_internal = model.forward_with_internal(**mt_input)
                mt_loss, mt_nll_loss = self.compute_loss(model, mt_net_output, sample, reduce=reduce)
            else:
                mt_loss = mt_nll_loss = 0.

            if self.loss_ratio[2] > 0:
                qua_loss = self.compute_qua(audio_internal, sample["src_text_lengths"], reduce)
            else:
                qua_loss = 0.

            if self.loss_ratio[3] > 0:
                align_loss = self.compute_align(audio_internal, text_internal, reduce)
            else:
                align_loss = 0.

            if self.loss_ratio[4] > 0:
                kd_loss = self.compute_kd(st_net_output, mt_net_output, sample["target"], reduce)

            assert reduce

            loss = sum(self.loss_ratio[i] * part_loss \
                for i, part_loss in enumerate([st_loss, mt_loss, qua_loss, align_loss, kd_loss]))
            nll_loss = sum(self.loss_ratio[i] * part_loss \
                for i, part_loss in enumerate([st_nll_loss, mt_nll_loss]))

            sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            logging_output = {
                "loss": loss.data,
                "nll_loss": nll_loss.data,
                "st_loss": st_loss.data,
                "st_nll_loss": st_nll_loss.data,
                "mt_loss": mt_loss.data if self.loss_ratio[1] > 0 else 0,
                "mt_nll_loss": mt_nll_loss.data if self.loss_ratio[1] > 0 else 0,
                "qua_loss": qua_loss.data if self.loss_ratio[2] > 0 else 0,
                "align_loss": align_loss.data if self.loss_ratio[3] > 0 else 0,
                "kd_loss": kd_loss.data if self.loss_ratio[4] > 0 else 0,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
            }

            if self.report_accuracy:
                n_correct, total = self.compute_accuracy(model, st_net_output, sample)
                logging_output["n_correct"] = utils.item(n_correct.data)
                logging_output["total"] = utils.item(total.data)

            return loss, sample_size, logging_output

        elif sample["mode"] == "mt":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def compute_qua(self, audio_internal, src_lengths, reduce):
        '''
            audio_internal["alpha"]: seqlen * batch * 1
            src_lengths: batch
        '''
        alpha = audio_internal["alpha"].transpose(0, 1) # batch * seqlen * 1
        sum_alpha = alpha.sum(dim=1)
        src_lengths = src_lengths.view(-1, 1)
        qua_loss = F.mse_loss(sum_alpha, src_lengths, reduction='sum' if reduce else 'none')
        return qua_loss
    
    def compute_align(self, audio_internal, text_internal, reduce):
        '''
           audio_internal["feature"]: seqlen * batch * dim
           text_internal["feature"]: seqlen * batch * dim
        '''
        audio_feature = audio_internal["feature"].transpose(0, 1)
        text_feature = text_internal["feature"].detach().transpose(0, 1) # batch * seqlen * dim
        align_loss = F.mse_loss(audio_feature, text_feature, reduction='sum' if reduce else 'none')
        return align_loss

    def compute_kd(self, st_net_output, mt_net_output, target, reduce):
        '''
            st_net_output: batch * seqlen * vocab
            mt_net_output: batch * seqlen * vocab
            target: batch * seqlen
        '''
        st_logit = st_net_output[:, self.ignore_prefix_size:, :]
        mt_logit = mt_net_output[:, self.ignore_prefix_size:, :].detach()
        target = target[:, self.ignore_prefix_size:].contiguous()
        pad_mask = target.eq(self.padding_idx)
        mt_logp = th.log_softmax(mt_logit, dim=-1)
        st_logp = th.log_softmax(st_logit, dim=-1)
        kd_loss = F.kl_div(st_logp, mt_logp, reduce='none', log_target=True)
        kd_loss[pad_mask] = 0.
        if reduce:
            kd_loss = kd_loss.sum()
        return kd_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        # TODO: reduce logging outputs from multiple devices
        pass

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        # TODO: reduceable logging outputs
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False