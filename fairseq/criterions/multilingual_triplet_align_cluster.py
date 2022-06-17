import numpy as np
from sklearn import cluster
import torch as th
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

from examples.speech_recognition.data.data_utils import padding_mask_to_lengths

@register_criterion('multilingual_triplet_align_cluster_criterion')
class MultilingualTripletAlignClusterCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self, 
        task, 
        sentence_avg, 
        label_smoothing, 
        ignore_prefix_size=0,
        report_accuracy=False, 
        loss_ratio=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        cluster_gamma=0.02,
        gamma=0.05,
        use_emb=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        loss_mask = np.array(loss_ratio) != 0.
        assert loss_mask[-3:].sum() <= 1
        self.loss_ratio = loss_ratio
        self.cluster_gamma = cluster_gamma
        self.gamma = gamma
        self.use_emb = use_emb

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
            default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # st, mt, asr, adv, token-level, sent-level, ctc
            type=float,
            nargs='+'
        )
        parser.add_argument(
            '--cluster-gamma',
            default=0.02,
            type=float,
        )
        parser.add_argument(
            '--gamma',
            default=0.05,
            type=float,
        )
        parser.add_argument(
            '--use-emb',
            action='store_true'
        )

    def forward(self, model, sample, reduce=True):
        st_loss = st_nll_loss = th.tensor(0.)
        mt_loss = mt_nll_loss = th.tensor(0.)
        asr_loss = asr_nll_loss = th.tensor(0.)
        cluster_loss = th.tensor(0.)
        align_token_loss = th.tensor(0.)
        align_sent_loss = th.tensor(0.)
        align_ctc_loss = th.tensor(0.)

        target_sample_size = 0
        asr_sample_size = 0
        align_sample_size = 0
        cluster_sample_size = 0

        if sample["st_indices"].size(0) > 0:

            if self.loss_ratio[0] > 0:
                st_net_output = model(**sample["net_input"])
                st_loss, st_nll_loss = self.compute_loss(model, st_net_output, sample["target"], reduce=reduce)
                target_sample_size = sample["target_lengths"].sum()

            if self.loss_ratio[1] > 0:
                mt_net_output = model(**sample["mt_input"])
                mt_loss, mt_nll_loss = self.compute_loss(model, mt_net_output, sample["target"], reduce=reduce)

            if self.loss_ratio[2] > 0:
                asr_net_output = model(**sample["asr_input"])
                asr_loss, asr_nll_loss = self.compute_loss(model, asr_net_output, sample["asr_target"], reduce=reduce)
                asr_sample_size = sample["asr_target_lengths"].sum()
        
        if sample["align_indices"].size(0) > 0:
            
            if self.loss_ratio[4] > 0:
                s_input = {
                    "src_tokens": sample["align_inputs"]["src_tokens"],
                    "src_lengths": sample["align_inputs"]["src_lengths"],
                }
                s_enc_out = model.encoder(**s_input)

                t_input = {
                    "src_tokens": sample["align_inputs"]["src_text"],
                    "src_lengths": sample["align_inputs"]["src_text_lengths"],
                }
                with th.no_grad():
                    t_enc_out = model.encoder(**t_input)

                align_token_loss, align_sample_size = self.compute_align(
                    s_enc_out, t_enc_out, sample["align"]
                )

            if self.loss_ratio[5] > 0:
                s_input = {
                    "src_tokens": sample["align_inputs"]["src_tokens"],
                    "src_lengths": sample["align_inputs"]["src_lengths"],
                }
                s_enc_out = model.encoder(**s_input)

                t_input = {
                    "src_tokens": sample["align_inputs"]["src_text"],
                    "src_lengths": sample["align_inputs"]["src_text_lengths"],
                }
                with th.no_grad():
                    t_enc_out = model.encoder(**t_input)

                align_sent_loss, align_sample_size = self.compute_sent_align(
                    s_enc_out, t_enc_out, sample["align"]
                )

            if self.loss_ratio[6] > 0:
                s_input = {
                    "src_tokens": sample["align_inputs"]["src_tokens"],
                    "src_lengths": sample["align_inputs"]["src_lengths"],
                }
                s_enc_out = model.encoder(**s_input)

                align_ctc_loss, align_sample_size = self.compute_ctc(
                    s_enc_out, model, 
                    sample["align_inputs"]["src_text"], sample["align_inputs"]["src_text_lengths"]
                )

        if self.loss_ratio[3] > 0 and sample["cluster_indices"].size(0) > 0:
            s_encoder_out = model.encoder(**sample["cluster_s_inputs"])
            with th.no_grad():
                sc_s_encoder_out = model.encoder(**sample["cluster_positive_s_inputs"])
            s_cluster_loss, s_cluster_sample_size = self.compute_cluster_contrastive(s_encoder_out, sc_s_encoder_out)

            t_encoder_out = model.encoder(**sample["cluster_t_inputs"])
            with th.no_grad():
                sc_t_encoder_out = model.encoder(**sample["cluster_positive_t_inputs"])
            t_cluster_loss, t_cluster_sample_size = self.compute_cluster_contrastive(t_encoder_out, sc_t_encoder_out)

            cluster_loss = s_cluster_loss + t_cluster_loss
            cluster_sample_size = s_cluster_sample_size + t_cluster_sample_size

        loss = self.loss_ratio[0] * st_loss + \
               self.loss_ratio[1] * mt_loss + \
               self.loss_ratio[2] * asr_loss + \
               self.loss_ratio[3] * cluster_loss + \
               self.loss_ratio[4] * align_token_loss + \
               self.loss_ratio[5] * align_sent_loss + \
               self.loss_ratio[6] * align_ctc_loss
        nll_loss = self.loss_ratio[0] * st_nll_loss + \
                   self.loss_ratio[1] * mt_nll_loss + \
                   self.loss_ratio[2] * asr_nll_loss

        sample_size = sample["ntokens"]

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "st_loss": st_loss.data if self.loss_ratio[0] > 0 else 0,
            "st_nll_loss": st_nll_loss.data if self.loss_ratio[0] > 0 else 0,
            "mt_loss": mt_loss.data if self.loss_ratio[1] > 0 else 0,
            "mt_nll_loss": mt_nll_loss.data if self.loss_ratio[1] > 0 else 0,
            "asr_loss": asr_loss.data if self.loss_ratio[2] > 0 else 0,
            "asr_nll_loss": asr_nll_loss.data if self.loss_ratio[2] > 0 else 0,
            "align_loss": align_token_loss.data + align_sent_loss.data + align_ctc_loss.data if sum(self.loss_ratio[-3:]) > 0 else 0,
            "cluster_loss": cluster_loss.data if self.loss_ratio[3] > 0 else 0,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0) if sample["target"] is not None else 0,
            "sample_size": sample_size,
            "target_sample_size": target_sample_size,
            "asr_sample_size": asr_sample_size,
            "align_sample_size": align_sample_size,
            "cluster_sample_size": cluster_sample_size,
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

            if self.loss_ratio[2] > 0 and sample["st_indices"].size(0) > 0:
                asr_n_correct, asr_total = self.compute_accuracy(model, asr_net_output, sample["asr_target"])
                logging_output["asr_n_correct"] = utils.item(asr_n_correct.data)
                logging_output["asr_total"] = utils.item(asr_total.data)

        return loss, sample_size, logging_output

    def compute_align(self, s_enc_out, t_enc_out, align):
        if self.use_emb:
            s_x, t_x = s_enc_out.encoder_embedding.transpose(0, 1), t_enc_out.encoder_embedding.transpose(0, 1)
        else:
            s_x, t_x = s_enc_out.encoder_out, t_enc_out.encoder_out

        s_len = padding_mask_to_lengths(s_enc_out.encoder_padding_mask)
        t_len = padding_mask_to_lengths(t_enc_out.encoder_padding_mask)

        # targets = []
        # for l in t_len:
        #     targets.append(th.arange(1, l + 1).long().to(s_x.device))
        # targets = th.cat(targets, dim=0)

        bsz = s_x.size(1)

        align_loss = th.tensor(0.)
        align_sample_size = 0

        s_f = []
        t_f = []

        for i in range(bsz):
            segment, interval = align[i]
            
            for (t_l, t_r), (s_l, s_r) in zip(segment, interval):
                s_l = int((s_l * s_len[i]).floor())
                s_r = int((s_r * s_len[i]).ceil())

                t_feature = t_x[t_l : t_r + 1, i].mean(dim=0)
                s_feature = s_x[s_l : s_r + 1, i].mean(dim=0)

                s_f.append(s_feature)
                t_f.append(t_feature)

        s_f = th.stack(s_f, dim=0)
        t_f = th.stack(t_f, dim=0)

        logits = F.cosine_similarity(
            s_f.unsqueeze(1),
            t_f.unsqueeze(0),
            dim=-1
        ) / self.gamma

        label = th.arange(s_f.size(0)).to(logits.device)

        align_loss = F.cross_entropy(logits, label, reduction='sum')
        align_sample_size = s_f.size(0)
        
        return align_loss, align_sample_size
    
    def compute_sent_align(self, s_enc_out, t_enc_out, align):
        if self.use_emb:
            s_x, t_x = s_enc_out.encoder_embedding.transpose(0, 1), t_enc_out.encoder_embedding.transpose(0, 1)
        else:
            s_x, t_x = s_enc_out.encoder_out, t_enc_out.encoder_out

        s_len = padding_mask_to_lengths(s_enc_out.encoder_padding_mask)
        t_len = padding_mask_to_lengths(t_enc_out.encoder_padding_mask)

        # targets = []
        # for l in t_len:
        #     targets.append(th.arange(1, l + 1).long().to(s_x.device))
        # targets = th.cat(targets, dim=0)

        bsz = s_x.size(1)

        align_loss = th.tensor(0.)
        align_sample_size = 0

        s_f = []
        t_f = []

        for i in range(bsz):
            t_feature = t_x[:t_len[i], i].mean(dim=0)
            s_feature = s_x[:s_len[i], i].mean(dim=0)

            s_f.append(s_feature)
            t_f.append(t_feature)

        s_f = th.stack(s_f, dim=0)
        t_f = th.stack(t_f, dim=0)

        logits = F.cosine_similarity(
            s_f.unsqueeze(1),
            t_f.unsqueeze(0),
            dim=-1
        ) / self.gamma

        label = th.arange(s_f.size(0)).to(logits.device)

        align_loss = F.cross_entropy(logits, label, reduction='sum')
        align_sample_size = s_f.size(0)
        
        return align_loss, align_sample_size

    def compute_ctc(self, s_enc_out, model, src_text, src_text_lengths):
        s_x = s_enc_out.encoder_embedding.transpose(0, 1)
        s_len = padding_mask_to_lengths(s_enc_out.encoder_padding_mask)

        align_sample_size = 0

        logits = th.matmul(s_x, model.encoder.text_embedding.weight.T)
        silence_logit = th.matmul(s_x, model.encoder.silence_emb.unsqueeze(-1))
        logits = th.cat([silence_logit, logits], dim=-1)
        logps = F.log_softmax(logits, dim=-1)

        align_loss = F.ctc_loss(
            logps, src_text + 1,
            s_len, src_text_lengths,
            reduction='sum'
        )
        align_sample_size = src_text_lengths.sum()

        return align_loss, align_sample_size

    def compute_cluster_contrastive(self, enc_out, pos_enc_out):
        x, pos_x = enc_out.encoder_out, pos_enc_out.encoder_out

        len = padding_mask_to_lengths(enc_out.encoder_padding_mask)
        pos_len = padding_mask_to_lengths(pos_enc_out.encoder_padding_mask)

        bsz = x.size(1)

        f = []
        pos_f = []
        for i in range(bsz):
            feature = x[:len[i], i].mean(dim=0)
            pos_feature = pos_x[:pos_len[i], i].mean(dim=0)

            f.append(feature)
            pos_f.append(pos_feature)

        f = th.stack(f, dim=0)
        pos_f = th.stack(pos_f, dim=0)

        sim_matrix_1 = F.cosine_similarity(
            f.unsqueeze(1),
            f.unsqueeze(0),
            dim=-1
        )
        sim_matrix_1.fill_diagonal_(-1.)
        sim_matrix_2 = F.cosine_similarity(
            f.unsqueeze(1),
            pos_f.unsqueeze(0),
            dim=-1
        )
        
        logits = th.cat(
            [sim_matrix_2, sim_matrix_1], dim=1
        ) / self.cluster_gamma
        label = th.arange(logits.size(0)).to(logits.device)
        
        cluster_loss = F.cross_entropy(logits, label, reduction='sum')
        cluster_sample_size = bsz

        return cluster_loss, cluster_sample_size


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

        asr_sample_size = sum(log.get('asr_sample_size', 0) for log in logging_outputs)
        if asr_sample_size > 0:
            for name in ('asr_loss', 'asr_nll_loss'):
                _sum = sum(log.get(name, 0) for log in logging_outputs)
                metrics.log_scalar(name, _sum / asr_sample_size, asr_sample_size, round=3)

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

        if asr_sample_size > 0:
            _sum = sum(log.get('asr_nll_loss', 0) for log in logging_outputs)
            metrics.log_scalar('asr_ppl', utils.get_perplexity(_sum / asr_sample_size, base=th.e))
            
            total = utils.item(sum(log.get('asr_total', 0) for log in logging_outputs))
            if total > 0:
                metrics.log_scalar('asr_total', total)
                n_correct = utils.item(sum(log.get('asr_n_correct', 0) for log in logging_outputs))
                metrics.log_scalar('asr_n_correct', n_correct)
                metrics.log_scalar('asr_accuracy', n_correct / total)

        # align
        align_sample_size = sum(log.get('align_sample_size', 0) for log in logging_outputs)
        align_loss_sum = sum(log.get('align_loss', 0) for log in logging_outputs)
        # print(ctc_loss_sum, ctc_sample_size)
        if align_sample_size > 0:
            metrics.log_scalar('align_loss', align_loss_sum / align_sample_size, align_sample_size, round=3)

        cluster_sample_size = sum(log.get('cluster_sample_size', 0) for log in logging_outputs)
        cluster_loss_sum = sum(log.get('cluster_loss', 0) for log in logging_outputs)
        # print(ctc_loss_sum, ctc_sample_size)
        if cluster_sample_size > 0:
            metrics.log_scalar('cluster_loss', cluster_loss_sum / cluster_sample_size, cluster_sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False