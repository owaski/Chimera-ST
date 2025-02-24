import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from fairseq.data.data_utils import lengths_to_mask, lengths_to_padding_mask
from fairseq.models.fairseq_encoder import EncoderOut, FairseqEncoder
from fairseq.models.transformer import Linear
from fairseq.models.wav2vec import Wav2Vec2Model
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.layer_norm import LayerNorm
from fairseq.modules.positional_embedding import PositionalEmbedding
from fairseq.modules.transformer_layer import TransformerEncoderLayer
from fairseq.modules.grad_multiply import GradMultiply

class CS291KEncoder(FairseqEncoder):
    '''Speech-to-text Transformer encoder that consists of
    input wav2vec2Encoder, CIF and Transformer encoder.'''

    def __init__(self, args, encoder_embedding):
        super().__init__(None)

        assert args.w2v2_model_path is not None
        self.w2v2_model_path = args.w2v2_model_path

        w2v_ckpt = th.load(self.w2v2_model_path)
        self.w2v_args = w2v_ckpt['args']
        self.w2v_model = Wav2Vec2Model.build_model(self.w2v_args, task=None)
        self.w2v_model.load_state_dict(w2v_ckpt['model'])
        self.w2v2_grad_mult = getattr(args, "w2v2_grad_mult", 1.0)

        self.dropout = FairseqDropout(p=args.dropout, module_name=self.__class__.__name__)
        self.padding_idx = 1
        
        self.cif_proj = Linear(self.w2v_args.encoder_embed_dim - 1, args.encoder_embed_dim)

        self.text_embedding = encoder_embedding
        self.embed_scale = 1.0 if args.no_scale_embedding else np.sqrt(args.encoder_embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        )

        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )

        self.layer_norm = None
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)

        self.cif_avg_pool = getattr(args, 'cif_avg_pool', False)
        self.fix_cif = getattr(args, 'fix_cif', False)
        self.align_after_encoder = getattr(args, 'align_after_encoder', -1)
        self.cnn_subsampler = None
        if args.cnn_subsampler:
            self.cnn_subsampler = Conv1dSubsampler(
                self.w2v_args.encoder_embed_dim,
                args.conv_channels,
                args.encoder_embed_dim,
                [int(k) for k in args.conv_kernel_sizes.split(",")],
            )

        
        self.no_shrink = getattr(args, 'no_shrink', False)
        if self.no_shrink:
            self.proj = nn.Linear(self.w2v_args.encoder_embed_dim, args.encoder_embed_dim)

        # self.sum_src_length = 0.
        # self.sum_src_text_length = 0.

    def _get_w2v_feature(self, src_tokens, src_lengths):
        '''
            src_tokens: b * n_frame
            src_lengths: b
        '''
        padding_mask = lengths_to_padding_mask(src_lengths)
        w2v_feature, padding_mask = self.w2v_model.extract_features(src_tokens, padding_mask)
        output_lengths = (1 - padding_mask.long()).sum(dim=1)
        if self.w2v2_grad_mult != 1.0:
            GradMultiply.apply(w2v_feature, self.w2v2_grad_mult)
        return w2v_feature, padding_mask, output_lengths

    def max_positions(self):
        return None

    def _cif(self, src_feature, padding_mask, src_lengths, src_group_lengths=None):
        '''
            src_feature: b * l * dim
            src_lengths: b
        '''
        
        # th.save(self.cif_proj(src_feature[:, :, 1:]), '/home/ubuntu/work/experiments/tmp/st_full_feature.pt')

        device = src_feature.device
        is_fp16 = src_feature.dtype == th.float16
        bs = src_feature.size()[0]
        alpha = th.sigmoid(src_feature[:, :, 0].float())

        src_feature = src_feature[:, :, 1:]
        alpha = alpha.masked_fill(padding_mask, 0.)

        # th.save(alpha, '/home/ubuntu/work/experiments/tmp/alpha.pt')

        sum_alpha = alpha.sum(dim=1)
        if src_group_lengths is not None:
            scale_factor = src_group_lengths / sum_alpha
            alpha = alpha * scale_factor.unsqueeze(-1)

        # self.sum_src_length += src_lengths.sum()
        # self.sum_src_text_length += src_text_lengths.sum()
        # print(self.sum_src_length / self.sum_src_text_length)

        if self.fix_cif > 0:
            alpha[...] = self.fix_cif

        alpha_cumsum = alpha.cumsum(dim=1)

        alpha_cumsum_trunc = alpha_cumsum.long()
        diff_mask = alpha_cumsum_trunc[:, :-1] != alpha_cumsum_trunc[:, 1:]
        batch_features = []
        for i in range(bs):
            cur_len = src_lengths[i]
            indices = th.arange(cur_len - 1).to(device)
            separation = th.cat([th.tensor([0]).to(device), indices[diff_mask[i, :cur_len - 1]] + 1])

            # print(sum_alpha, cur_len, separation / 49)

            if separation[-1] != cur_len - 1:
                separation = th.cat([separation, th.tensor([cur_len - 1]).to(device)], dim=0)
            features = []
            for j in range(1, separation.size(0)):
                lb, rb = separation[j - 1 : j + 1]
                vlb = alpha_cumsum_trunc[i, lb]
                vrb = min(alpha_cumsum_trunc[i, rb - 1] + 1, alpha_cumsum[i, rb])

                # deal with >1 alpha after scaling
                n_rep = alpha_cumsum_trunc[i, lb] - (alpha_cumsum_trunc[i, lb - 1] + 1 if lb > 0 else 0)
                if n_rep >= 1:
                    features.extend([src_feature[i, lb].unsqueeze(0) for _ in range(n_rep)])

                cur_feature = th.zeros_like(src_feature[0, 0])
                cur_sum_alpha = 0.

                if not self.cif_avg_pool:
                    if lb + 1 < rb:
                        cur_feature += (
                            alpha[i, lb + 1 : rb].unsqueeze(-1) * src_feature[i, lb + 1 : rb]
                        ).sum(dim=0)
                        cur_sum_alpha += alpha[i, lb + 1 : rb].sum()
                    
                    cur_feature += (alpha_cumsum[i, lb] - vlb) * src_feature[i, lb]
                    cur_feature += (vrb - alpha_cumsum[i, rb - 1]) * src_feature[i, rb]
                    
                    cur_sum_alpha += (alpha_cumsum[i, lb] - vlb) + (vrb - alpha_cumsum[i, rb - 1])
                else:
                    cur_feature += src_feature[i, lb:rb + 1].mean(dim=0)
                    if lb + 1 < rb:
                        cur_sum_alpha += alpha[i, lb + 1 : rb].sum()
                    cur_sum_alpha += (alpha_cumsum[i, lb] - vlb) + (vrb - alpha_cumsum[i, rb - 1])

                if cur_sum_alpha > 0.5 or len(features) == 0:
                    features.append(cur_feature.unsqueeze(0))

            # deal with >1 alpha after scaling
            last_idx = separation[-1]
            if alpha[i, last_idx] >= 1:
                n_rep = alpha[i, last_idx].long()
                features.extend([src_feature[i, last_idx].unsqueeze(0) for _ in range(n_rep)])

            batch_features.append(th.cat(features, dim=0))
        
        output_lengths = th.tensor([features.size(0) for features in batch_features]).to(device)
        if src_group_lengths is not None and not self.fix_cif:
            assert ~(output_lengths != src_group_lengths).any(), (output_lengths, src_group_lengths)
        max_length = output_lengths.max()
        output_features = []
        for feature in batch_features:
            if feature.size(0) < max_length:
                padding = th.zeros(max_length - feature.size(0), feature.size(1)).to(feature)
                feature = th.cat([feature, padding], dim=0)
            output_features.append(feature.unsqueeze(0))
        output_features = th.cat(output_features, dim=0)
        output_features = self.cif_proj(output_features).transpose(0, 1)

        if is_fp16:
            output_features = output_features.half()
            sum_alpha = sum_alpha.half()

        return output_features, output_lengths, sum_alpha
        
    def forward(self, src_tokens, src_lengths, src_group_lengths=None, **extra_args):
        # print(src_tokens.size())
        
        is_text = not src_tokens.dtype.is_floating_point
        if is_text:
            embedding = self.text_embedding(src_tokens).transpose(0, 1)
            input_lengths = src_lengths
            sum_alpha = None
        else:
            w2v_feature, padding_mask, w2v_lengths = self._get_w2v_feature(src_tokens, src_lengths)
            sum_alpha = 0.
            if self.no_shrink:
                embedding = self.proj(w2v_feature).transpose(0, 1)
                input_lengths = w2v_lengths
            elif self.cnn_subsampler is None:
                embedding, input_lengths, sum_alpha = self._cif(w2v_feature, padding_mask, \
                    w2v_lengths, src_group_lengths)
            else:
                embedding, input_lengths = self.cnn_subsampler(w2v_feature, w2v_lengths)
        
        if self.align_after_encoder == -1:
            internal_states = {'sum_alpha': sum_alpha, 'feature': embedding}

        x = embedding
        x = self.embed_scale * x
        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = self.dropout(x)

        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, encoder_padding_mask)
            if i == self.align_after_encoder:
                internal_states = {'sum_alpha': sum_alpha, 'feature': x}

        return EncoderOut(
            encoder_out=x,
            encoder_padding_mask=encoder_padding_mask,
            internal_states=internal_states,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )          

    @th.jit.export
    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_out.encoder_padding_mask
            if encoder_out.encoder_padding_mask is None
            else encoder_out.encoder_padding_mask.index_select(0, new_order)
        )
        new_internal_feature = (
            encoder_out.internal_states['feature']
            if encoder_out.internal_states['feature'] is None
            else encoder_out.internal_states['feature'].index_select(1, new_order)
        )
        new_internal_states = {
            'sum_alpha': encoder_out.internal_states['sum_alpha'], 
            'feature': new_internal_feature
        }
        return EncoderOut(
            encoder_out=new_encoder_out,
            encoder_padding_mask=new_encoder_padding_mask,
            internal_states=new_internal_states,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )

