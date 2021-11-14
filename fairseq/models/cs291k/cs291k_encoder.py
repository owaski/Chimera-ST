import torch as th
import torch.nn as nn
import torch.nn.functional as F
from fairseq.data.data_utils import lengths_to_mask, lengths_to_padding_mask

from fairseq.models.fairseq_encoder import EncoderOut, FairseqEncoder
from fairseq.models.transformer import Linear
from fairseq.models.wav2vec import Wav2Vec2Model
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.layer_norm import LayerNorm
from fairseq.modules.positional_embedding import PositionalEmbedding
from fairseq.modules.transformer_layer import TransformerEncoderLayer

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

        self.dropout = FairseqDropout(p=args.dropout, module_name=self.__class__.__name__)
        self.padding_idx = 1
        
        self.cif_proj = Linear(self.w2v_args.encoder_embed_dim - 1, args.encoder_embed_dim)

        self.text_embedding = encoder_embedding
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        )

        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )

        self.layer_norm = None
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)

    def _get_w2v_feature(self, src_tokens, src_lengths):
        '''
            src_tokens: b * n_frame
            src_lengths: b
        '''
        padding_mask = lengths_to_padding_mask(src_lengths)
        w2v_feature, padding_mask = self.w2v_model.extract_features(src_tokens, padding_mask)
        output_lengths = (1 - padding_mask.long()).sum(dim=1)
        return w2v_feature, padding_mask, output_lengths

    def max_positions(self):
        return None

    def _cif(self, src_feature, padding_mask, src_lengths, src_text_lengths=None):
        '''
            src_feature: b * l * dim
            src_lengths: b
        '''
        device = src_feature.device
        bs, l = src_feature.size()[:2]
        alpha = F.sigmoid(src_feature[:, :, 0]).squeeze(-1)
        alpha[padding_mask] = 0.
        sum_alpha = alpha.sum(dim=1)
        if src_text_lengths is not None:
            scale_factor = src_text_lengths / sum_alpha
            alpha = alpha * scale_factor.unsqueeze(-1)
        alpha_cumsum = alpha.cumsum(dim=1)

        alpha_cumsum_trunc = alpha_cumsum.long()
        diff_mask = alpha_cumsum_trunc[:, :-1] != alpha_cumsum_trunc[:, 1:]
        batch_features = []
        for i in range(bs):
            indices = th.arange(l - 1).to(device)
            separation = th.cat([th.tensor([0]).to(device), indices[diff_mask[i]] + 1])
            if separation[-1] != src_lengths[i] - 1:
                separation = th.cat([separation, th.tensor([src_lengths[i] - 1]).to(device)], dim=0)
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
                if lb + 1 < rb:
                    cur_feature += (
                        alpha[i, lb + 1 : rb].unsqueeze(-1) * src_feature[i, lb + 1 : rb]
                    ).sum(dim=0)
                    cur_sum_alpha += alpha[i, lb + 1 : rb].sum()
                cur_feature += (alpha_cumsum[i, lb] - vlb) * src_feature[i, lb]
                cur_feature += (vrb - alpha_cumsum[i, rb - 1]) * src_feature[i, rb]
                cur_sum_alpha += (alpha_cumsum[i, lb] - vlb) + (vrb - alpha_cumsum[i, rb - 1])

                if cur_sum_alpha > 0.5:
                    features.append(cur_feature.unsqueeze(0))

            # deal with >1 alpha after scaling
            last_idx = separation[-1]
            if alpha[i, last_idx] >= 1:
                n_rep = alpha[i, last_idx].long()
                features.extend([src_feature[i, last_idx].unsqueeze(0) for _ in range(n_rep)])

            batch_features.append(th.cat(features, dim=0))
        
        output_lengths = th.tensor([features.size(0) for features in batch_features]).to(device)
        assert ~(output_lengths != src_text_lengths).any()
        max_length = output_lengths.max()
        output_features = []
        for feature in batch_features:
            if feature.size(0) < max_length:
                padding = th.zeros(max_length - feature.size(0), feature.size(1)).to(feature)
                feature = th.cat([feature, padding], dim=0)
            output_features.append(feature.unsqueeze(0))
        output_features = th.cat(output_features, dim=0)
        output_features = self.cif_proj(output_features[:, :, 1:])

        return output_features.transpose(0, 1), output_lengths, sum_alpha

    def forward(self, src_tokens, src_lengths, src_text_lengths=None, **extra_args):
        is_text = not src_tokens.dtype.is_floating_point
        if is_text:
            embedding = self.text_embedding(src_tokens).transpose(0, 1)
            input_lengths = src_lengths
            sum_alpha = None
        else:
            w2v_feature, padding_mask, w2v_lengths = self._get_w2v_feature(src_tokens, src_lengths)
            embedding, input_lengths, sum_alpha = self._cif(w2v_feature, padding_mask, \
                w2v_lengths, src_text_lengths)
        
        internal_states = {'sum_alpha': sum_alpha, 'feature': embedding}

        x = embedding
        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        if is_text:
            positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            x += positions
        x = self.dropout(x)

        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)

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

