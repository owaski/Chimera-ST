import copy
from typing import Optional
from argparse import Namespace
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq.data.data_utils import lengths_to_mask, lengths_to_padding_mask
from fairseq.model_parallel.models.transformer import TransformerEncoder
from fairseq.models.fairseq_encoder import EncoderOut, FairseqEncoder
from fairseq.models.transformer import Linear
from fairseq.models.wav2vec import Wav2Vec2Model
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.layer_norm import LayerNorm
from fairseq.modules import LayerDropModuleList, TransformerEncoderLayer
from fairseq.modules.positional_embedding import PositionalEmbedding
from fairseq.modules.grad_multiply import GradMultiply
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

class W2V2TransformerEncoder(FairseqEncoder):
    '''Speech-to-text Transformer encoder that consists of
    input wav2vec2Encoder, CIF and Transformer encoder.'''

    def __init__(self, args, src_dict, encoder_embedding):
        super().__init__(None)

        assert args.w2v2_model_path is not None
        self.w2v2_model_path = args.w2v2_model_path

        w2v2_ckpt = th.load(self.w2v2_model_path)
        self.w2v2_args = Namespace(**w2v2_ckpt['cfg']['model'])
        self.w2v2_model = Wav2Vec2Model.build_model(self.w2v2_args, task=None)
        self.w2v2_model.load_state_dict(w2v2_ckpt['model'])
        self.w2v2_grad_mult = getattr(args, "w2v2_grad_mult", 1.0)

        self.dropout = FairseqDropout(p=args.dropout, module_name=self.__class__.__name__)
        self.padding_idx = 1
        
        self.text_embedding = encoder_embedding

        t_args = copy.deepcopy(args)
        t_args.max_source_positions = min(t_args.max_source_positions, 1024)
        self.transformer_encoder = _TransformerEncoder(t_args, src_dict, encoder_embedding)

        self.embed_scale = 1.0 if args.no_scale_embedding else np.sqrt(args.encoder_embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        ) # This is for audio only

        # self.transformer_layers = nn.ModuleList(
        #     [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        # )

        # self.layer_norm = None
        # if args.encoder_normalize_before:
        #     self.layer_norm = LayerNorm(args.encoder_embed_dim)

        self.cnn_subsampler = None
        if args.cnn_subsampler:
            self.cnn_subsampler = Conv1dSubsampler(
                self.w2v2_args.encoder_embed_dim,
                args.conv_channels,
                args.encoder_embed_dim,
                [int(k) for k in args.conv_kernel_sizes.split(",")],
            )

    def _get_w2v2_feature(self, src_tokens, src_lengths):
        '''
            src_tokens: b * n_frame
            src_lengths: b
        '''
        padding_mask = lengths_to_padding_mask(src_lengths)
        w2v2_feature, padding_mask = self.w2v2_model.extract_features(src_tokens, padding_mask)
        output_lengths = (1 - padding_mask.long()).sum(dim=1)
        if self.w2v2_grad_mult != 1.0:
            GradMultiply.apply(w2v2_feature, self.w2v2_grad_mult)
        return w2v2_feature, padding_mask, output_lengths

    def max_positions(self):
        return None
        
    def forward(self, src_tokens, src_lengths, **extra_args):       
        is_text = not src_tokens.dtype.is_floating_point
        if is_text:
            embedding = self.text_embedding(src_tokens)
            input_lengths = src_lengths
            embed_positions = self.transformer_encoder.embed_positions
        else:
            w2v2_feature, w2v2_padding_mask, w2v2_lengths = self._get_w2v2_feature(src_tokens, src_lengths)
            embedding, input_lengths = self.cnn_subsampler(w2v2_feature, w2v2_lengths)
            embedding = embedding.transpose(0, 1)
            embed_positions = self.embed_positions

        x = self.embed_scale * embedding

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = embed_positions(encoder_padding_mask)
        x = x + positions
        if self.transformer_encoder.layernorm_embedding is not None:
            x = self.transformer_encoder.layernorm_embedding(x)
        x = self.transformer_encoder.dropout_module(x)
        
        # x = embedding
        # x = self.embed_scale * x
        # encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        # positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        # x += positions
        # x = self.dropout(x)

        # for _, layer in enumerate(self.transformer_layers):
        #     x = layer(x, encoder_padding_mask)

        transformer_encoder_out = self.transformer_encoder(x, input_lengths, return_all_hiddens=True)

        return EncoderOut(
            encoder_out=transformer_encoder_out.encoder_out,
            encoder_padding_mask=transformer_encoder_out.encoder_padding_mask,
            internal_states=None, 
            encoder_embedding=transformer_encoder_out.encoder_embedding,
            encoder_states=transformer_encoder_out.encoder_states,
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
        new_encoder_embedding = (
            encoder_out.encoder_embedding
            if encoder_out.encoder_embedding is None
            else encoder_out.encoder_embedding.index_select(0, new_order)
        )
        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)
        internal_states = encoder_out.internal_states
        if internal_states is not None:
            for idx, state in enumerate(internal_states):
                internal_states[idx] = state.index_select(0, new_order)        
        return EncoderOut(
            encoder_out=new_encoder_out,
            encoder_padding_mask=new_encoder_padding_mask,
            internal_states=None,
            encoder_embedding=new_encoder_embedding,
            encoder_states=encoder_states,
            src_tokens=None,
            src_lengths=None,
        )


class _TransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super(TransformerEncoder, self).__init__(dictionary)
        self.register_buffer("version", th.Tensor([3]))

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args, i) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, args, idx):
        return _TransformerEncoderLayer(args, idx)

    def forward(
        self,
        token_embeddings,
        src_lengths,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = lengths_to_padding_mask(src_lengths)

        # x, encoder_embedding = self.forward_embedding(encoder_padding_mask, token_embeddings)

        # B x T x C -> T x B x C
        x = token_embeddings.transpose(0, 1)

        encoder_states = [] if return_all_hiddens else None
        
        # encoder layers
        for layer in self.layers:
            x, internal = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(internal)

        if self.layer_norm is not None:
            x = self.layer_norm(x)      

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=token_embeddings,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            internal_states=None,
            src_tokens=None,
            src_lengths=None,
        )


class _TransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args, idx):
        super().__init__(args)
        self.remove_residual = idx in args.encoder_layer_to_remove_residual

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(th.bool), -1e8)

        internal_states = [x]

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        internal_states.append(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        internal_states.append(x)
        x = self.dropout_module(x)
        internal_states.append(x)
        if not self.remove_residual: # optional to remove residual connection
            x = self.residual_connection(x, residual)
        internal_states.append(x)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        internal_states.append(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        internal_states.append(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        internal_states.append(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        internal_states.append(x)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, internal_states