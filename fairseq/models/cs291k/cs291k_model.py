import logging
from typing import Optional, Dict

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq import utils
from fairseq.models import (
    register_model, register_model_architecture,
    FairseqEncoderDecoderModel
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import Embedding, TransformerDecoder

from .cs291k_encoder import CS291KEncoder

logger = logging.getLogger(__name__)

@register_model("cs291k_model")
class CS291KModel(FairseqEncoderDecoderModel):
    '''Transformer model for ST/MT tasks'''

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--w2v2-model-path', 
            type=str, 
            metavar='STR',
            help='path to wav2vec2 model'
        )
        parser.add_argument(
            '--activation-fn', 
            type=str, 
            default='relu', 
            choices=utils.get_available_activation_fns(),
            help='activation function to use'
        )
        parser.add_argument(
            '--dropout', 
            type=float, 
            metavar='FLOAT', 
            help='dropout rate'
        )

        # encoder
        parser.add_argument(
            '--encoder-embed-dim', 
            type=int, 
            metavar='INT',
            help='encoder embedding dimension'
        )
        parser.add_argument(
            '--encoder-ffn-embed-dim', 
            type=int, 
            metavar='INT',
            help='encoder dimension for FFN hidden layer'
        )
        parser.add_argument(
            '--encoder-layers', 
            type=int, 
            metavar='INT',
            help='number of encoder layers'
        )
        parser.add_argument(
            '--encoder-attention-heads', 
            type=int, 
            metavar='INT',
            help='number of encoder attention heads'
        )
        parser.add_argument(
            '--encoder-normalize-before', 
            action='store_true',
            help='whether to apply layernorm before each encoder block'
        )

        parser.add_argument(
            '--align-after-encoder',
            action='store_true',
            help='apply align loss after encoder'
        )

        # decoder
        parser.add_argument(
            '--decoder-embed-dim', 
            type=int, 
            metavar='INT',
            help='decoder embedding dimension'
        )
        parser.add_argument(
            '--decoder-ffn-dim', 
            type=int, 
            metavar='INT',
            help='decoder dimension for FFN hidden layer'
        )
        parser.add_argument(
            '--decoder-layers', 
            type=int, 
            metavar='INT',
            help='number of decoder layers'
        )
        parser.add_argument(
            '--decoder-attention-heads', 
            type=int, 
            metavar='INT',
            help='number of decoder attention heads'
        )
        parser.add_argument(
            '--decoder-normalize-before', 
            action='store_true',
            help='whether to apply layernorm before each decoder block'
        )
        parser.add_argument(
            '--share-decoder-input-output-embed',
            action="store_true",
            help="share decoder input and output embeddings",
        )

    @classmethod
    def build_model(cls, args, task):
        # initialize base model arguments
        cs291k_model_base(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)
        
        encoder_embedding = None
        if task.source_dictionary is not None:
            encoder_embedding = build_embedding(task.source_dictionary, args.encoder_embed_dim)
        encoder = cls.build_encoder(args, encoder_embedding)

        decoder_embedding = build_embedding(task.target_dictionary, args.decoder_embed_dim)
        decoder = cls.build_decoder(args, task.target_dictionary, decoder_embedding)

        return cls(encoder, decoder)

    @classmethod
    def build_encoder(cls, args, encoder_embedding):
        encoder = CS291KEncoder(args, encoder_embedding)
        return encoder
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, decoder_embedding):
        decoder = TransformerDecoderScriptable(args, tgt_dict, decoder_embedding)
        return decoder

    def forward_with_internal(self, src_tokens, src_lengths, prev_output_tokens, \
        src_text_lengths=None, **extra_args):
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths, \
            src_text_lengths=src_text_lengths)
        decoder_out = self.decoder(prev_output_tokens=prev_output_tokens, \
            encoder_out=encoder_out)
        return decoder_out, encoder_out.internal_states

    def forward(self, src_tokens, src_lengths, prev_output_tokens, src_text_lengths=None, \
        **extra_args):
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths, \
            src_text_lengths=src_text_lengths)
        decoder_out = self.decoder(prev_output_tokens=prev_output_tokens, \
            encoder_out=encoder_out)
        return decoder_out


class TransformerDecoderScriptable(TransformerDecoder):
    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, _ = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        return x, None


@register_model_architecture('cs291k_model', 'cs291k_model_base')
def cs291k_model_base(args):
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = args.dropout
    args.activation_dropout = args.dropout
    args.relu_dropout = args.dropout

    # encoder
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    
    # decoder
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)

    args.decoder_input_dim = args.decoder_embed_dim
    args.decoder_output_dim = args.decoder_embed_dim
    args.decoder_learned_pos = False
    args.no_scale_embedding = False
    args.adaptive_input = False
    args.adaptive_softmax_cutoff = None
    args.adaptive_softmax_dropout = 0.
    args.no_token_positional_embeddings = False
    args.quant_noise_pq = 0
    args.decoder_layerdrop = 0.

    # other
    args.max_source_positions = getattr(args, 'max_source_positions', 1000000)