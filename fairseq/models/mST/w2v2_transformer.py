import logging
import os

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq import checkpoint_utils, utils
from fairseq.models import (
    register_model, register_model_architecture,
    FairseqEncoderDecoderModel
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.models.bart import BARTModel
from fairseq.modules.layer_norm import LayerNorm

from .w2v2_transformer_encoder import W2V2TransformerEncoder
from .discriminator import Classifier

logger = logging.getLogger(__name__)

@register_model("w2v2_transformer")
class W2V2Transformer(FairseqEncoderDecoderModel):
    '''Transformer model for ST tasks'''

    def __init__(self, encoder, decoder, discriminator):
        super().__init__(encoder, decoder)
        self.discriminator = discriminator

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--w2v2-model-path', 
            type=str, 
            metavar='STR',
            help='path to wav2vec2 model'
        )
        parser.add_argument(
            '--w2v2-grad-mult',
            type=float,
            default=1.0,
            help='gradient multiplier of w2v2'
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

        # mBART50 dir
        parser.add_argument(
            '--mbart50-dir', 
            type=str, 
            metavar='STR',
            help='directory to mbart50 model'
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
            '--cnn-subsampler',
            action='store_true',
            help='use cnn subsampler to shrink the length'
        )
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            metavar="N",
            help="kernel sizes of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv1d subsampling layers",
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

        # hyper-parameters for discriminators
        parser.add_argument(
            '--disc-nlayer',
            type=int,
        )
        parser.add_argument(
            '--disc-ndim',
            type=int,
        )
        parser.add_argument(
            '--disc-nhid',
            type=int
        )
        parser.add_argument(
            '--disc-nhead',
            type=int
        )
        parser.add_argument(
            '--disc-nclass',
            type=int
        )
        parser.add_argument(
            '--disc-drop',
            type=float
        )

    @classmethod
    def build_model(cls, args, task):
        # initialize base model arguments
        w2v2_transformer_base(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)
        
        encoder_embedding = None
        task.src_dict = getattr(task, "src_dict", task.tgt_dict) # in case of s2t gen without src dict
        encoder_embedding = build_embedding(task.src_dict, args.encoder_embed_dim)
        encoder = cls.build_encoder(args, task.src_dict, encoder_embedding)

        decoder_embedding = build_embedding(task.tgt_dict, args.decoder_embed_dim)
        decoder = cls.build_decoder(args, task.tgt_dict, decoder_embedding)

        mbart50_params = checkpoint_utils.load_checkpoint_to_cpu(os.path.join(args.mbart50_dir, 'model.pt'))['model']
        mbart50_encoder_params = {k[8:]: v for k, v in mbart50_params.items() if k.startswith('encoder.')}
        mbart50_decoder_params = {k[8:]: v for k, v in mbart50_params.items() if k.startswith('decoder.')}
        encoder.transformer_encoder.load_state_dict(mbart50_encoder_params)
        decoder.load_state_dict(mbart50_decoder_params, strict=False)

        # discriminator
        discriminator = cls.build_discriminator(args)

        return cls(encoder, decoder, discriminator)

    @classmethod
    def build_encoder(cls, args, src_dict, encoder_embedding):
        encoder = W2V2TransformerEncoder(args, src_dict, encoder_embedding)
        return encoder
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, decoder_embedding):
        decoder = TransformerDecoder(args, tgt_dict, decoder_embedding)
        return decoder

    @classmethod
    def build_discriminator(cls, args):
        discriminator = Classifier(
            nlayer=args.disc_nlayer,
            ndim=args.disc_ndim,
            nhid=args.disc_nhid,
            nhead=args.disc_nhead,
            nclass=args.disc_nclass,
            drop=args.disc_drop,
        )
        return discriminator

    def forward_with_internal(self, src_tokens, src_lengths, prev_output_tokens, **extra_args):
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(prev_output_tokens=prev_output_tokens, encoder_out=encoder_out)
        return decoder_out, encoder_out

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **extra_args):
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(prev_output_tokens=prev_output_tokens, encoder_out=encoder_out)
        return decoder_out


@register_model_architecture('w2v2_transformer', 'xlsr_mbart50_base')
def w2v2_transformer_base(args):
    # args.activation_fn = getattr(args, 'activation_fn', 'relu')
    # args.dropout = getattr(args, 'dropout', 0.1)
    # args.attention_dropout = args.dropout
    # args.activation_dropout = args.dropout
    # args.relu_dropout = args.dropout

    # embedding
    # args.no_scale_embedding = getattr(args, "no_scale_embedding", False)

    # # encoder
    # args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    # args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    # args.encoder_layers = getattr(args, 'encoder_layers', 6)
    # args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    # args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    
    # # decoder
    # args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    # args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    # args.decoder_layers = getattr(args, 'decoder_layers', 6)
    # args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    # args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    # args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)

    # args.decoder_input_dim = args.decoder_embed_dim
    # args.decoder_output_dim = args.decoder_embed_dim
    # args.decoder_learned_pos = False
    # args.no_scale_embedding = False
    # args.adaptive_input = False
    # args.adaptive_softmax_cutoff = None
    # args.adaptive_softmax_dropout = 0.
    # args.no_token_positional_embeddings = False
    # args.quant_noise_pq = 0
    # args.decoder_layerdrop = 0.

    # Convolutional subsampler
    args.cnn_subsampler = getattr(args, 'cnn_subsampler', True)
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)

    # mBART-large config
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)

    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", True)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    args.decoder_output_dim = getattr(args, "decoder_output_dim", args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)

    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0.0)

    # Discriminator
    args.disc_nlayer = getattr(args, "disc_nlayer", 2)
    args.disc_ndim = getattr(args, "disc_ndim", args.encoder_embed_dim)
    args.disc_nhid = getattr(args, "disc_nhid", args.encoder_ffn_embed_dim)
    args.disc_nhead = getattr(args, "disc_nhead", 4)
    args.disc_nclass = getattr(args, "disc_nclass", 19)
    args.disc_drop = getattr(args, "disc_drop", args.dropout)