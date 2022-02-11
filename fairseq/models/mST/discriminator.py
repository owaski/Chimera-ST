from argparse import Namespace

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models.transformer import TransformerEncoderLayer

class Classifier(nn.Module):
    def __init__(self, nlayer, ndim, nhid, nhead, nclass, drop=0.1) -> None:
        super().__init__()
        
        args = {
            'encoder_embed_dim': ndim,
            'encoder_attention_heads': nhead,
            'attention_dropout': drop,
            'dropout': drop,
            'activation_dropout': drop,
            'encoder_normalize_before': True,
            'encoder_ffn_embed_dim': nhid, 
        }
        args = Namespace(**args)

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(nlayer)]
        )
        self.linear = nn.Linear(ndim, nclass)

    def forward(self, x, padding_mask):
        for layer in self.layers:
            x = layer(x, padding_mask)
        logits = self.linear(x)[0]
        return logits