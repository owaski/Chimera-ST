from argparse import Namespace

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from fairseq.models.transformer import TransformerEncoderLayer

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)

    Implementation comes from https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(th.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class Classifier(nn.Module):
    def __init__(self, nlayer, ndim, nhid, nhead, nclass, drop=0.1, rev=1.0) -> None:
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

        self.grad_rev = GradientReversal(rev)

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(nlayer)]
        )
        self.linear = nn.Linear(ndim, nclass)

    def forward(self, x, padding_mask):
        x = self.grad_rev(x)
        for layer in self.layers:
            x = layer(x, padding_mask)
        logits = self.linear(x)[0]
        return logits