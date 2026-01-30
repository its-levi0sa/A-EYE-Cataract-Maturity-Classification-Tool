"""
Program Title: transformer_block.py

Programmers:
  Albonia, Jade Lorenz M.
  Caspe, Mark Vincent G.
  Rivera, Rei Djemf M.
  Velante, Kamilah Kaye M.
  Villegas, Jedidiah S.

Where the program fits in the general system design:
  This file is located in `src/`. It provides the standard building blocks for
  the transformer part of A-EYE. It is imported by `modified_mobilevit.py`
  to actually process the radial tokens we generate.

Date Written: July 2025
Date Revised: December 2025

Purpose:
  To implement a standard, stable Transformer Encoder stack. It defines the
  Attention mechanism (so rings can "talk" to each other) and the FeedForward
  networks, wrapped in Pre-Normalization for better training stability.

Data Structures, Algorithms, and Control:
  Data Structures:
    nn.ModuleList: Used to stack multiple layers of attention and feedforward
      blocks based on the defined `depth`.

  Algorithms:
    Multi-Head Self-Attention: The core logic that calculates relationships
      between different tokens (rings). It uses scaled dot-product attention.
    Pre-Norm Residual Connection: Normalize the data before passing it
      into the layers and add the original input back to the output (skip
      connection) to prevent vanishing gradients.
    FeedForward Network: A simple expansion-contraction MLP using SiLU
      activation to process features after attention.

  Control:
    Iterative Layering: The `TransformerBlock` loops through the list of layers
      defined by the `depth` parameter, applying Attention and FeedForward
      sequentially with residual adds.
"""

import torch
import torch.nn as nn

class PreNorm(nn.Module):
    """LayerNorm followed by a function (Pre-Norm Transformer)."""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """Standard Transformer Feed-Forward Network."""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Multi-Head Self-Attention (fixed version)."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, n, self.heads, -1).transpose(1, 2) for t in qkv]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)

        return self.to_out(out)


class TransformerBlock(nn.Module):
    """
    A stack of Transformer Encoder blocks using Pre-Norm for stability.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        """Forward pass for the TransformerBlock."""
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x