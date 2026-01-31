"""
Program Title: transformer_block.py

Programmers:
  Albonia, Jade Lorenz M.
  Caspe, Mark Vincent G.
  Rivera, Rei Djemf M.
  Velante, Kamilah Kaye M.
  Villegas, Jedidiah S.

Where the program fits in the general system design:
  This file is located in the `src/` directory. It serves as the modular
  implementation of the Transformer Encoder architecture. It is imported by
  `modified_mobilevit.py` to process the sequence of radial tokens extracted
  from the input image.

Date Written: July 2025
Date Revised: December 2025

Purpose:
  To implement a stable, Pre-Norm Transformer Encoder stack. It utilizes
  Multi-Head Self-Attention (MHSA) to model the dependencies between different
  anatomical zones (rings) and a Feed-Forward Network (FFN) to project these
  features into a higher-dimensional space for non-linear processing.

Data Structures, Algorithms, and Control:
  Data Structures:
    nn.ModuleList: A PyTorch container used to stack multiple iterations of
      Attention and Feed-Forward blocks based on the configured depth.

  Algorithms:
    Multi-Head Self-Attention (MHSA): The core mechanism that calculates the
      relational importance between tokens using Scaled Dot-Product Attention.
      
    Pre-Normalization: Applies LayerNorm before the sub-layers (Attention/FFN)
      rather than after. This structure (Pre-Norm) creates a direct path for
      gradients in the residual stream, significantly improving training stability.
    Feed-Forward Network (FFN): A position-wise MLP that expands and contracts
      the feature dimension using SiLU activation.

  Control:
    Iterative Layering: The `TransformerBlock` iterates through the defined
      `depth`, sequentially applying the Attention and Feed-Forward sub-layers
      with residual skip connections at each step.
"""

import torch
import torch.nn as nn

class PreNorm(nn.Module):
    """
    Applies Layer Normalization before passing the input to a function/layer.
    Used for the "Pre-Norm" Transformer variant which stabilizes training.
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    Standard Transformer Feed-Forward Network (FFN).
    Structure: Linear -> SiLU -> Dropout -> Linear -> Dropout
    """
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
    """
    Multi-Head Self-Attention (MHSA).
    Calculates relationships between all tokens in the sequence.
    """
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
        """
        Args:
            x (Tensor): Input tensor of shape (Batch, Num_Tokens, Dim)
        """
        b, n, _ = x.shape

        # 1. Linear Projection & Split into Heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, n, self.heads, -1).transpose(1, 2) for t in qkv]

        # 2. Scaled Dot-Product Attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        # 3. Aggregate Values
        out = torch.matmul(attn, v)

        # 4. Concatenate Heads and Project
        out = out.transpose(1, 2).reshape(b, n, -1)

        return self.to_out(out)


class TransformerBlock(nn.Module):
    """
    A wrapper class that stacks multiple Transformer Encoder layers.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        """
        Args:
            dim (int): Input feature dimension.
            depth (int): Number of stacked Transformer layers.
            heads (int): Number of attention heads.
            dim_head (int): Dimension of each attention head.
            mlp_dim (int): Hidden dimension of the FeedForward layer.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.layers = nn.ModuleList([])

        # Stack the layers defined by 'depth'
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        """
        Forward pass with Residual Connections.
        x = x + Attention(Norm(x))
        x = x + FeedForward(Norm(x))
        """
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x