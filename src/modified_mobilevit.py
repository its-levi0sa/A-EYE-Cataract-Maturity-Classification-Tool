"""
Program Title: modified_mobilevit.py

Programmers:
  Albonia, Jade Lorenz M.
  Caspe, Mark Vincent G.
  Rivera, Rei Djemf M.
  Velante, Kamilah Kaye M.
  Villegas, Jedidiah S.

Where the program fits in the general system design:
  This file is located in the `src/` directory. It defines the core "Radial-Aware"
  processing block of the A-EYE architecture. It replaces the standard
  MobileViT block found in the baseline, serving as the fusion point where
  radial statistics (from the tokenizer) are integrated with spatial CNN features.

Date Written: July 2025
Date Revised: December 2025

Purpose:
  To implement a Cross-Modal Fusion mechanism. It enables the model to
  simultaneously process local spatial details (via the residual input) and
  global radial context (via the token input), allowing the network to learn
  relationships between lens opacity density and spatial texture.

Data Structures, Algorithms, and Control:
  Data Structures:
    tokens (Tensor): A tensor representing the statistical features of the
      concentric rings (Shape: Batch x Num_Rings x Feature_Dim).
    x (Tensor): The spatial feature map from the preceding convolutional layers.

  Algorithms:
    Radial Positional Encoding: Injects geometric context into the tokens,
      allowing the Transformer to distinguish between core and peripheral rings.
    Feature Fusion: A reconstruction technique that projects 1D ring tokens
      back into a 2D spatial grid (`fold_tokens_to_grid`) to concatenate
      them with the image feature map.

  Control:
    Pipeline Flow: Token Projection -> Positional Encoding -> Transformer
      Attention -> Spatial Reconstruction -> Concatenation -> Convolutional Fusion.
"""

import torch
import torch.nn as nn
from .transformer_block import TransformerBlock
from .radial_positional_encoding import RadialPositionEmbedding
from .utils import fold_tokens_to_grid

class ModifiedMobileViT(nn.Module):
    """
    A modified MobileViT block that integrates Radial Token injection.
    """
    def __init__(self, in_channels, embed_dim, depth, num_rings, num_heads=4, mlp_dim=384):
        """
        Args:
            in_channels (int): Number of channels in the input feature map.
            embed_dim (int): Embedding dimension for the Transformer.
            depth (int): Number of Transformer layers.
            num_rings (int): Number of concentric rings (defines sequence length).
            num_heads (int): Number of attention heads.
            mlp_dim (int): Hidden dimension of the Feed-Forward Network.
        """
        super().__init__()
        
        # --- Global (Radial Token) Path ---
        self.token_proj = nn.Linear(9, embed_dim) 
        self.pos_encoder = RadialPositionEmbedding(num_rings=num_rings, embed_dim=embed_dim)
        
        # Transformer backbone for processing radial relationships
        self.transformer = TransformerBlock(
            dim=embed_dim, 
            depth=depth, 
            heads=num_heads, 
            dim_head=embed_dim // num_heads,
            mlp_dim=mlp_dim
        )

        # --- Fusion Path ---
        # Projects the transformed tokens back to the channel dimension
        self.proj_out = nn.Conv2d(embed_dim, in_channels, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

    def forward(self, x, tokens):
        """
        Args:
            x (Tensor): Input feature map [Batch, Channels, Height, Width]
            tokens (Tensor): Radial statistics [Batch, Num_Rings, 9]
        """
        res = x
        
        # 1. Process Global Radial Tokens
        tokens_proj = self.token_proj(tokens)
        tokens_encoded = self.pos_encoder(tokens_proj)
        tokens_transformed = self.transformer(tokens_encoded)

        # 2. Reconstruct Feature Map from Tokens
        x_global = fold_tokens_to_grid(tokens_transformed, output_size=x.shape[2:])
        x_global = self.proj_out(x_global)
        
        # 3. Fuse Residual Input with Global Context
        x_fused = torch.cat([res, x_global], dim=1)
        x = self.fuse(x_fused)
        
        return x