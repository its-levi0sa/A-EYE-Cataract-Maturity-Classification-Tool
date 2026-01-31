"""
Program Title: radial_positional_encoding.py

Programmers:
  Albonia, Jade Lorenz M.
  Caspe, Mark Vincent G.
  Rivera, Rei Djemf M.
  Velante, Kamilah Kaye M.
  Villegas, Jedidiah S.

Where the program fits in the general system design:
  This file is located in the `src/` directory. It is a critical sub-component
  of the `ModifiedMobileViT` backbone. It is invoked immediately before the
  Transformer attention mechanism to inject spatial awareness into the sequence.

Date Written: July 2025
Date Revised: December 2025

Purpose:
  To address the permutation-invariance of standard Transformers. Since the
  self-attention mechanism treats all tokens as an unordered set, this module
  injects a learnable "Radial Bias." This ensures the model distinguishes
  between the center (Ring 0) and the periphery (Ring N), preserving the
  anatomical hierarchy of the lens.

Data Structures, Algorithms, and Control:
  Data Structures:
    nn.Embedding: A lookup table that stores a unique, learnable vector for
      each ring index.

  Algorithms:
    Learnable Additive Embedding: Unlike fixed sinusoidal encodings (used in
      NLP), this implementation allows the model to learn the optimal
      positional representations via backpropagation during training.
    Vector Addition: The position vector is added element-wise to the feature
      token, mathematically embedding location data into the feature space.

  Control:
    Dimensionality Verification: Includes assertions to enforce that the
      input tensor dimensions strictly match the configuration defined in
      `train.py`.
"""

import torch
import torch.nn as nn

class RadialPositionEmbedding(nn.Module):
    """
    Learnable Positional Encoding for Radial Tokens.
    """
    def __init__(self, num_rings, embed_dim):
        """
        Args:
            num_rings (int): The total number of concentric rings (sequence length).
            embed_dim (int): The dimension of the feature vector for each ring.
        """
        super().__init__()
        self.num_rings = num_rings
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_embeddings=self.num_rings, embedding_dim=self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tokens of shape [Batch, Num_Rings, Embed_Dim]
            
        Returns:
            Tensor: Output tokens with positional info added [Batch, Num_Rings, Embed_Dim]
        """
        B, num_tokens, dim = x.shape
        assert num_tokens == self.num_rings, f"Input tensor has {num_tokens} tokens, but model expects {self.num_rings}."
        assert dim == self.embed_dim, f"Input tensor has embedding dim {dim}, but model expects {self.embed_dim}."

        indices = torch.arange(self.num_rings, device=x.device).unsqueeze(0).expand(B, -1)
        pos_embed = self.embedding(indices)
        
        return x + pos_embed