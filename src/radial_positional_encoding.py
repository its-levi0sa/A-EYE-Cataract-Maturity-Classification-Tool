"""
Program Title: radial_positional_encoding.py

Programmers:
  Albonia, Jade Lorenz M.
  Caspe, Mark Vincent G.
  Rivera, Rei Djemf M.
  Velante, Kamilah Kaye M.
  Villegas, Jedidiah S.

Where the program fits in the general system design:
  This file is in `src/`. This is a critical component used by the
  `ModifiedMobileViT` backbone. It sits right before the Transformer block to
  give the rings a sense of order.

Date Written: July 2025
Date Revised: December 2025

Purpose:
  To inject "positional" information into the ring tokens. Since Transformers
  treat all tokens as equal/unordered by default, this module adds a learnable
  vector to each token so the model knows which ring is the center (Ring 0)
  and which is the edge (Ring N).

Data Structures, Algorithms, and Control:
  Data Structures:
    nn.Embedding: A lookup table that stores a unique vector for each ring index.

  Algorithms:
    Learnable Additive Embedding: Instead of using fixed sine/cosine waves, the
      model is able learn the best position vectors during training. This simply
      generate indices [0, 1, ... N] and add the corresponding embedding vector
      to the input data.

  Control:
    Validation: Assertions are added to make sure the number of tokens coming in
      actually matches the number of rings defined in the config.
"""

import torch
import torch.nn as nn

class RadialPositionEmbedding(nn.Module):
    def __init__(self, num_rings, embed_dim):
        super().__init__()
        self.num_rings = num_rings
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_embeddings=self.num_rings, embedding_dim=self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, num_tokens, dim = x.shape
        assert num_tokens == self.num_rings, f"Input tensor has {num_tokens} tokens, but model expects {self.num_rings}."
        assert dim == self.embed_dim, f"Input tensor has embedding dim {dim}, but model expects {self.embed_dim}."

        indices = torch.arange(self.num_rings, device=x.device).unsqueeze(0).expand(B, -1)
        pos_embed = self.embedding(indices)
        
        return x + pos_embed