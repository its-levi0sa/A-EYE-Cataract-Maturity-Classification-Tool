import torch
import torch.nn as nn
from .transformer_block import TransformerBlock
from .radial_positional_encoding import RadialPositionEmbedding
from .utils import fold_tokens_to_grid

class ModifiedMobileViT(nn.Module):
    def __init__(self, in_channels, embed_dim, depth, num_rings, num_heads=4, mlp_dim=384):
        super().__init__()
        
        # Local feature path
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        self.proj_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # Global (token) path
        self.token_proj = nn.Linear(9, embed_dim) 
        self.pos_encoder = RadialPositionEmbedding(num_rings=num_rings, embed_dim=embed_dim)
        
        # --- UPDATED TO BE DEEPER AND MATCH BASELINE ---
        self.transformer = TransformerBlock(
            dim=embed_dim, 
            depth=depth, 
            heads=num_heads, 
            dim_head=embed_dim // num_heads,
            mlp_dim=mlp_dim
        )

        # Fusion path
        self.proj_out = nn.Conv2d(embed_dim, in_channels, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

    def forward(self, x, tokens):
        res = x
        
        # Process global radial tokens first
        tokens_proj = self.token_proj(tokens)
        tokens_encoded = self.pos_encoder(tokens_proj)
        tokens_transformed = self.transformer(tokens_encoded)

        # Reconstruct a feature map from the processed tokens
        x_global = fold_tokens_to_grid(tokens_transformed, output_size=x.shape[2:])
        x_global = self.proj_out(x_global)
        
        # Fuse the residual input (original feature map) with the global feature map
        x_fused = torch.cat([res, x_global], dim=1)
        x = self.fuse(x_fused)
        
        return x