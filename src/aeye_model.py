"""
Program Title: aeye_model.py

Programmers:
  Albonia, Jade Lorenz M.
  Caspe, Mark Vincent G.
  Rivera, Rei Djemf M.
  Velante, Kamilah Kaye M.
  Villegas, Jedidiah S.

Where the program fits in the general system design:
  This file is located in the `src/` directory and serves as the central
  architectural definition of the A-EYE system. It integrates the
  `RadialTokenizer` (for radial feature extraction) and the `ModifiedMobileViT`
  (for spatial-semantic processing). This class is instantiated by external
  scripts to perform both model training and inference.

Date Written: July 2025
Date Revised: December 2025

Purpose:
  To define the hierarchical structure of the A-EYE Convolutional-Transformer
  hybrid network. It manages the tensor flow through initial convolutional
  stages, radial-aware transformer blocks, and the final classification head.

Data Structures, Algorithms, and Control:
  Data Structures:
    config (dict): A configuration dictionary containing hyperparameters
      such as model dimensions, embedding size, and ring count.
    nn.ModuleList / nn.Sequential: PyTorch containers used to organize
      the network layers sequentially.

  Algorithms:
    Hybrid Feature Extraction: Combines standard CNN blocks (MV2Block) with
      custom radial-aware transformer blocks.
    Adaptive Pooling: Reduces variable spatial dimensions to a fixed
      feature vector for the final linear classification layer.

  Control:
    Forward Pass: The `forward` method directs the data flow. It passes the
      input image to the tokenizer to generate radial tokens, which are then
      injected into specific network stages (Stages 3, 5, and 7) for fusion.
"""

import torch.nn as nn
from .utils import conv_3x3_bn, MV2Block
from .modified_mobilevit import ModifiedMobileViT
from .radial_tokenizer import RadialTokenizer

class AEyeModel(nn.Module):
    """
    The main A-EYE architecture combining Radial Tokenization with a 
    Modified MobileViT backbone for cataract maturity classification.
    """
    def __init__(self, config):
        """
        Initializes the model architecture based on the provided configuration.
        
        Args:
            config (dict): Must contain 'dims' (list of ints), 'embed_dim' (int), 
                           and 'num_rings' (int).
        """
        super().__init__()
        dims = config['dims']
        embed_dim = config['embed_dim']
        num_rings = config['num_rings']
        
        # Transformer block depths for Stages 3, 5, and 7
        transformer_depths = [2, 4, 3]

        self.tokenizer = RadialTokenizer(num_rings=num_rings)
        self.num_rings = self.tokenizer.num_rings

        # --- Model Architecture Stages ---
        # Stage 1: Initial Feature Extraction (Downsample 2x)
        self.stage1 = conv_3x3_bn(3, dims[0], stride=2)
        
        # Stage 2: MV2 Block (Downsample 2x)
        self.stage2 = MV2Block(dims[0], dims[1], stride=2)
        
        # Stage 3: Radial-Aware Transformer Block
        self.stage3 = ModifiedMobileViT(
            in_channels=dims[1], embed_dim=embed_dim, depth=transformer_depths[0], num_rings=self.num_rings
        )
        
        # Stage 4: MV2 Block (Downsample 2x)
        self.stage4 = MV2Block(dims[1], dims[2], stride=2)
        
        # Stage 5: Radial-Aware Transformer Block
        self.stage5 = ModifiedMobileViT(
            in_channels=dims[2], embed_dim=embed_dim, depth=transformer_depths[1], num_rings=self.num_rings
        )
        
        # Stage 6: MV2 Block (Downsample 2x)
        self.stage6 = MV2Block(dims[2], dims[3], stride=2)
        
        # Stage 7: Radial-Aware Transformer Block
        self.stage7 = ModifiedMobileViT(
            in_channels=dims[3], embed_dim=embed_dim, depth=transformer_depths[2], num_rings=self.num_rings
        )
        
        # --- Classification Head ---
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(dims[3], dims[3] // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dims[3] // 2, 1)
        )

    def forward(self, x_img, return_tokens=False):
        """
        Defines the computation performed at every call.

        Args:
            x_img (Tensor): Input image tensor of shape (B, C, H, W).
            return_tokens (bool): If True, returns the radial tokens for explainability.

        Returns:
            Tensor: The classification logit (or tuple if return_tokens is True).
        """
        # 1. Extract Radial Tokens (Global Context)
        tokens = self.tokenizer(x_img)

        # 2. Backbone Processing
        x = self.stage1(x_img)
        x = self.stage2(x)
        x = self.stage3(x, tokens)
        x = self.stage4(x)
        x = self.stage5(x, tokens)
        x = self.stage6(x)
        x = self.stage7(x, tokens)

        # 3. Classification
        x = self.pool(x).view(x.size(0), -1)
        output = self.fc(x)

        if return_tokens:
            return output, tokens
        return output