"""
Program Title: utils.py

Programmers:
  Albonia, Jade Lorenz M.
  Caspe, Mark Vincent G.
  Rivera, Rei Djemf M.
  Velante, Kamilah Kaye M.
  Villegas, Jedidiah S.

Where the program fits in the general system design:
  This file is located in the `src/` directory. It functions as the shared
  utility library for the A-EYE project, containing reusable architectural
  components, custom loss functions, and reproducibility protocols that are
  invoked by multiple modules in the system.

Date Written: July 2025
Date Revised: December 2025

Purpose:
  To modularize common operations for code maintainability. It implements:
  1. The Inverted Residual Blocks (MV2Block) for the CNN backbone.
  2. The Focal Loss function to address class imbalance during training.
  3. Tensor manipulation utilities for bridging the Transformer and CNN domains.

Data Structures, Algorithms, and Control:
  Data Structures:
    nn.Sequential: Used to encapsulate sequences of Convolution, Normalization,
      and Activation layers into atomic, reusable blocks.

  Algorithms:
    Focal Loss: A dynamically scaled Cross-Entropy loss that down-weights the
      contribution of easy examples (well-classified inputs) to focus model
      training on hard, misclassified examples.
    Inverted Residuals: The core mechanism of MobileNetV2 that expands low-
      dimensional inputs to high dimensions for filtering, then projects them
      back down, preserving information via residual skip connections.
    Global Context Aggregation (Token Folding): A technique to condense the
      radial token sequence into a global context vector and broadcast it
      spatially to match the dimensions of the convolutional feature maps.

  Control:
    Residual Branching: The `MV2Block` implements conditional execution flow;
      if the input and output dimensions align, it adds the residual connection,
      otherwise, it performs a standard forward pass.
    Reduction Logic: The `FocalLoss` class implements conditional return statements
      to handle different loss aggregation strategies required by different 
      training phases.
"""

import os
import random
import torch
import torch.nn as nn
import numpy as np

def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary classification tasks.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): Weighting factor for the rare class.
            gamma (float): Focusing parameter to down-weight easy examples.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Predicted logits.
            targets (Tensor): Ground truth labels (0 or 1).
        """
        bce_loss = self.bce_loss(inputs, targets)
        p_t = torch.exp(-bce_loss)

        # Calculate alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate Focal Loss
        focal_loss = alpha_t * (1 - p_t)**self.gamma * bce_loss
        
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

def fold_tokens_to_grid(tokens, output_size):
    """
    Aggregates radial tokens into a global context vector and broadcasts it
    to a 2D spatial grid.

    Args:
        tokens (Tensor): Input tokens [Batch, Num_Rings, Dim]
        output_size (tuple): Target spatial dimensions (Height, Width)

    Returns:
        Tensor: Broadcasted feature map [Batch, Dim, Height, Width]
    """
    B, P, D = tokens.shape
    H, W = output_size
    global_context = tokens.mean(dim=1)
    return global_context.unsqueeze(-1).unsqueeze(-1).expand(B, D, H, W)

def conv_3x3_bn(inp, oup, stride=1):
    """
    Helper to create a standard 3x3 Convolution-BatchNorm-SiLU block.
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class MV2Block(nn.Module):
    """
    MobileNetV2 Inverted Residual Block.
    """
    def __init__(self, inp, oup, stride=1, expansion=4):
        """
        Args:
            inp (int): Input channels.
            oup (int): Output channels.
            stride (int): Stride for the depthwise convolution.
            expansion (int): Expansion factor for the hidden dimension.
        """
        super().__init__()
        hidden_dim = int(inp * expansion)
        self.use_res_connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU(),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)