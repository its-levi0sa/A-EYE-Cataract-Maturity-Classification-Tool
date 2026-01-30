"""
Program Title: utils.py

Programmers:
  Albonia, Jade Lorenz M.
  Caspe, Mark Vincent G.
  Rivera, Rei Djemf M.
  Velante, Kamilah Kaye M.
  Villegas, Jedidiah S.

Where the program fits in the general system design:
  This file is the toolbox of the project, located in `src/`. It holds all the
  helper functions and standard building blocks that get reused across different
  parts of the model (like `aeye_model.py`) and training scripts.

Date Written: July 2025
Date Revised: December 2025

Purpose:
  To keep the main code clean by offloading common tasks here. It defines the
  standard MobileNet blocks used for the CNN parts, custom Loss function,
  and tools for reproducibility (seeding).

Data Structures, Algorithms, and Control:
  Data Structures:
    nn.Sequential: Use this a lot here to bundle layers (like Conv+BN+ReLU)
      into single reusable blocks.

  Algorithms:
    Focal Loss: Custom loss calculation. It uses a mathematical formula to
      down-weight easy examples so the model focuses on the hard ones.
    Inverted Residuals (MV2Block): The standard MobileNetV2 algorithm where the
      channels are expanded, do a depthwise convolution, and then project back down.
    Token Folding: A simple logic to turn the 1D transformer tokens back into
      a 2D feature map by taking the average and expanding it across the grid.

  Control:
    Residual Connections: Inside `MV2Block`, logic that checks if the
      input and output shapes match. If they do, input is added back to the
      output (skip connection); otherwise, omitted.
    Reduction Logic: In `FocalLoss`, it checks the mean, sum,
      or raw loss and return the correct format.
"""

import os
import random
import torch
import torch.nn as nn
import numpy as np

def seed_everything(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in binary classification.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        p_t = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t)**self.gamma * bce_loss
        
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

def fold_tokens_to_grid(tokens, output_size):
    """Reconstructs a 2D feature map from a sequence of 1D tokens."""
    B, P, D = tokens.shape
    H, W = output_size
    global_context = tokens.mean(dim=1)
    return global_context.unsqueeze(-1).unsqueeze(-1).expand(B, D, H, W)

def conv_3x3_bn(inp, oup, stride=1):
    """3x3 convolution with BatchNorm and SiLU."""
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.SiLU())

class MV2Block(nn.Module):
    """MobileNetV2 inverted residual block."""
    def __init__(self, inp, oup, stride=1, expansion=4):
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