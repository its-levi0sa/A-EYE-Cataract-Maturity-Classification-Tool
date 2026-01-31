"""
Program Title: baseline_model.py

Programmers:
  Albonia, Jade Lorenz M.
  Caspe, Mark Vincent G.
  Rivera, Rei Djemf M.
  Velante, Kamilah Kaye M.
  Villegas, Jedidiah S.

Where the program fits in the general system design:
  This file is located in the `src/` directory and implements the standard
  MobileViT architecture. It serves as the "Baseline" (Control) model for
  comparative analysis against the custom A-EYE architecture.

Date Written: July 2025
Date Revised: December 2025

Purpose:
  To provide a rigorous implementation of the original MobileViT network
  without any radial modifications. This facilitates fair side-by-side
  comparisons (SOP 2) to statistically validate the performance improvements
  introduced by the A-EYE system.

Data Structures, Algorithms, and Control:
  Data Structures:
    nn.ModuleList: Manages the sequential organization of MobileNetV2 (MV2)
      inverted residual blocks and MobileViT transformer blocks.
  
  Algorithms:
    Inverted Residuals: Efficient convolutional blocks that expand and
      squeeze channel dimensions to preserve information with fewer parameters.
    MobileViT Block: A hybrid processing unit that unfolds image tensors into
      non-overlapping patches, applies a global Transformer, and folds them
      back to coordinate space.
  
  Control:
    Layer Sequencing: The `forward` method explicitly orchestrates the tensor
      flow through specific indices of the module lists (e.g., `mv2[0]`,
      `mvit[0]`) to replicate the exact downsampling architecture of the
      original MobileViT paper.
"""

import torch
import torch.nn as nn
from einops import rearrange

# --- Helper Modules ---

def conv_1x1_bn(inp, oup):
    """
    Creates a 1x1 Convolution followed by BatchNorm and SiLU activation.
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    """
    Creates an NxN Convolution followed by BatchNorm and SiLU activation.
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class PreNorm(nn.Module):
    """
    Applies Layer Normalization before passing the input to a function/layer.
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm, self.fn = nn.LayerNorm(dim), fn

    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    """
    Standard Transformer Feed-Forward Network (FFN) with expansion and dropout.
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
    Multi-Head Self-Attention (MHSA) mechanism.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads, self.scale = heads, dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    """
    A Transformer Encoder composed of stacked Attention and FeedForward layers.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MV2Block(nn.Module):
    """
    MobileNetV2 Inverted Residual Block.
    Uses depthwise separable convolutions to reduce computation.
    """
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU(),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
        ) if expansion != 1 else nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU(),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileViTBlock(nn.Module):
    """
    The core MobileViT block.
    Learn local representations with CNN and global representations with Transformers.
    """
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        
        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)
        
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    
    def forward(self, x):
        y = x.clone()

        # Local Representation
        x = self.conv1(x)
        x = self.conv2(x)

        # Global Representation (Unfold -> Transformer -> Fold)
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

# --- Main MobileViT Model ---

class MobileViT(nn.Module):
    """
    The complete MobileViT architecture (S-Variant).
    """
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0
        
        L = [2, 4, 3] # Transformer depths

        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        self.mv2 = nn.ModuleList([
            MV2Block(channels[0], channels[1], 1, expansion),
            MV2Block(channels[1], channels[2], 2, expansion),
            MV2Block(channels[2], channels[3], 1, expansion),
            MV2Block(channels[3], channels[4], 1, expansion),
            MV2Block(channels[4], channels[5], 2, expansion),
            MV2Block(channels[5], channels[6], 2, expansion),
            MV2Block(channels[7], channels[8], 2, expansion),
        ])

        self.mvit = nn.ModuleList([
            MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)),
            MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4)),
            MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2]*4)),
        ])

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])
        self.pool = nn.AvgPool2d(ih//32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)
        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)
        x = self.mv2[4](x)
        
        x = self.mvit[0](x)
        
        x = self.mv2[5](x)
        x = self.mvit[1](x)
        
        x = self.mv2[6](x)
        x = self.mvit[2](x)
        
        x = self.conv2(x)
        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x

# --- Instantiation Function ---

def mobilevit_s():
    """
    Returns the MobileViT-Small (S) configuration used as the Baseline.
    """
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT((256, 256), dims, channels, num_classes=1000)