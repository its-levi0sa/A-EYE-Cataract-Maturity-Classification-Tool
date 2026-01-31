"""
Program Title: radial_tokenizer.py

Programmers:
  Albonia, Jade Lorenz M.
  Caspe, Mark Vincent G.
  Rivera, Rei Djemf M.
  Velante, Kamilah Kaye M.
  Villegas, Jedidiah S.

Where the program fits in the general system design:
  This file is located in the `src/` directory. It serves as the initial
  preprocessing module of the A-EYE architecture. It transforms raw pixel data
  into a sequence of radially-aggregated feature vectors, which are subsequently
  ingested by the `ModifiedMobileViT` backbone.

Date Written: July 2025
Date Revised: December 2025

Purpose:
  To implement the "Radial-Aware" tokenization strategy. This module mathematically
  partitions the pupil image into concentric zones (4, 8, or 16 rings) and
  extracts statistical descriptors (Mean, Standard Deviation, Median) for each
  zone. This converts a 2D spatial image into a 1D sequence of anatomical features.

Data Structures, Algorithms, and Control:
  Data Structures:
    ring_masks (Tensor): Pre-computed boolean masks stored as a persistent
      buffer. This prevents redundant Euclidean distance calculations during
      every forward pass.
    tokens (Tensor): The output tensor containing statistical features.
      Shape: [Batch, Num_Rings, 9]. (The 9 features are Mean/Std/Median for R, G, B).

  Algorithms:
    Euclidean Distance Transformation: Computes a pixel-wise distance grid from
      the image center to determine ring membership.
    Vectorized Statistical Pooling: Aggregates pixel values within each mask
      to compute Mean and Standard Deviation entirely on the GPU.
    CPU-Offloaded Deterministic Median: Computes the median on the CPU to
      bypass non-deterministic behavior in CUDA's sorting algorithms.

  Control:
    Input Validation: Enforces strict constraints on the `num_rings` argument
      (valid sets: 4, 8, 16) to align with the ablation study configurations.
    Device Management: Manages the transfer of tensors between GPU and CPU
      specifically for the median calculation step to balance speed and determinism.
"""

import torch
import torch.nn as nn

class RadialTokenizer(nn.Module):
    """
    Transforms an input image into a sequence of Radial Tokens.
    """
    def __init__(self, image_size=256, num_rings=16):
        """
        Args:
            image_size (int): Height/Width of the input image.
            num_rings (int): Number of concentric rings to divide the image into.
        """
        super().__init__()
        self.image_size = image_size
        self.center = (image_size // 2, image_size // 2)
        self.num_rings = num_rings

        if self.num_rings == 4: ring_width = 32
        elif self.num_rings == 8: ring_width = 16
        elif self.num_rings == 16: ring_width = 8
        else: raise ValueError("Unsupported number of rings. Must be 4, 8, or 16.")
            
        self.rings = [(i * ring_width, (i + 1) * ring_width) for i in range(self.num_rings)]

        y, x = torch.meshgrid(torch.arange(0, image_size), torch.arange(0, image_size), indexing='ij')
        distance_grid = torch.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
        
        ring_masks = [(distance_grid >= r_inner) & (distance_grid < r_outer) for r_inner, r_outer in self.rings]
        self.register_buffer('ring_masks', torch.stack(ring_masks, dim=0).float())

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_tensor (Tensor): Input images [Batch, 3, 256, 256]
            
        Returns:
            Tensor: Radial tokens [Batch, Num_Rings, 9]
                    (9 features = MeanRGB + StdRGB + MedianRGB)
        """
        B, C, H, W = image_tensor.shape
        original_device = image_tensor.device
        
        masks = self.ring_masks.to(original_device)
        masked_pixels = masks.unsqueeze(0).unsqueeze(2) * image_tensor.unsqueeze(1)
        num_pixels_per_ring = masks.sum(dim=[1, 2]) + 1e-6

        # Calculate statistics
        # --- 1. Mean Calculation (GPU) ---
        mean_vals = masked_pixels.sum(dim=[3, 4]) / num_pixels_per_ring.view(1, self.num_rings, 1)

        # --- 2. Standard Deviation Calculation (GPU) ---
        mean_sq_vals = (masked_pixels**2).sum(dim=[3, 4]) / num_pixels_per_ring.view(1, self.num_rings, 1)
        std_vals = torch.sqrt(torch.clamp(mean_sq_vals - mean_vals**2, min=0))
        
        # --- 3. Median Calculation (CPU - Deterministic) ---
        flat_pixels = masked_pixels.view(B, self.num_rings, C, -1)
        flat_pixels_cpu = flat_pixels.cpu()
        flat_pixels_cpu[flat_pixels_cpu == 0] = float('nan')
        
        median_vals_cpu = torch.nanmedian(flat_pixels_cpu, dim=3).values
        
        # Move results back to GPU
        median_vals = median_vals_cpu.to(original_device)

        # --- 4. Concatenate Features ---
        tokens = torch.cat([mean_vals, std_vals, median_vals], dim=2)
        return tokens.to(device=original_device, dtype=torch.float32)