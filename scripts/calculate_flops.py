"""
Program Title: calculate_flops.py

Programmers:
  Albonia, Jade Lorenz M.
  Caspe, Mark Vincent G.
  Rivera, Rei Djemf M.
  Velante, Kamilah Kaye M.
  Villegas, Jedidiah S.

Where the program fits in the general system design:
  This utility script is located in the `scripts/` directory. It is executed
  independently of the training loop to generate the "Computational Efficiency"
  metrics reported in the thesis results.

Date Written: July 2025
Date Revised: December 2025

Purpose:
  To quantify the computational complexity of the A-EYE architecture variants
  versus the Baseline. It calculates two critical resource metrics:
  1. Parameters (M): Proxy for memory footprint and storage size.
  2. GFLOPs (Giga Floating Point Operations): Proxy for inference latency and
     power consumption on mobile devices.

Data Structures, Algorithms, and Control:
  Data Structures:
    dummy_input (Tensor): A synthetic tensor of shape (1, 3, 256, 256) used
      to trigger a forward pass for tracing.

  Algorithms:
    THOP Profiling: Utilizes the `thop` (PyTorch OpCounter) library to hook
      into the model's computational graph and count exact Multiply-Accumulate
      (MAC) operations.
    FLOPs Conversion: Converts MACs to GFLOPs using the standard hardware
      formula.

  Control:
    Model Instantiation: dynamically initializes the specific model architecture
      (Baseline or A-EYE 4/8/16-Ring) based on command-line arguments to ensure
      accurate benchmarking.
"""

import argparse
import torch
import torch.nn as nn
from thop import profile

from src.aeye_model import AEyeModel
from src.baseline_model import mobilevit_s

def main(args):
    """
    Main execution function to measure and report model complexity.
    """
    # --- Model Selection & Instantiation ---
    if args.model_type == 'aeye':
        # Hardcoded dims/embed_dim match the fixed architecture used in the study
        model = AEyeModel({'dims': [32, 64, 128, 160], 'embed_dim': 256, 'num_rings': args.num_rings})
        model_name = f"A-EYE ({args.num_rings} rings)"
    else:
        # Baseline MobileViT-S
        model = mobilevit_s()
        model.fc = nn.Linear(model.fc.in_features, 1)
        model_name = "Baseline MobileViT-S"

    # --- Computational Profiling ---
    # Create a dummy input tensor matching the standard training resolution (256x256)
    input_size = (1, 3, 256, 256)
    dummy_input = torch.randn(*input_size)

    # Calculate FLOPs and Parameters using thop
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)

    # Convert to GFLOPs (Giga FLOPs) and Millions of parameters
    gflops = (macs * 2) / 1e9
    params_m = params / 1e6

    # --- Report Generation ---
    print("\n" + "="*40)
    print(f"Model Efficiency Report")
    print("="*40)
    print(f"Model:           {model_name}")
    print(f"Input Size:      {input_size}")
    print(f"Parameters:      {params_m:.2f} M")
    print(f"GFLOPs:          {gflops:.2f} G")
    print("="*40 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate FLOPs and Parameters for a model")
    parser.add_argument('--model_type', type=str, required=True, choices=['aeye', 'baseline'])
    parser.add_argument('--num_rings', type=int, choices=[4, 8, 16], help="Required for 'aeye' model.")
    args = parser.parse_args()

    if args.model_type == 'aeye' and args.num_rings is None:
        parser.error("--num_rings is required when --model_type is 'aeye'")
    
    main(args)
