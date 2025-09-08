import argparse
import torch
import torch.nn as nn
from thop import profile

from src.aeye_model import AEyeModel
from src.baseline_model import mobilevit_s

def main(args):
    """Main function to instantiate the model and calculate its complexity."""
    # --- Model Selection ---
    if args.model_type == 'aeye':
        model = AEyeModel({'dims': [32, 64, 128, 160], 'embed_dim': 256, 'num_rings': args.num_rings})
        model_name = f"A-EYE ({args.num_rings} rings)"
    else: # baseline
        model = mobilevit_s()
        model.fc = nn.Linear(model.fc.in_features, 1)
        model_name = "Baseline MobileViT-S"

    # Create a dummy input tensor with the standard size used for training
    input_size = (1, 3, 256, 256)
    dummy_input = torch.randn(*input_size)

    # Calculate FLOPs and Parameters using thop
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)

    # Convert to GFLOPs (Giga FLOPs) and Millions of parameters
    gflops = (macs * 2) / 1e9
    params_m = params / 1e6

    # --- Print Results ---
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
