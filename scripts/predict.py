"""
Program Title: predict.py

Programmers:
  Albonia, Jade Lorenz M.
  Caspe, Mark Vincent G.
  Rivera, Rei Djemf M.
  Velante, Kamilah Kaye M.
  Villegas, Jedidiah S.

Where the program fits in the general system design:
  Located in `scripts/`. This is the user-facing script for testing the model
  on a single image. It simulates how the final mobile app would behave: taking
  an image as input and outputting a diagnosis ("Mature" or "Immature") along
  with an explainability report.

Date Written: July 2025
Date Revised: December 2025

Purpose:
  To demonstrate the A-EYE model's inference capabilities. It loads a trained
  model (or an ensemble of models), processes a new image, and provides both a
  classification result and a human-readable explanation based on the radial
  token statistics.

Data Structures, Algorithms, and Control:
  Data Structures:
    Ensemble List: A list of loaded PyTorch models. Predictions were averaged
      to get a more robust final result.
    Token Tensor: The internal radial features extracted from the model, which
      were analyze to generate the explanation.

  Algorithms:
    Model Ensembling: Instead of relying on just one training run, available checkpoints are loaded 
      in the folder and average their probabilities.
    Heuristic Explainability: The raw token values (which represent brightness and texture 
    in each ring) were taken and convert them into user-friendly
      percentages like "Opacity Extent" and "Density."

  Control:
    Argument Logic: Checks if the user wants to test the 'aeye' model or the
      'baseline'. If 'aeye', it enforces that the `num_rings` argument is provided.
    Feedback Loop: Prints a formatted report to the console so the user can
      see exactly why the model made its decision.
"""

import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import cv2

from src.aeye_model import AEyeModel
from src.baseline_model import mobilevit_s
from src.data_utils import get_transforms

def generate_aeye_explanation(tokens_tensor, num_rings):
    """
    Generates a user-friendly summary and a detailed statistical report from
    the A-EYE model's internal radial tokens.
    """
    # Average the tokens from the model ensemble
    avg_tokens = tokens_tensor.mean(dim=0).squeeze(0).cpu().numpy()
    
    # Denormalize tokens back to an approximate [0, 255] pixel scale for interpretation
    mean_rgb = (avg_tokens[:, 0:3] * 0.5 + 0.5) * 255
    std_rgb = (avg_tokens[:, 3:6] * 0.5) * 255

    # --- Heuristic Calculations for Human-Readable Summary ---
    overall_mean_brightness = np.mean(mean_rgb)
    overall_mean_texture = np.mean(std_rgb)
    core_ring_count = max(1, num_rings // 4)
    core_brightness = np.mean(mean_rgb[0:core_ring_count])

    brightness_proxy = min(100.0, max(0.0, (overall_mean_brightness - 100) / 80 * 100))
    opacity_proxy = min(100.0, (overall_mean_texture / 45.0) * 100)
    
    report = "\n" + "="*50 + "\n"
    report += "   A-EYE MODEL EXPLAINABILITY REPORT\n"
    report += "="*50 + "\n"
    report += "Disclaimer: The following percentages are heuristic proxies derived\n"
    report += "from the model's internal statistics, not clinical measurements.\n\n"
    
    report += "--- Human-Readable Summary ---\n"
    report += f"  - Estimated Opacity Extent (Brightness Proxy): {brightness_proxy:.1f}%\n"
    report += f"  - Estimated Opacity Density (Texture Proxy):    {opacity_proxy:.1f}%\n\n"

    report += "--- Data-Driven Details for Thesis Discussion ---\n"
    for i in range(num_rings):
        mean_gray = np.mean(mean_rgb[i])
        std_gray = np.mean(std_rgb[i])
        report += f"  - Ring {i+1:02d}: Brightness={mean_gray:6.2f}, Texture={std_gray:6.2f}\n"
        
    report += "="*50 + "\n"
    return report

def predict(args):
    """Main function to load models and run prediction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_paths = glob.glob(os.path.join(args.model_dir, '*.pth'))
    if not model_paths:
        print(f"Error: No model files (.pth) found in '{args.model_dir}'.")
        return

    models = []
    for path in model_paths:
        if args.model_type == 'aeye':
            model = AEyeModel({'dims': args.dims, 'embed_dim': args.embed_dim, 'num_rings': args.num_rings})
        else:
            model = mobilevit_s()
            model.fc = nn.Linear(model.fc.in_features, 1)
        
        model.load_state_dict(torch.load(path, map_location=device))
        models.append(model.to(device).eval())
    print(f"Loaded {len(models)} models from '{args.model_dir}' for ensembling.")

    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = get_transforms(is_train=False)(image=image)['image'].unsqueeze(0).to(device)

    all_probs, all_tokens = [], []
    with torch.no_grad():
        for model in models:
            if args.model_type == 'aeye':
                output, tokens = model(input_tensor, return_tokens=True)
                all_tokens.append(tokens)
            else:
                output = model(input_tensor)
            all_probs.append(torch.sigmoid(output).item())

    final_prob = np.mean(all_probs)
    prediction = "Mature" if final_prob >= 0.5 else "Immature"
    
    print("\n--- PREDICTION RESULT ---")
    print(f"Image:           {os.path.basename(args.image_path)}")
    print(f"Predicted Class:   {prediction}")
    print(f"Confidence Score:  {final_prob:.2%}")
    
    if args.model_type == 'aeye' and all_tokens:
        print(generate_aeye_explanation(torch.stack(all_tokens, dim=0), args.num_rings))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single Image Prediction & Explainability Script")
    parser.add_argument('--model_type', required=True, choices=['aeye', 'baseline'])
    parser.add_argument('--num_rings', type=int, choices=[4, 8, 16], help="Required for 'aeye' model.")
    parser.add_argument('--model_dir', required=True, help='Directory containing trained .pth files.')
    parser.add_argument('--image_path', required=True, help='Path to the input image.')

    # --- Arguments for model configuration ---
    parser.add_argument('--dims', type=int, nargs='+', default=[32, 64, 128, 160])
    parser.add_argument('--embed_dim', type=int, default=256)

    args = parser.parse_args()

    if args.model_type == 'aeye' and args.num_rings is None:
        parser.error("--num_rings is required for --model_type 'aeye'")
    
    predict(args)