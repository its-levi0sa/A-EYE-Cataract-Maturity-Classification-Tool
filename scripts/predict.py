import os
import argparse
import glob
import numpy as np
import torch
import cv2
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.aeye_model import AEyeModel
from src.data_utils import get_transforms

def generate_aeye_explanation(tokens_tensor, num_rings):
    """
    Generates a user-friendly summary and a detailed statistical report from
    the A-EYE model's internal radial tokens using a data-driven,
    percentile-based normalization approach.
    """
    # --- CALIBRATED CONSTANTS ---
    P1_BRIGHTNESS = 85.30
    P99_BRIGHTNESS = 210.10
    P1_TEXTURE = 5.70
    P99_TEXTURE = 65.20
    
    # --- Calculation ---
    # The tokens_tensor is already the average from the ensemble
    avg_tokens = tokens_tensor.squeeze(0).cpu().numpy()
    
    mean_rgb = (avg_tokens[:, 0:3] * 0.5 + 0.5) * 255
    std_rgb = (avg_tokens[:, 3:6] * 0.5) * 255

    overall_mean_brightness = np.mean(mean_rgb)
    overall_mean_texture = np.mean(std_rgb)

    # --- Robust Percentile-Based Normalization ---
    brightness_proxy = 100 * (overall_mean_brightness - P1_BRIGHTNESS) / (P99_BRIGHTNESS - P1_BRIGHTNESS)
    opacity_proxy = 100 * (overall_mean_texture - P1_TEXTURE) / (P99_TEXTURE - P1_TEXTURE)
    
    brightness_proxy = np.clip(brightness_proxy, 0, 100)
    opacity_proxy = np.clip(opacity_proxy, 0, 100)
    
    final_opacity_proxy = opacity_proxy
    brightness_threshold = 5.0
    if brightness_proxy < brightness_threshold:
        final_opacity_proxy = opacity_proxy * (brightness_proxy / brightness_threshold)

    # --- Build the Report String ---
    report = "\n" + "="*50 + "\n"
    report += "   A-EYE MODEL EXPLAINABILITY REPORT\n"
    report += "="*50 + "\n"
    report += "Disclaimer: The following percentages are data-driven proxies derived\n"
    report += "from the model's statistics, not direct clinical measurements.\n\n"
    
    report += "--- Human-Readable Summary ---\n"
    report += f"   - Estimated Opacity Extent (Brightness Proxy): {brightness_proxy:.1f}%\n"
    report += f"   - Estimated Opacity Density (Texture Proxy):   {final_opacity_proxy:.1f}%\n\n"

    report += "--- Data-Driven Details for Thesis Discussion ---\n"
    for i in range(num_rings):
        mean_gray = np.mean(mean_rgb[i])
        std_gray = np.mean(std_rgb[i])
        report += f"   - Ring {i+1:02d}: Brightness={mean_gray:6.2f}, Texture={std_gray:6.2f}\n"
        
    report += "="*50 + "\n"
    return report

def predict(args):
    """Main function to load models and run prediction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_paths = glob.glob(os.path.join(args.model_dir, 'best_model_fold_*.pth'))
    if not model_paths:
        print(f"Error: No model files found in '{args.model_dir}'.")
        return

    models = []
    model_config = {'dims': args.dims, 'embed_dim': args.embed_dim, 'num_rings': args.num_rings}
    for path in model_paths:
        model = AEyeModel(model_config)
        model.load_state_dict(torch.load(path, map_location=device))
        models.append(model.to(device).eval())
    print(f"Loaded {len(models)} models from '{args.Mymodel_dir}' for ensembling.")

    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = get_transforms(is_train=False)(image=image)['image'].unsqueeze(0).to(device)

    all_probs, all_tokens = [], []
    with torch.no_grad():
        for model in models:
            output, tokens = model(input_tensor, return_tokens=True)
            all_tokens.append(tokens)
            all_probs.append(torch.sigmoid(output).item())

    # Average the predictions and tokens from the ensemble
    final_prob = np.mean(all_probs)
    avg_tokens_tensor = torch.stack(all_tokens, dim=0).mean(dim=0)
    prediction = "Mature" if final_prob >= 0.5 else "Immature"
    
    print("\n--- PREDICTION RESULT ---")
    print(f"Image:             {os.path.basename(args.image_path)}")
    print(f"Predicted Class:   {prediction}")
    print(f"Confidence Score:  {final_prob:.2%}")
    
    print(generate_aeye_explanation(avg_tokens_tensor, args.num_rings))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ensemble Prediction & Explainability Script")
    parser.add_argument('--model_dir', required=True, help="Directory containing the 5 best fold models (e.g., 'saved_models/aeye_4_ring').")
    parser.add_argument('--image_path', required=True, help='Path to the input image.')
    parser.add_argument('--num_rings', type=int, default=4, help="Number of rings for the A-EYE model.")

    # A-EYE model config
    parser.add_argument('--dims', type=int, nargs='+', default=[32, 64, 128, 160])
    parser.add_argument('--embed_dim', type=int, default=256)

    args = parser.parse_args()    
    predict(args)