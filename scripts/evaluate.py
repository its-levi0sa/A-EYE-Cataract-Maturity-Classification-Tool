import os
import argparse
import logging
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys

# --- Import models and utils from 'src' folder ---
from src.aeye_model import AEyeModel
from src.baseline_model import mobilevit_s
from src.data_utils import AlbumentationsDataset, get_transforms
from src.utils import seed_everything

# --- Set environment variable BEFORE importing torch for full determinism ---
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def main(args):
    """Main function to set up and run the evaluation process."""
    
    seed_everything(42)
    
    # --- Enforce PyTorch's strictest deterministic settings ---
    torch.use_deterministic_algorithms(True)

    # --- Pre-flight checks to validate paths before starting ---
    print(f"--- Running pre-flight checks for model: {args.model_type} ---")
    
    # Sort glob results for consistent file ordering
    model_paths = sorted(glob.glob(os.path.join(args.model_dir, '*.pth')))
    if not model_paths:
        print(f"FATAL ERROR: No model files (.pth) were found in the directory '{args.model_dir}'.")
        print("   Please check the --model_dir path in your command.")
        sys.exit(1)
    else:
        print(f"Found {len(model_paths)} model file(s) in '{args.model_dir}'.")

    # Sort glob results for consistent file ordering
    test_image_paths = sorted(glob.glob(os.path.join(args.data_dir, '*/*.[jp][pn]g')))
    if not test_image_paths:
        print(f"FATAL ERROR: No image files were found in the subfolders of '{args.data_dir}'.")
        print("   Please check the --data_dir path and ensure it has 'immature'/'mature' subfolders with images.")
        sys.exit(1)
    else:
        print(f"Found {len(test_image_paths)} images in '{args.data_dir}'.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Setup Logging ---
    log_name = f"evaluation_results_{args.model_type}" + (f"_{args.num_rings}_rings" if args.model_type == 'aeye' else "")
    log_path = os.path.join("results", f"{log_name}.txt")
    os.makedirs("results", exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler()])

    logging.info(f"--- Starting Evaluation for {log_name.replace('_', ' ').title()} ---")
    logging.info(f"Using device: {device}")
    logging.info(f"Found {len(model_paths)} models for ensembling.")

    # Load models
    models = []
    for path in model_paths:
        if args.model_type == 'aeye':
            model = AEyeModel({'dims': [32, 64, 128, 160], 'embed_dim': 256, 'num_rings': args.num_rings})
        else: # baseline
            model = mobilevit_s()
            model.fc = nn.Linear(model.fc.in_features, 1)
        model.load_state_dict(torch.load(path, map_location=device))
        models.append(model.to(device).eval())

    # Load test data
    test_labels = [0 if 'immature' in path else 1 for path in test_image_paths]
    test_ds = AlbumentationsDataset(test_image_paths, test_labels, transform=get_transforms(is_train=False))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logging.info(f"Loaded {len(test_ds)} images from the test set.")

    # Run evaluation
    all_fold_preds = []
    total_inference_time = 0
    with torch.no_grad():
        for i, model in enumerate(models):
            fold_preds = []
            start_time = time.time()
            for inputs, _ in tqdm(test_loader, desc=f"Evaluating Fold {i+1}/{len(models)}", leave=False):
                outputs = model(inputs.to(device))
                preds = torch.sigmoid(outputs)
                fold_preds.extend(preds.cpu().numpy().flatten())
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            all_fold_preds.append(fold_preds)

    # Ensemble predictions and calculate metrics
    avg_preds = np.mean(all_fold_preds, axis=0)
    final_preds = (avg_preds >= 0.5).astype(int)
    accuracy = accuracy_score(test_labels, final_preds)
    precision = precision_score(test_labels, final_preds, zero_division=0)
    recall = recall_score(test_labels, final_preds, zero_division=0)
    f1 = f1_score(test_labels, final_preds, zero_division=0)
    avg_inference_time_per_image = (total_inference_time / len(models) / len(test_ds)) * 1000

    logging.info("\n--- Final Ensemble Performance ---")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")
    logging.info(f"Avg. Inference Time (per image): {avg_inference_time_per_image:.2f} ms")

    # Generate and save confusion matrix
    cm = confusion_matrix(test_labels, final_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Immature', 'Mature'], yticklabels=['Immature', 'Mature'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {log_name.replace("_", " ").title()}')
    cm_path = os.path.join("results", f"confusion_matrix_{log_name}.png")
    plt.savefig(cm_path)
    plt.close()
    logging.info(f"Confusion matrix saved to {cm_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified Evaluation Script")
    parser.add_argument('--model_type', required=True, choices=['aeye', 'baseline'])
    parser.add_argument('--num_rings', type=int, choices=[4, 8, 16], help="Required for 'aeye' model.")
    parser.add_argument('--model_dir', required=True, help='Directory containing trained .pth model folds.')
    parser.add_argument('--data_dir', required=True, help='Path to the test data directory.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation.')
    args = parser.parse_args()

    if args.model_type == 'aeye' and args.num_rings is None:
        parser.error("--num_rings is required for --model_type 'aeye'")
    
    main(args)