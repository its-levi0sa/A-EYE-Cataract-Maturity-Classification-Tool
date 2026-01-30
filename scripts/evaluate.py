"""
Program Title: evaluate.py

Programmers:
  Albonia, Jade Lorenz M.
  Caspe, Mark Vincent G.
  Rivera, Rei Djemf M.
  Velante, Kamilah Kaye M.
  Villegas, Jedidiah S.

Where the program fits in the general system design:
  Located in `scripts/`. This script runs the formal testing phase. It loads
  the 5 models trained during Cross-Validation (from `train.py`) and tests
  them as an ensemble on the unseen Test Set. This generates the final numbers
  and Confusion Matrices for the study.

Date Written: July 2025
Date Revised: December 2025

Purpose:
  To provide a fair evaluation of the models. It ensures that the
  metrics report are based on data the model has never seen before. It also
  generates the confusion matrix plots for error analysis.

Data Structures, Algorithms, and Control:
  Data Structures:
    Ensemble Logic: Load 5 different .pth files (one for each fold).
    Numpy Arrays: Store predictions in arrays to easily calculate averages.

  Algorithms:
    Ensemble Averaging: For every test image, acquired 5 different probability
      scores (one from each fold model) and average them together. If the
      average > 0.5, it is classified as Mature. This reduces the risk of one
      bad model ruining the results.
    Confusion Matrix Generation: Uses Seaborn to draw the heatmap that shows
      True Positives vs. False Negatives.

  Control:
    Pre-flight Checks: The script verifies that the model folder and data folder
      actually contain files before crashing halfway through.
    Argparse Logic: Forces to specify which model to test (Baseline
      vs. A-EYE) and where the files are.
"""

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
    torch.use_deterministic_algorithms(True)

    # --- Pre-flight checks ---
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

    # --- Run evaluation ---
    all_fold_preds = []
    
    with torch.no_grad():
        for i, model in enumerate(models):
            fold_preds = []
            for inputs, _ in tqdm(test_loader, desc=f"Evaluating Fold {i+1}/{len(models)}", leave=False):
                outputs = model(inputs.to(device))
                preds = torch.sigmoid(outputs)
                fold_preds.extend(preds.cpu().numpy().flatten())
            all_fold_preds.append(fold_preds)

    # --- Ensemble predictions and calculate metrics ---
    avg_preds = np.mean(all_fold_preds, axis=0)
    final_preds = (avg_preds >= 0.5).astype(int)
    accuracy = accuracy_score(test_labels, final_preds)
    precision = precision_score(test_labels, final_preds, zero_division=0)
    recall = recall_score(test_labels, final_preds, zero_division=0)
    f1 = f1_score(test_labels, final_preds, zero_division=0)

    logging.info("\n--- Final Ensemble Performance ---")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")

    # --- Generate and save confusion matrix ---
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