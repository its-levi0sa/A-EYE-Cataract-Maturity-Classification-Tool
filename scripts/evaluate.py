"""
Program Title: evaluate.py

Programmers:
  Albonia, Jade Lorenz M.
  Caspe, Mark Vincent G.
  Rivera, Rei Djemf M.
  Velante, Kamilah Kaye M.
  Villegas, Jedidiah S.

Where the program fits in the general system design:
  This script is located in the `scripts/` directory and executes the formal
  evaluation phase (Phase 2). It aggregates the 5 independent model checkpoints
  generated during K-Fold Cross-Validation (from `train.py`) and evaluates them
  as a "Soft Voting" Ensemble on the unseen Test Set.

Date Written: July 2025
Date Revised: December 2025

Purpose:
  To measure the model's generalization capability on strictly unseen data.
  By averaging the predictions of 5 models trained on different data subsets,
  this script produces the definitive performance metrics (Accuracy, F1-Score)
  and Confusion Matrices used in the Results and Discussion chapter.

Data Structures, Algorithms, and Control:
  Data Structures:
    Ensemble List (List[nn.Module]): A collection of 5 distinct model instances,
      each loaded with weights from a specific training fold.
    Prediction Matrix (Numpy Array): Stores the probability outputs from all
      models to facilitate vectorized averaging.

  Algorithms:
    Soft Voting Ensemble: Instead of voting on the final class labels (Hard Voting),
      this algorithm averages the continuous probability scores from all 5 models.
      This technique is statistically robust against outliers and reduces variance.
    Confusion Matrix Generation: Visualizes the True Positive vs. False Negative
      rates using a Seaborn heatmap.

  Control:
    Pre-flight Integrity Checks: Validates the existence of model checkpoints
      and test data before initializing the GPU, preventing runtime failures.
    Deterministic Loading: Forces file paths to be sorted alphabetically to
      ensure that "Fold 1" is always loaded into index 0, guaranteeing
      reproducible debugging.
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

from src.aeye_model import AEyeModel
from src.baseline_model import mobilevit_s
from src.data_utils import AlbumentationsDataset, get_transforms
from src.utils import seed_everything

# --- Set environment variable BEFORE importing torch for full determinism ---
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def main(args):
    """
    Main execution function for the Ensemble Evaluation.
    
    Workflow:
      1. Setup logging and device (GPU/CPU).
      2. Identify and load the 5 model checkpoints (folds).
      3. Run inference with EACH model on the Test Set.
      4. Average the predictions (Ensembling).
      5. Calculate metrics and save the Confusion Matrix.
    """
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

    # --- Load Ensemble Models ---
    models = []
    for path in model_paths:
        if args.model_type == 'aeye':
            # Instantiate A-EYE with fixed architecture
            model = AEyeModel({'dims': [32, 64, 128, 160], 'embed_dim': 256, 'num_rings': args.num_rings})
        else:
            # Instantiate Baseline
            model = mobilevit_s()
            model.fc = nn.Linear(model.fc.in_features, 1)
        
        # Load weights
        model.load_state_dict(torch.load(path, map_location=device))
        models.append(model.to(device).eval())

    # --- Load Test Data ---
    test_labels = [0 if 'immature' in path else 1 for path in test_image_paths]
    test_ds = AlbumentationsDataset(test_image_paths, test_labels, transform=get_transforms(is_train=False))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logging.info(f"Loaded {len(test_ds)} images from the test set.")

    # --- Run Inference (Per Model) ---
    all_fold_preds = []
    
    with torch.no_grad():
        for i, model in enumerate(models):
            fold_preds = []
            for inputs, _ in tqdm(test_loader, desc=f"Evaluating Fold {i+1}/{len(models)}", leave=False):
                outputs = model(inputs.to(device))
                preds = torch.sigmoid(outputs)
                fold_preds.extend(preds.cpu().numpy().flatten())
            all_fold_preds.append(fold_preds)

    # --- Ensemble Averaging (Soft Voting) ---
    # Calculate the mean probability across the 5 models
    avg_preds = np.mean(all_fold_preds, axis=0)

    # Apply Thresholding (0.5)
    final_preds = (avg_preds >= 0.5).astype(int)

    # --- Calculate Metrics ---
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