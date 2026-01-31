"""
Program Title: final_evaluate.py

Programmers:
  Albonia, Jade Lorenz M.
  Caspe, Mark Vincent G.
  Rivera, Rei Djemf M.
  Velante, Kamilah Kaye M.
  Villegas, Jedidiah S.

Where the program fits in the general system design:
  This script is located in the `scripts/` directory. It executes the Phase 2
  (Final System Evaluation) protocols. Unlike the cross-validation script
  (`evaluate.py`) which aggregates an ensemble, this script evaluates the
  performance of the single, production-ready `.pth` artifact that will be
  deployed to the mobile application.

Date Written: July 2025
Date Revised: December 2025

Purpose:
  To validate the performance of the finalized model on the Test Set before
  deployment. It simulates the exact inference pipeline of the mobile app
  (Single Model Inference) to generate the definitive Accuracy, Precision,
  Recall, and F1-Score metrics reported in the Conclusion.

Data Structures, Algorithms, and Control:
  Data Structures:
    Single Model Instance: Loads a specific checkpoint file defined by the
      `--model_path` argument, rather than iterating through a list of folds.

  Algorithms:
    Standard Inference: Performs a forward pass on test images and applies a
      Sigmoid threshold (θ = 0.5) to generate binary classifications
      (0 = Immature, 1 = Mature).
    Confusion Matrix Visualization: Generates a heatmap to visualize the
      distribution of True Positives, False Positives, True Negatives, and
      False Negatives.

  Control:
    Architecture Switching: Dynamically instantiates the correct model class
      (`AEyeModel` or `MobileViT`) based on command-line arguments to ensure
      the state dictionary loads without key mismatch errors.
"""

import argparse
import logging
import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.aeye_model import AEyeModel
from src.baseline_model import MobileViT as BaselineModel 
from src.data_utils import get_transforms, AlbumentationsDataset as CataractDataset

def evaluate_single_model(model, dataloader, device):
    """
    Performs inference using a single model instance.

    Args:
        model (nn.Module): The loaded PyTorch model.
        dataloader (DataLoader): The test data iterator.
        device (torch.device): CPU or CUDA.

    Returns:
        tuple: (Predicted Labels, True Labels)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", file=sys.stdout, disable=not sys.stdout.isatty()):
            images = images.to(device)

            outputs = model(images)

            preds = torch.sigmoid(outputs).round().cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.numpy().flatten())

    return all_preds, all_labels

def main(args):
    """
    Main execution function for Final Model Evaluation.
    """
    log_name = f"final_evaluation_results_{os.path.basename(args.model_path).replace('.pth', '')}"
    log_file = f"results/{log_name}.txt"
    os.makedirs('results', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler(sys.stdout)])

    logging.info(f"--- Starting Final Evaluation for {log_name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- 1. Load Test Dataset ---
    image_paths = glob.glob(os.path.join(args.data_dir, '*/*.[jp][pn]g'))
    labels = [0 if 'immature' in path else 1 for path in image_paths]

    test_dataset = CataractDataset(image_paths, labels, transform=get_transforms(is_train=False))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    logging.info(f"Loaded {len(test_dataset)} images from the test set.")

    # --- 2. Initialize and Load the SINGLE Model ---
    if args.model_type == 'baseline':
        from src.baseline_model import mobilevit_s
        model = mobilevit_s()
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
    elif args.model_type == 'aeye':
        config = {
            "dims": args.dims,
            "embed_dim": args.embed_dim,
            "num_rings": args.num_rings
        }
        model = AEyeModel(config)
    else:
        raise ValueError("Invalid model_type specified.")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    logging.info(f"Successfully loaded single model from: {args.model_path}")

    # --- 3. Run Evaluation ---
    predictions, true_labels = evaluate_single_model(model, test_loader, device)

    # --- 4. Calculate and Log Metrics ---
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    logging.info("\n--- Final Model Performance ---")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")

    # --- 5. Save Confusion Matrix ---
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Immature', 'Mature'], yticklabels=['Immature', 'Mature'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {log_name}')
    cm_path = os.path.join("results", f"confusion_matrix_{log_name}.png")
    plt.savefig(cm_path)
    logging.info(f"Confusion matrix saved to {cm_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Final evaluation script for a single trained model.")
    # --- Key Arguments ---
    parser.add_argument('--model_path', type=str, required=True, help='Path to the single .pth model file to evaluate.')
    parser.add_argument('--model_type', type=str, required=True, choices=['baseline', 'aeye'], help='Type of model architecture.')
    parser.add_argument('--num_rings', type=int, choices=[4, 8, 16], help='Number of rings if model_type is aeye.')
    parser.add_argument('--data_dir', type=str, default='data/test', help='Directory containing the test data.')

    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation.')

    parser.add_argument('--dims', type=int, nargs='+', default=[32, 64, 128, 160])
    parser.add_argument('--embed_dim', type=int, default=256)

    args = parser.parse_args()
    
    if args.model_type == 'aeye' and args.num_rings is None:
        parser.error("--num_rings is required for --model_type 'aeye'")

    main(args)