"""
Program Title: train.py

Programmers:
  Albonia, Jade Lorenz M.
  Caspe, Mark Vincent G.
  Rivera, Rei Djemf M.
  Velante, Kamilah Kaye M.
  Villegas, Jedidiah S.

Where the program fits in the general system design:
  This script is located in the `scripts/` directory. It functions as the
  primary experimental harness for Phase 1 (Ablation Study & Baseline Comparison).
  Unlike `final_train.py` (which trains on the full dataset), this script
  executes Stratified K-Fold Cross-Validation to generate statistically
  robust performance metrics (Mean ± Std Dev) for the thesis.

Date Written: July 2025
Date Revised: December 2025

Purpose:
  To rigorously evaluate model stability and generalization. It implements a
  5-Fold Cross-Validation loop that trains five independent instances of the
  model on stratified data splits, ensuring that performance gains are consistent
  and not artifacts of a specific random seed.

Data Structures, Algorithms, and Control:
  Data Structures:
    StratifiedKFold: A Scikit-Learn cross-validator that splits the dataset
      into k=5 disjoint folds while preserving the percentage of samples for
      each class (Mature vs. Immature).

  Algorithms:
    Mixed Precision Training (AMP): Utilizes `torch.cuda.amp` to perform
      forward/backward passes in FP16 (Half-Precision), reducing memory
      footprint and accelerating training on Tensor Core GPUs.
    Cosine Annealing with Warm Restarts: A learning rate scheduler that
      periodically resets the learning rate to prevent the optimizer from
      getting stuck in local minima.
    Early Stopping: Monitors the Validation F1-Score and terminates training
      if performance plateaus for `patience=20` epochs to prevent overfitting.

  Control:
    Strict Determinism: Sets seeds for Python, NumPy, PyTorch, and CUDA
      backends to ensure that the K-Fold splits and weight initializations
      are identical across runs.
    Dynamic Model Instantiation: Switches between `AEyeModel` and `MobileViT`
      architectures based on the `--model_type` flag.
"""

import os
import argparse
import logging
import glob
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import random

# Set environment variable BEFORE importing torch for full determinism
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from src.aeye_model import AEyeModel
from src.baseline_model import mobilevit_s
from src.data_utils import AlbumentationsDataset, get_transforms
from src.utils import FocalLoss

# --- Training reproducibility function ---
def seed_everything(seed=42):
    """
    Seeds all relevant random number generators (Python, NumPy, PyTorch, CUDA)
    to ensure bit-exact reproducibility of results.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """
    Seeds the DataLoader workers to ensure data augmentation is deterministic.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# --- Training and Validation logic ---
def train_one_fold(fold, model, train_loader, val_loader, config):
    """
    Executes the training and validation loop for a single K-Fold split.
    
    Args:
        fold (int): The current fold index (0-4).
        model (nn.Module): The model instance to train.
        train_loader (DataLoader): Iterator for training data.
        val_loader (DataLoader): Iterator for validation data.
        config (dict): Configuration dictionary containing hyperparameters.
        
    Returns:
        float: The best Validation F1-Score achieved in this fold.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = FocalLoss(alpha=0.25, gamma=2.5)
    
    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader) * 10, T_mult=1, eta_min=1e-6)
    
    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler()

    best_val_f1 = 0.0
    patience_counter = 0
    logging.info(f"--- Starting Fold {fold+1}/{config['n_splits']} ---")

    for epoch in range(config['epochs']):
        # --- Training Phase ---
        model.train()
        train_loop = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}", leave=False, file=sys.stdout)
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

            # Forward Pass with Mixed Precision
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Backward Pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Step Scheduler per batch
            scheduler.step()

            train_loop.set_postfix(loss=loss.item())
        
        # Log training summary
        final_loss = loss.item()
        avg_speed = train_loop.format_dict.get("rate", 0)
        logging.info(
            f"Epoch {epoch+1} - Train Summary | Speed: {avg_speed:.2f} it/s, Loss: {final_loss:.5f}"
        )

        # --- Validation Phase ---
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs.to(device))
                preds = torch.sigmoid(outputs) > 0.5
                val_preds.extend(preds.cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy().flatten())
        
        # Calculate Validation Metrics
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)

        logging.info(
            f"Epoch {epoch+1} | Val Acc: {val_accuracy:.5f}, P: {val_precision:.5f}, R: {val_recall:.5f}, F1: {val_f1:.5f}"
        )

        # --- Checkpointing & Early Stopping ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config['save_dir'], f"best_model_fold_{fold+1}.pth"))
            logging.info(f"  -> New best model saved with F1: {val_f1:.5f}")
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            logging.info(f"  -> Early stopping triggered. Best F1: {best_val_f1:.5f}")
            break
            
    return best_val_f1

# --- MAIN EXECUTION SCRIPT ---
def main(args):
    """
    Main function to orchestrate the K-Fold Cross-Validation process.
    """
    # Enforce Determinism globally
    torch.use_deterministic_algorithms(True)
    
    config = vars(args)
    
    # Setup Logging
    log_name = f"training_log_{config['model_type']}" + (f"_{config['num_rings']}_rings" if config['model_type'] == 'aeye' else "")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join("results", f"{log_name}.txt")), logging.StreamHandler()])

    # Load Data Paths
    image_paths = np.array(glob.glob(os.path.join(config['data_dir'], '*/*.[jp][pn]g')))
    labels = np.array([0 if 'immature' in path else 1 for path in image_paths])
    logging.info(f"Found {len(image_paths)} images for training.")

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=42)
    fold_scores = []
    
    # --- K-Fold Loop ---
    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
        seed_everything(seed=42 + fold)
        g = torch.Generator()
        g.manual_seed(42 + fold)
        
        # --- Model Initialization ---
        if config['model_type'] == 'aeye':
            model = AEyeModel(config)
        else:
            model = mobilevit_s()
            model.fc = nn.Linear(model.fc.in_features, 1)

        # --- Dataset Initialization ---
        # is_train=True for Training (Apply Augmentation)
        # is_train=False for Validation (No Augmentation)
        train_ds = AlbumentationsDataset(image_paths[train_idx], labels[train_idx], transform=get_transforms(is_train=True))
        val_ds = AlbumentationsDataset(image_paths[val_idx], labels[val_idx], transform=get_transforms(is_train=False))
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True, 
            worker_init_fn=seed_worker, 
            generator=g
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True, 
            worker_init_fn=seed_worker,
            generator=g
        )
        
        # --- Run Training for Current Fold ---
        fold_f1 = train_one_fold(fold, model, train_loader, val_loader, config)
        fold_scores.append(fold_f1)
    
    # --- Final Report ---
    logging.info("\n--- K-Fold Training Finished ---")
    logging.info(f"Fold F1 Scores: {[f'{s:.5f}' for s in fold_scores]}")
    logging.info(f"Average F1-Score: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified Training Script for Cataract Classification")
    parser.add_argument('--model_type', type=str, required=True, choices=['aeye', 'baseline'])
    parser.add_argument('--num_rings', type=int, choices=[4, 8, 16], help="Required for 'aeye' model.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to training data.')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Root directory to save models.')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--patience', type=int, default=20, help="Epochs for early stopping.")
    parser.add_argument('--n_splits', type=int, default=5, help="Number of K-Fold splits.")
    
    parser.add_argument('--dims', type=int, nargs='+', default=[32, 64, 128, 160])
    parser.add_argument('--embed_dim', type=int, default=256)

    args = parser.parse_args()

    if args.model_type == 'aeye' and args.num_rings is None:
        parser.error("--num_rings is required when --model_type is 'aeye'")

    model_folder = args.model_type + (f"_{args.num_rings}_ring" if args.model_type == 'aeye' else "")
    args.save_dir = os.path.join(args.save_dir, model_folder)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    main(args)