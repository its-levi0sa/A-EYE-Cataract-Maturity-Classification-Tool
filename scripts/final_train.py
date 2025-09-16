import argparse
import logging
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.aeye_model import AEyeModel
from src.data_utils import get_transforms, AlbumentationsDataset
from src.utils import FocalLoss

# --- Training reproducibility function ---
def set_seed(seed=42):
    """Sets the seed for reproducibility across all libraries."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    """Main function to run the final model training on all data."""
    set_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- 1. Load the FULL Dataset for Training ---
    image_paths = glob.glob(os.path.join(args.data_dir, '*/*.[jp][pn]g'))
    labels = [0 if 'immature' in path else 1 for path in image_paths]

    full_dataset = AlbumentationsDataset(image_paths, labels, transform=get_transforms(is_train=True))
    train_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    logging.info(f"Loaded {len(full_dataset)} images for final training.")

    # --- 2. Initialize the Model ---
    config = vars(args)
    model = AEyeModel(config)
    model.to(device)

    # --- 3. Setup Optimizer, Loss, and Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = FocalLoss(alpha=0.25, gamma=2.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=len(train_loader) * 10, T_mult=1, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler()

    # --- 4. Final Training Loop ---
    logging.info(f"Starting final training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        total_loss, num_batches, total_samples = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            total_samples += images.size(0) 

        end_time = time.time()
        avg_loss = total_loss / num_batches
        avg_it_speed = num_batches / (end_time - start_time)

        logging.info(
            f"Epoch {epoch+1} Train Summary |     Speed: {avg_it_speed:.2f} it/s     Loss: {avg_loss:.5f}"
        )



    # --- 5. Save the Final Trained Model ---
    os.makedirs(args.save_dir, exist_ok=True)
    model_name = f"{args.model_type}_{args.num_rings}_rings_final_model.pth"
    save_path = os.path.join(args.save_dir, model_name)
    torch.save(model.state_dict(), save_path)
    logging.info(f"✅ Final model for deployment saved successfully to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Final training script for creating a deployment-ready model.")

    # --- Key Arguments ---
    parser.add_argument('--model_type', type=str, default='aeye', help='Type of model to train.')
    parser.add_argument('--num_rings', type=int, default=4, help='Number of rings for the A-EYE model.')
    parser.add_argument('--data_dir', type=str, default='data/train', help='Directory containing the full training dataset.')
    parser.add_argument('--save_dir', type=str, default='saved_models/final', help='Directory to save the final deployment model.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

    # --- Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=100, help='Fixed number of epochs, determined from CV logs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate for the AdamW optimizer.')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay (L2 penalty) for the optimizer.')

    # --- Model Architecture Arguments ---
    parser.add_argument('--dims', type=int, nargs='+', default=[32, 64, 128, 160], help='Dimensions for the model stages.')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension for the transformer blocks.')

    args = parser.parse_args()
    
    # Setup basic logging to see output in the console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    main(args)