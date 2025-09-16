import argparse
import logging
import os
import sys
import time
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
from src.baseline_model import BaselineModel
from src.data_utils import get_transforms, CataractDataset

def evaluate_single_model(model, dataloader, device):
    """Evaluates a single model on the provided dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    inference_times = []
    peak_memory_usages = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)

            # Measure performance
            torch.cuda.synchronize()
            start_time = time.time()

            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            outputs = model(images)

            torch.cuda.synchronize()
            end_time = time.time()

            inference_times.append(end_time - start_time)
            if device.type == 'cuda':
                peak_memory_usages.append(torch.cuda.max_memory_allocated() / (1024 * 1024)) # In MB

            preds = torch.sigmoid(outputs).round().cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.numpy().flatten())

    return all_preds, all_labels, inference_times, peak_memory_usages

def main(args):
    """Main function to run the final model evaluation."""
    log_name = f"final_evaluation_results_{os.path.basename(args.model_path).replace('.pth', '')}"
    log_file = f"results/{log_name}.txt"
    os.makedirs('results', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler(sys.stdout)])

    logging.info(f"--- Starting Final Evaluation for {log_name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- 1. Load Test Dataset ---
    test_dataset = CataractDataset(args.data_dir, transform=get_transforms(train=False))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    logging.info(f"Loaded {len(test_dataset)} images from the test set.")

    # --- 2. Initialize and Load the SINGLE Model ---
    if args.model_type == 'baseline':
        model = BaselineModel(dims=args.dims)
    elif args.model_type == 'aeye':
        model = AEyeModel(dims=args.dims, embed_dim=args.embed_dim, num_rings=args.num_rings)
    else:
        raise ValueError("Invalid model_type specified.")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    logging.info(f"Successfully loaded single model from: {args.model_path}")

    # --- 3. Run Evaluation ---
    predictions, true_labels, inference_times, peak_memory = evaluate_single_model(model, test_loader, device)

    # --- 4. Calculate and Log Metrics ---
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    avg_inference_time = np.mean(inference_times) * 1000 / args.batch_size
    avg_peak_memory = np.mean(peak_memory) if peak_memory else 0

    logging.info("\n--- Final Model Performance ---")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")
    logging.info(f"Avg. Inference Time (per image): {avg_inference_time:.2f} ms")
    logging.info(f"Avg. Peak Memory Usage: {avg_peak_memory:.2f} MB")

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
    parser.add_argument('--num_rings', type=int, default=4, help='Number of rings if model_type is aeye.')
    parser.add_argument('--data_dir', type=str, default='data/test', help='Directory containing the test data.')

    # --- Hyperparameters ---
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation. Set to 1 for precise per-image timing.')

    # --- Model Architecture Arguments ---
    parser.add_argument('--dims', type=int, nargs='+', default=[32, 64, 128, 160])
    parser.add_argument('--embed_dim', type=int, default=256)

    args = parser.parse_args()
    main(args)