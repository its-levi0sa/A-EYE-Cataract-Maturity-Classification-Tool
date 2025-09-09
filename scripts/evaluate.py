import os
import argparse
import logging
import glob
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from src.aeye_model import AEyeModel
from src.baseline_model import mobilevit_s
from src.data_utils import AlbumentationsDataset, get_transforms

def evaluate_ensemble(models, test_loader, device):
    """Performs ensembled inference and returns predictions and labels."""
    all_preds = []
    true_labels = []
    inference_times = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            inputs = inputs.to(device)
            batch_fold_probs = []

            # Get predictions from each model in the ensemble
            for model in models:
                start_time = time.time()
                outputs = model(inputs)
                inference_times.append(time.time() - start_time)
                
                probs = torch.sigmoid(outputs)
                batch_fold_probs.append(probs.cpu())

            # Average the probabilities across all fold models
            ensembled_probs = torch.stack(batch_fold_probs).mean(dim=0)
            preds = (ensembled_probs > 0.5).numpy().flatten()

            all_preds.extend(preds)
            true_labels.extend(labels.numpy().flatten())

    # Calculate average inference time per image
    avg_inference_time = np.sum(inference_times) / len(test_loader.dataset)
    return np.array(all_preds), np.array(true_labels), avg_inference_time

def main(args):
    """Main function to set up and run the evaluation process."""
    config = vars(args)

    # Setup Logging
    log_name = f"evaluation_results_{config['model_type']}" + (f"_{config['num_rings']}_rings" if config['model_type'] == 'aeye' else "")
    log_path = os.path.join("results", f"{log_name}.txt")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load Model Paths
    model_paths = glob.glob(os.path.join(config['model_dir'], '*.pth'))
    if not model_paths:
        logging.error(f"No models found in '{config['model_dir']}'. Please check the path.")
        return
    logging.info(f"Found {len(model_paths)} models for ensembling.")

    # Load all fold models
    models = []
    for path in model_paths:
        if config['model_type'] == 'aeye':
            model = AEyeModel({'dims': [32, 64, 128, 160], 'embed_dim': 256, 'num_rings': config['num_rings']})
        else:
            model = mobilevit_s()
            model.fc = nn.Linear(model.fc.in_features, 1)
        
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    # Setup Test Dataset
    image_paths = np.array(glob.glob(os.path.join(config['data_dir'], '*/*.[jp][pn]g')))
    labels = np.array([0 if 'immature' in path else 1 for path in image_paths])
    test_ds = AlbumentationsDataset(image_paths, labels, transform=get_transforms(is_train=False))
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    logging.info(f"Evaluating on {len(test_ds)} test images.")

    # Get predictions
    predictions, true_labels, avg_time = evaluate_ensemble(models, test_loader, device)

    # Calculate and Log Metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    logging.info("\n--- Final Evaluation Results ---")
    logging.info(f"Accuracy:  {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall:    {recall:.4f}")
    logging.info(f"F1-Score:  {f1:.4f}")
    logging.info(f"Avg. Inference Time: {avg_time * 1000:.2f} ms per image")

    # Generate and Save Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Immature', 'Mature'], yticklabels=['Immature', 'Mature'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {log_name}')
    
    cm_path = os.path.join("results", f"confusion_matrix_{log_name}.png")
    plt.savefig(cm_path)
    logging.info(f"Confusion matrix saved to: {cm_path}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified Evaluation Script for Cataract Classification")
    parser.add_argument('--model_type', type=str, required=True, choices=['aeye', 'baseline'])
    parser.add_argument('--num_rings', type=int, choices=[4, 8, 16], help="Required for 'aeye' model.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the test data directory.')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing trained model .pth files.')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    if args.model_type == 'aeye' and args.num_rings is None:
        parser.error("--num_rings is required when --model_type is 'aeye'")

    os.makedirs("results", exist_ok=True)
    main(args)