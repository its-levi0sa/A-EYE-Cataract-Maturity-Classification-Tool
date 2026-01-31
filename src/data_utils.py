"""
Program Title: data_utils.py

Programmers:
  Albonia, Jade Lorenz M.
  Caspe, Mark Vincent G.
  Rivera, Rei Djemf M.
  Velante, Kamilah Kaye M.
  Villegas, Jedidiah S.

Where the program fits in the general system design:
  This file is located in the `src/` directory and serves as the data processing
  engine for the A-EYE system. It is imported by `train.py`, `evaluate.py`, and
  `predict.py` to standardize how images are loaded, enhanced, and converted
  into tensors before entering the neural network.

Date Written: July 2025
Date Revised: December 2025

Purpose:
  To standardize the preprocessing logic, ensuring that all input images undergo
  Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance cataract
  features.

Data Structures, Algorithms, and Control:
  Data Structures:
    A.Compose: An Albumentations pipeline wrapper that sequences multiple
      image transformation operations.
    AlbumentationsDataset: A custom PyTorch Dataset class that bridges raw
      image files on disk with the augmentation pipeline.

  Algorithms:
    CLAHE (Contrast Limited Adaptive Histogram Equalization): An algorithm
      applied in the LAB color space to locally enhance contrast, making
      opacity patterns in the lens more visible.
    Online Augmentation: A specific combination of geometric distortions (Grid,
      Optical) and regularization techniques (Cutout, Blur) designed to
      simulate diverse cataract presentations and prevent overfitting.

  Control:
    Pipeline Branching: The `get_transforms` function acts as a switch. If
      `is_train=True`, it activates the full stochastic augmentation suite.
      If `is_train=False`, it activates a deterministic evaluation pipeline
      that only applies resizing and CLAHE.
"""

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Deterministic CLAHE function ---
def clahe_deterministic(image, clip_limit=2.0, tile_grid_size=(8, 8), **kwargs):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.
    
    Algorithm:
      1. Convert RGB image to LAB color space.
      2. Apply CLAHE only to the L-channel (Lightness) to preserve color info.
      3. Merge channels and convert back to RGB.
    
    Args:
        image (np.array): Input image in RGB format.
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Size of grid for histogram equalization.
        
    Returns:
        np.array: Contrast-enhanced image in RGB format.
    """
    # Convert to LAB color space
    lab_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab_img)
    
    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    
    # Merge the CLAHE enhanced L-channel back with A and B channels
    updated_lab_img = cv2.merge((l_clahe, a, b))
    
    # Convert back to RGB
    return cv2.cvtColor(updated_lab_img, cv2.COLOR_LAB2RGB)


def get_transforms(is_train=True):
    """
    Constructs the data augmentation pipeline.
    
    Args:
        is_train (bool): If True, returns the full training pipeline (BSRDA).
                         If False, returns the validation/test pipeline.
    
    Returns:
        A.Compose: The composed albumentations transformation pipeline.
    """
    if is_train:
        """
        # Augmentation pipeline for training.
        """
        return A.Compose([
            A.Resize(256, 256),
            A.Lambda(image=clahe_deterministic, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
            A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.12, rotate_limit=25, p=0.75),
            A.Blur(blur_limit=3, p=0.2),
            A.GridDistortion(p=0.2),
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.2),
            A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    else:
        """
        Preprocessing pipeline for validation/testing.
        """
        return A.Compose([
            A.Resize(256, 256),
            # --- DETERMINISM ---
            A.Lambda(image=clahe_deterministic, p=1.0),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])

class AlbumentationsDataset(Dataset):
    """
    Custom PyTorch Dataset for Albumentations.
    """
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of file paths to images.
            labels (list): List of corresponding labels (0 or 1).
            transform (A.Compose): Albumentations transformation pipeline.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image=image)['image']
            
        return image, label