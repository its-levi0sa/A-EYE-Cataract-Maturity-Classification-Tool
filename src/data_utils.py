import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Deterministic CLAHE function ---
def clahe_deterministic(image, clip_limit=2.0, tile_grid_size=(8, 8), **kwargs):
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
    if is_train:
        """
        Augmentation pipeline for training.
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