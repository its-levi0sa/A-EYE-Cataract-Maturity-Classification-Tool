import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(is_train=True):
    if is_train:
        """
        Augmentation pipeline for training.
        """
        return A.Compose([
            A.Resize(256, 256),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
            A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.12, rotate_limit=25, p=0.75),
            A.Blur(blur_limit=3, p=0.2),
            A.GridDistortion(p=0.2),
            A.OpticalDistortion(distort_limit=0.2, p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    else:
        """
        Preprocessing pipeline for validation/testing.
        """
        return A.Compose([
            A.Resize(256, 256),
            A.CLAHE(p=1.0),
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