import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PneumoDataset(Dataset):
    """
    PyTorch Dataset for Pneumothorax Segmentation and Classification.
    Reads processed PNG files from disk and applies augmentations.
    Returns: Image, Mask, and Classification Label.
    """

    def __init__(self, metadata_df, root_dir, phase='train', config=None):
        """
        Args:
            metadata_df (pd.DataFrame): Dataframe with file paths.
            root_dir (str): Root directory of the dataset (e.g., Kaggle Input).
            phase (str): 'train', 'val', or 'test'.
            config (dict): Configuration dictionary.
        """
        self.df = metadata_df
        self.root_dir = root_dir
        self.phase = phase
        self.config = config
        self.transforms = self._get_transforms()

    def _get_transforms(self):
        """
        Returns Albumentations composition based on the phase.
        Train phase includes geometric and pixel-level augmentations.
        """
        mean = (0.485, 0.456, 0.406) # ImageNet defaults
        std = (0.229, 0.224, 0.225)
        
        if self.phase == 'train' and self.config['train']['use_augmentation']:
            prob = self.config['train']['aug_prob']
            rot_lim = self.config['train']['aug_rotate_limit']
            
            return A.Compose([
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=rot_lim, p=prob),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=prob),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        else:
            # Validation/Test: Only Normalize and ToTensor
            return A.Compose([
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Resolve Image Path
        # Handle cases where CSV has absolute paths from preprocessing step
        img_name = os.path.basename(row['ImagePath'])
        # Construct path relative to current dataset location
        # Structure: root_dir / split / images / filename
        split_folder = row['Split'] if 'Split' in row else 'test'
        img_path = os.path.join(self.root_dir, split_folder, 'images', img_name)

        # 2. Read Image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. Read Mask and Generate Label
        # Default for test set
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        label = 0.0

        if self.phase != 'test':
            mask_name = os.path.basename(row['MaskPath'])
            mask_path = os.path.join(self.root_dir, split_folder, 'masks', mask_name)
            
            if os.path.exists(mask_path):
                mask_data = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_data is not None:
                    # Binarize: 0 or 255 -> 0.0 or 1.0
                    mask = (mask_data > 127).astype(np.float32)
            
            # Generate classification label: 1 if pneumothorax exists, else 0
            if mask.max() > 0:
                label = 1.0

        # 4. Apply Augmentations
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask']
            
            # Ensure mask has channel dimension (1, H, W)
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
        
        label_tensor = torch.tensor([label], dtype=torch.float32)

        return image_tensor, mask_tensor, label_tensor