# PLIK: src/dataset.py
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PneumoDataset(Dataset):
    def __init__(self, metadata_df, root_dir, phase='train', config=None):
        self.df = metadata_df
        self.root_dir = root_dir
        self.phase = phase
        self.config = config
        self.transforms = self._get_transforms()

    def _get_transforms(self):
        # --- ZMIANA KLUCZOWA ---
        # Usuwamy normalizację ImageNet (mean=0.485...).
        # Stosujemy proste skalowanie do zakresu [0, 1], co działało w "dobrym kodzie".
        
        if self.phase == 'train' and self.config['train']['use_augmentation']:
            prob = self.config['train']['aug_prob']
            # Zwiększamy nieco limit obrotu, żeby model się nie przeuczał
            return A.Compose([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=prob),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                # ToFloat dzieli przez max_value (255), dając zakres [0, 1]
                A.ToFloat(max_value=255.0), 
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.ToFloat(max_value=255.0),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_name = os.path.basename(row['ImagePath'])
        split_folder = row['Split']
        img_path = os.path.join(self.root_dir, split_folder, 'images', img_name)

        # --- ZMIANA KLUCZOWA ---
        # Wczytujemy jako GRAYSCALE (1 kanał), nie RGB!
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        
        # Albumentations wymaga wymiaru HxWxC nawet dla grayscale, więc dodajemy wymiar
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        mask_name = os.path.basename(row['MaskPath'])
        mask_path = os.path.join(self.root_dir, split_folder, 'masks', mask_name)
        
        # Maska też musi być wczytana bezpiecznie
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Binaryzacja (dla pewności, "dobry kod" to robił przez > 0)
            mask = (mask > 0).astype(np.float32)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # Augmentacja
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask']
            
            # Albumentations dla maski zwraca HxW, dodajemy kanał -> 1xHxW
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
        
        # Etykieta (nieużywana w czystej segmentacji, ale zostawiam dla kompatybilności)
        label_tensor = torch.tensor([1.0 if mask.max() > 0 else 0.0], dtype=torch.float32)

        return image_tensor, mask_tensor, label_tensor