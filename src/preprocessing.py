import os
import glob
import pandas as pd
import numpy as np
import pydicom
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from joblib import Parallel, delayed
from src.utils import rle_to_mask

class DataPreprocessor:
    """
    Handles the conversion of raw DICOM data to processed PNG format.
    Manages the split of training data into Train/Val and processes the existing Test set.
    """

    def __init__(self, config):
        """
        Initializes the preprocessor with configuration parameters.

        Args:
            config (dict): Configuration dictionary containing data paths and parameters.
        """
        # Paths
        self.raw_train_dir = config['data']['raw_train_dir']
        self.raw_test_dir = config['data']['raw_test_dir']
        self.raw_csv_path = config['data']['raw_csv_path']
        self.processed_dir = config['data']['processed_dir']
        
        # Params
        self.height = config['data']['img_height']
        self.width = config['data']['img_width']
        self.split_ratios = config['data']['split_ratios']
        self.random_state = config['data']['random_state']
        self.n_workers = config['data']['num_workers']

        # Maps: ImageId -> full DICOM file path
        self.train_path_map = {}
        self.test_path_map = {}

    def _build_path_maps(self):
        """
        Recursively finds all DICOM files for both train and test directories.
        Populates self.train_path_map and self.test_path_map.
        """
        print("Building file path maps...")
        
        # 1. Map Training Data
        train_files = glob.glob(os.path.join(self.raw_train_dir, '**', '*.dcm'), recursive=True)
        for path in train_files:
            image_id = os.path.splitext(os.path.basename(path))[0]
            self.train_path_map[image_id] = path
            
        # 2. Map Test Data
        test_files = glob.glob(os.path.join(self.raw_test_dir, '**', '*.dcm'), recursive=True)
        for path in test_files:
            image_id = os.path.splitext(os.path.basename(path))[0]
            self.test_path_map[image_id] = path
        
        print(f"Found {len(self.train_path_map)} Train DICOMs and {len(self.test_path_map)} Test DICOMs.")

    def _split_train_data(self, image_ids):
        """
        Splits available training image IDs into train and validation sets.

        Args:
            image_ids (list): List of unique ImageIds from the training set.

        Returns:
            dict: Dictionary with keys 'train', 'val'.
        """
        train_ids, val_ids = train_test_split(
            image_ids, 
            train_size=self.split_ratios['train'], 
            random_state=self.random_state
        )
        return {'train': train_ids, 'val': val_ids}

    def _process_single_image(self, image_id, split_name, path_map, rle_df=None):
        """
        Processes a single image.
        If rle_df is provided (Train/Val), it generates and saves a mask.
        If rle_df is None (Test), it skips mask generation.

        Args:
            image_id (str): Unique ID of the image.
            split_name (str): 'train', 'val', or 'test'.
            path_map (dict): The specific map to look up the file path.
            rle_df (pd.DataFrame, optional): DataFrame containing RLE info. Defaults to None.
        
        Returns:
            dict: Metadata record for the processed image.
        """
        try:
            # 1. Load DICOM
            dicom_path = path_map.get(image_id)
            if not dicom_path:
                return None
            
            dicom_data = pydicom.dcmread(dicom_path)
            image = dicom_data.pixel_array

            # 2. Resize Image
            img_resized = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)

            # 3. Handle Mask (Only for Train/Val)
            mask_save_path = None
            has_pneumothorax = -1 # -1 indicates unknown (for test)

            if rle_df is not None:
                # Get all RLE entries for this image
                rle_entries = rle_df[rle_df['ImageId'] == image_id][' EncodedPixels'].values
                
                orig_h, orig_w = image.shape
                final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

                has_pneumothorax = 0
                if len(rle_entries) > 0 and rle_entries[0] != '-1':
                    has_pneumothorax = 1
                    for rle in rle_entries:
                        if rle != '-1':
                            mask = rle_to_mask(rle, orig_h, orig_w)
                            final_mask = np.logical_or(final_mask, mask)
                    final_mask = final_mask.astype(np.uint8)

                # Resize Mask
                mask_resized = cv2.resize(final_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                
                # Save Mask
                mask_save_path = os.path.join(self.processed_dir, split_name, 'masks', f"{image_id}.png")
                os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
                # Scale 0/1 to 0/255 for PNG
                cv2.imwrite(mask_save_path, mask_resized * 255)

            # 4. Save Image
            img_save_path = os.path.join(self.processed_dir, split_name, 'images', f"{image_id}.png")
            os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
            cv2.imwrite(img_save_path, img_resized)

            return {
                'ImageId': image_id,
                'Split': split_name,
                'HasPneumothorax': has_pneumothorax, # 0, 1 or -1 (unknown)
                'ImagePath': img_save_path,
                'MaskPath': mask_save_path if mask_save_path else "" # Empty string if no mask
            }

        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            return None

    def run(self):
        """
        Executes the full preprocessing pipeline.
        """
        print("--- Starting Data Preprocessing ---")
        
        # 1. Prepare paths
        self._build_path_maps()
        
        # Load labels only for training data
        train_df = pd.read_csv(self.raw_csv_path)

        # 2. Process Training Data (Train + Val)
        available_train_ids = [uid for uid in train_df['ImageId'].unique() if uid in self.train_path_map]
        splits = self._split_train_data(available_train_ids)
        
        processed_metadata = []

        # Process Train and Val
        for split_name, ids in splits.items():
            print(f"Processing {split_name} set ({len(ids)} images)...")
            results = Parallel(n_jobs=self.n_workers, backend="threading")(
                delayed(self._process_single_image)(uid, split_name, self.train_path_map, train_df) 
                for uid in tqdm(ids)
            )
            processed_metadata.extend([r for r in results if r is not None])

        # 3. Process Test Data (Existing Test Folder)
        # Note: We pass rle_df=None because we don't have masks for test
        test_ids = list(self.test_path_map.keys())
        print(f"Processing test set ({len(test_ids)} images)...")
        
        results_test = Parallel(n_jobs=self.n_workers, backend="threading")(
            delayed(self._process_single_image)(uid, 'test', self.test_path_map, None) 
            for uid in tqdm(test_ids)
        )
        processed_metadata.extend([r for r in results_test if r is not None])

        # 4. Save Metadata CSV
        meta_df = pd.DataFrame(processed_metadata)
        meta_save_path = os.path.join(self.processed_dir, "metadata_processed.csv")
        os.makedirs(self.processed_dir, exist_ok=True)
        meta_df.to_csv(meta_save_path, index=False)
        
        print(f"--- Preprocessing Complete. Metadata saved to {meta_save_path} ---")
        print(f"Total processed: {len(meta_df)} records.")