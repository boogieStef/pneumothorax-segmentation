import os
import glob
import pandas as pd
import numpy as np
import pydicom
import cv2
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from src.utils import rle_to_mask

class DataPreprocessor:
    """
    Manages the preprocessing pipeline.
    Splits labeled data into Train/Val/Test.
    Ignores external unlabeled data.
    """

    def __init__(self, config):
        self.raw_train_dir = config['data']['raw_train_dir']
        self.raw_csv_path = config['data']['raw_csv_path']
        self.processed_dir = config['data']['processed_dir']
        
        self.height = config['data']['img_height']
        self.width = config['data']['img_width']
        self.split_ratios = config['data']['split_ratios']
        self.random_state = config['data']['random_state']
        self.n_workers = config['data']['num_workers']

        self.labeled_path_map = {}

    def _build_path_maps(self):
        """
        Scans directories to build map of ImageId to DICOM file paths.
        """
        print("[INFO] Building file path map...")
        
        # Map Labeled Data (Original Train folder)
        train_files = glob.glob(os.path.join(self.raw_train_dir, '**', '*.dcm'), recursive=True)
        for path in train_files:
            image_id = os.path.splitext(os.path.basename(path))[0]
            self.labeled_path_map[image_id] = path
            
        print(f"[INFO] Found {len(self.labeled_path_map)} Labeled DICOMs.")

    def _split_labeled_data(self, image_ids):
        """
        Splits labeled image IDs into Train, Val, and Internal Test sets.
        """
        r_train = self.split_ratios['train']
        r_val = self.split_ratios['val']
        # r_test is the remainder
        
        # Normalize ratios just in case
        total = r_train + r_val + self.split_ratios['test']
        r_train /= total
        r_val /= total
        
        # First Split: Train vs (Val + Test)
        train_ids, temp_ids = train_test_split(
            image_ids, 
            train_size=r_train, 
            random_state=self.random_state
        )
        
        # Second Split: Val vs Test
        # Adjust val ratio relative to the remaining temp_ids size
        val_ratio_adjusted = r_val / (1.0 - r_train)
        
        val_ids, test_ids = train_test_split(
            temp_ids,
            train_size=val_ratio_adjusted,
            random_state=self.random_state
        )
        
        return {'train': train_ids, 'val': val_ids, 'test': test_ids}

    def _process_single_image(self, image_id, split_name, path_map, rle_df):
        """
        Processes a single image and generates a mask.
        """
        try:
            # 1. Load DICOM
            dicom_path = path_map.get(image_id)
            if not dicom_path:
                return f"Path not found for {image_id}"
            
            dicom_data = pydicom.dcmread(dicom_path)
            image = dicom_data.pixel_array

            # 2. Resize Image
            img_resized = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)

            # 3. Handle Mask (Always required for this pipeline)
            # Get RLEs for this image
            rle_entries = rle_df[rle_df['ImageId'] == image_id][' EncodedPixels'].values
            
            orig_h, orig_w = image.shape
            final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            
            has_pneumothorax = 0

            # Check if mask exists
            if len(rle_entries) > 0:
                first_rle = str(rle_entries[0]).strip()
                if first_rle != '-1':
                    has_pneumothorax = 1
                    for rle in rle_entries:
                        clean_rle = str(rle).strip()
                        if clean_rle != '-1':
                            mask = rle_to_mask(clean_rle, orig_h, orig_w)
                            final_mask = np.logical_or(final_mask, mask)
                    final_mask = final_mask.astype(np.uint8)

            # Resize Mask
            mask_resized = cv2.resize(final_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            
            # Save Mask
            mask_save_path = os.path.join(self.processed_dir, split_name, 'masks', f"{image_id}.png")
            cv2.imwrite(mask_save_path, mask_resized * 255)

            # 4. Save Image
            img_save_path = os.path.join(self.processed_dir, split_name, 'images', f"{image_id}.png")
            cv2.imwrite(img_save_path, img_resized)

            return {
                'ImageId': image_id,
                'Split': split_name,
                'HasPneumothorax': has_pneumothorax,
                'ImagePath': img_save_path,
                'MaskPath': mask_save_path
            }

        except Exception as e:
            return f"ERROR processing {image_id}: {str(e)}"

    def _process_dataset_split(self, id_list, split_name, path_map, rle_df):
        """
        Processes a list of IDs. First 3 sequentially, rest in parallel.
        """
        total_items = len(id_list)
        if total_items == 0:
            return []

        print(f"[INFO] Processing {split_name} set ({total_items} images)...")
        
        # Sanity Check (Sequential)
        check_count = min(3, total_items)
        sanity_batch = id_list[:check_count]
        bulk_batch = id_list[check_count:]
        
        results = []
        
        for i, uid in enumerate(sanity_batch):
            res = self._process_single_image(uid, split_name, path_map, rle_df)
            if isinstance(res, dict):
                print(f"[CHECK {i+1}/{check_count}] SUCCESS: {uid}")
                results.append(res)
            else:
                print(f"[CHECK {i+1}/{check_count}] FAILED: {uid}. Reason: {res}")

        # Bulk Processing (Parallel)
        if bulk_batch:
            print(f"[INFO] Starting bulk processing for remaining {len(bulk_batch)} images...")
            bulk_results = Parallel(n_jobs=self.n_workers, verbose=0)(
                delayed(self._process_single_image)(uid, split_name, path_map, rle_df) 
                for uid in bulk_batch
            )
            valid_bulk = [r for r in bulk_results if isinstance(r, dict)]
            results.extend(valid_bulk)
            
        print(f"[INFO] Finished {split_name} set. Valid: {len(results)}/{total_items}")
        return results

    def run(self):
        """
        Executes the preprocessing pipeline.
        """
        print("[INFO] --- Starting Data Preprocessing (Internal Splits Only) ---")
        
        self._build_path_maps()
        
        # Create directories
        for split_name in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.processed_dir, split_name, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.processed_dir, split_name, 'masks'), exist_ok=True)

        # Load Labels
        train_df = pd.read_csv(self.raw_csv_path)

        # Split Labeled Data
        available_ids = [uid for uid in train_df['ImageId'].unique() if uid in self.labeled_path_map]
        splits = self._split_labeled_data(available_ids)
        
        processed_metadata = []

        # Process Labeled Splits (Train, Val, Test)
        for split_name, ids in splits.items():
            processed_metadata.extend(
                self._process_dataset_split(ids, split_name, self.labeled_path_map, train_df)
            )

        # Save Metadata
        meta_df = pd.DataFrame(processed_metadata)
        meta_save_path = os.path.join(self.processed_dir, "metadata_processed.csv")
        meta_df.to_csv(meta_save_path, index=False)
        
        print(f"[INFO] --- Preprocessing Complete. Metadata saved to {meta_save_path} ---")
        print(f"[INFO] Split Statistics:\n{meta_df['Split'].value_counts()}")