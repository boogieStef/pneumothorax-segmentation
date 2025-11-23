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
    Manages the preprocessing pipeline using a hybrid approach:
    1. Sequential sanity check for the first few images (verification).
    2. Parallel bulk processing for the rest (efficiency).
    """

    def __init__(self, config):
        self.raw_train_dir = config['data']['raw_train_dir']
        self.raw_test_dir = config['data']['raw_test_dir']
        self.raw_csv_path = config['data']['raw_csv_path']
        self.processed_dir = config['data']['processed_dir']
        
        self.height = config['data']['img_height']
        self.width = config['data']['img_width']
        self.split_ratios = config['data']['split_ratios']
        self.random_state = config['data']['random_state']
        self.n_workers = config['data']['num_workers']

        self.train_path_map = {}
        self.test_path_map = {}

    def _build_path_maps(self):
        """
        Scans directories to build maps of ImageId to DICOM file paths.
        """
        print("[INFO] Building file path maps...")
        
        train_files = glob.glob(os.path.join(self.raw_train_dir, '**', '*.dcm'), recursive=True)
        for path in train_files:
            image_id = os.path.splitext(os.path.basename(path))[0]
            self.train_path_map[image_id] = path
            
        test_files = glob.glob(os.path.join(self.raw_test_dir, '**', '*.dcm'), recursive=True)
        for path in test_files:
            image_id = os.path.splitext(os.path.basename(path))[0]
            self.test_path_map[image_id] = path
        
        print(f"[INFO] Found {len(self.train_path_map)} Train DICOMs and {len(self.test_path_map)} Test DICOMs.")

    def _process_single_image(self, image_id, split_name, path_map, rle_df=None):
        """
        Processes a single image: reads DICOM, resizes, handles mask, and saves as PNG.
        Designed to be run in parallel.
        """
        try:
            # 1. Load DICOM
            dicom_path = path_map.get(image_id)
            if not dicom_path:
                return None
            
            dicom_data = pydicom.dcmread(dicom_path)
            image = dicom_data.pixel_array

            # 2. Resize Image (INTER_AREA is better for shrinking)
            img_resized = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)

            # 3. Handle Mask (Only if RLE data is provided)
            mask_save_path = ""
            has_pneumothorax = -1 # Default for unknown/test

            if rle_df is not None:
                # Retrieve RLE strings
                rle_entries = rle_df[rle_df['ImageId'] == image_id][' EncodedPixels'].values
                
                orig_h, orig_w = image.shape
                final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                
                has_pneumothorax = 0

                # FIX: Add .strip() to handle leading spaces in CSV (e.g. ' -1')
                if len(rle_entries) > 0:
                    first_rle = str(rle_entries[0]).strip()
                    
                    if first_rle != '-1':
                        has_pneumothorax = 1
                        for rle in rle_entries:
                            clean_rle = str(rle).strip()
                            if clean_rle != '-1':
                                mask = rle_to_mask(clean_rle, orig_h, orig_w)
                                # Combine masks using logical OR
                                final_mask = np.logical_or(final_mask, mask)
                        final_mask = final_mask.astype(np.uint8)

                # Resize Mask (INTER_NEAREST to keep binary values 0/1)
                mask_resized = cv2.resize(final_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                
                # Save Mask
                mask_save_path = os.path.join(self.processed_dir, split_name, 'masks', f"{image_id}.png")
                # Scale 0/1 to 0/255 for PNG format visibility
                cv2.imwrite(mask_save_path, mask_resized * 255)

            # 4. Save Image
            img_save_path = os.path.join(self.processed_dir, split_name, 'images', f"{image_id}.png")
            cv2.imwrite(img_save_path, img_resized)

            return {
                'ImageId': image_id,
                'Split': split_name,
                'HasPneumothorax': has_pneumothorax,
                'ImagePath': img_save_path,
                'MaskPath': mask_save_path if mask_save_path else ""
            }

        except Exception as e:
            # Log the specific error and ID but do not crash the pipeline
            return f"ERROR processing {image_id}: {str(e)}"

    def _process_dataset_split(self, id_list, split_name, path_map, rle_df=None):
        """
        Processes a list of IDs with a sanity check on the first few items.
        
        Args:
            id_list (list): List of ImageIds.
            split_name (str): 'train', 'val', or 'test'.
            path_map (dict): Map of IDs to paths.
            rle_df (DataFrame): DataFrame with masks (optional).
            
        Returns:
            list: Valid metadata records.
        """
        total_items = len(id_list)
        if total_items == 0:
            print(f"[WARNING] No items to process for {split_name}.")
            return []

        print(f"[INFO] Processing {split_name} set ({total_items} images)...")
        
        results = []
        
        # --- 1. Sanity Check (Sequential) ---
        # We pick the first 3 images to verify the pipeline works.
        check_count = min(3, total_items)
        sanity_batch = id_list[:check_count]
        bulk_batch = id_list[check_count:]
        
        print(f"--- Sanity Check: Verifying first {check_count} images... ---")
        
        for i, uid in enumerate(sanity_batch):
            res = self._process_single_image(uid, split_name, path_map, rle_df)
            
            if isinstance(res, dict):
                print(f"[CHECK {i+1}/{check_count}] SUCCESS: {uid} -> Saved to {res['ImagePath']}")
                results.append(res)
            elif isinstance(res, str):
                # It's an error message
                print(f"[CHECK {i+1}/{check_count}] FAILED: {uid}. Reason: {res}")
            else:
                print(f"[CHECK {i+1}/{check_count}] FAILED: {uid}. Unknown error.")

        # --- 2. Bulk Processing (Parallel) ---
        if bulk_batch:
            print(f"--- Starting bulk processing for remaining {len(bulk_batch)} images... ---")
            
            # Use joblib for efficiency. verbose=0 keeps logs clean.
            bulk_results = Parallel(n_jobs=self.n_workers, verbose=0)(
                delayed(self._process_single_image)(uid, split_name, path_map, rle_df) 
                for uid in bulk_batch
            )
            
            # Filter valid results (dicts) and log errors
            valid_bulk = []
            error_count = 0
            
            for res in bulk_results:
                if isinstance(res, dict):
                    valid_bulk.append(res)
                else:
                    error_count += 1
            
            results.extend(valid_bulk)
            print(f"--- Bulk processing complete. Errors encountered: {error_count} ---")

        print(f"[INFO] Finished {split_name} set. Total valid: {len(results)}/{total_items}")
        return results

    def run(self):
        """
        Executes the pipeline.
        """
        print("[INFO] --- Starting Data Preprocessing ---")
        
        self._build_path_maps()
        
        # Create directories
        for split_name in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.processed_dir, split_name, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.processed_dir, split_name, 'masks'), exist_ok=True)

        train_df = pd.read_csv(self.raw_csv_path)

        # Split Data
        available_train_ids = [uid for uid in train_df['ImageId'].unique() if uid in self.train_path_map]
        
        train_ids, val_ids = train_test_split(
            available_train_ids, 
            train_size=self.split_ratios['train'], 
            random_state=self.random_state
        )
        
        processed_metadata = []

        # 1. Process Train
        processed_metadata.extend(
            self._process_dataset_split(train_ids, "train", self.train_path_map, train_df)
        )
        
        # 2. Process Val
        processed_metadata.extend(
            self._process_dataset_split(val_ids, "val", self.train_path_map, train_df)
        )

        # 3. Process Test
        test_ids = list(self.test_path_map.keys())
        processed_metadata.extend(
            self._process_dataset_split(test_ids, "test", self.test_path_map, None)
        )

        # 4. Save Metadata
        meta_df = pd.DataFrame(processed_metadata)
        meta_save_path = os.path.join(self.processed_dir, "metadata_processed.csv")
        meta_df.to_csv(meta_save_path, index=False)
        
        print(f"[INFO] --- Preprocessing Complete. Metadata saved to {meta_save_path} ---")
        print(f"[INFO] Total records: {len(meta_df)}")