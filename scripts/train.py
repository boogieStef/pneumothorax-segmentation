import argparse
import sys
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment import EnvironmentSetup
from src.utils import load_config

# NOTE: We do NOT import src.model, src.trainer, or src.dataset here yet.
# They depend on libraries (like 'smp') that might not be installed yet.
# We import them inside main() after environment validation.

def main():
    # --- 1. Environment Validation & Setup ---
    # This must run BEFORE importing any custom modules that rely on external libs
    print("[INFO] Setting up environment...")
    setup = EnvironmentSetup()
    
    # Install dependencies (surgical)
    setup.validate_and_install_smp()
    
    # Check GPU stability
    setup.validate_gpu_forward_pass()
    
    # --- 2. Lazy Imports ---
    # Now it is safe to import modules requiring 'segmentation_models_pytorch'
    from src.dataset import PneumoDataset
    from src.model import ModelFactory
    from src.loss import SegmentationLoss
    from src.trainer import Trainer

    # --- 3. Config Loading ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = config['train'].get('architecture', 'Unknown')
    print(f"[INFO] Device: {device} | Model Architecture: {arch}")

    # --- 4. Data Preparation ---
    data_dir = config['data']['dataset_dir']
    csv_path = os.path.join(data_dir, "metadata_processed.csv")
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] Metadata not found at {csv_path}. Check your dataset path.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    
    # Filter by split
    train_df = df[df['Split'] == 'train'].reset_index(drop=True)
    val_df = df[df['Split'] == 'val'].reset_index(drop=True)
    test_df = df[df['Split'] == 'test'].reset_index(drop=True)
    
    print(f"[INFO] Dataset Loaded. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Create Datasets
    train_ds = PneumoDataset(train_df, data_dir, phase='train', config=config)
    val_ds = PneumoDataset(val_df, data_dir, phase='val', config=config)
    test_ds = PneumoDataset(test_df, data_dir, phase='test', config=config)

    # Create Loaders
    num_workers = config['data']['num_workers']
    batch_size = config['train']['batch_size']
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # --- 5. Model & Optimization ---
    model = ModelFactory.create(config).to(device)
    loss_fn = SegmentationLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

    # --- 6. Trainer ---
    trainer = Trainer(
        model=model,
        loaders={'train': train_loader, 'val': val_loader},
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        config=config
    )
    
    # Run Training
    trainer.fit()
    
    # Run Evaluation
    trainer.evaluate(test_loader)

if __name__ == "__main__":
    main()