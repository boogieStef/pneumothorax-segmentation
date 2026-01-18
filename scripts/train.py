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
from src.dataset import PneumoDataset
# --- FIX: Changed from MultiTaskUNet to ModelFactory ---
from src.model import ModelFactory 
from src.loss import SegmentationLoss
from src.trainer import Trainer

def main():
    # 1. Environment
    setup = EnvironmentSetup()
    setup.validate_and_install_smp() # Instalacja bibliotek
    setup.validate_gpu_forward_pass() # Test "Kernel Dead"
    
    # 2. Config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Check if 'architecture' key exists, fallback for older configs if needed
    arch = config['train'].get('architecture', 'Unknown')
    print(f"[INFO] Device: {device} | Model Architecture: {arch}")

    # 3. Data Preparation
    data_dir = config['data']['dataset_dir']
    csv_path = os.path.join(data_dir, "metadata_processed.csv")
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] Metadata not found at {csv_path}. Did you attach the dataset?")
        return

    df = pd.read_csv(csv_path)
    
    # Split DataFrames based on 'Split' column
    train_df = df[df['Split'] == 'train'].reset_index(drop=True)
    val_df = df[df['Split'] == 'val'].reset_index(drop=True)
    test_df = df[df['Split'] == 'test'].reset_index(drop=True)
    
    print(f"[INFO] Dataset Loaded.")
    print(f"       Train: {len(train_df)} samples")
    print(f"       Val:   {len(val_df)} samples")
    print(f"       Test:  {len(test_df)} samples")

    # Create Datasets
    # Train has augmentation enabled via phase='train'
    train_ds = PneumoDataset(train_df, data_dir, phase='train', config=config)
    val_ds = PneumoDataset(val_df, data_dir, phase='val', config=config)
    test_ds = PneumoDataset(test_df, data_dir, phase='test', config=config)

    # Create Loaders
    num_workers = config['data']['num_workers']
    batch_size = config['train']['batch_size']
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 4. Model & Optimization
    # --- FIX: Use ModelFactory to create the model ---
    model = ModelFactory.create(config).to(device)
    
    loss_fn = SegmentationLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

    # 5. Trainer
    trainer = Trainer(
        model=model,
        loaders={'train': train_loader, 'val': val_loader},
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        config=config
    )
    
    # Run Training Loop
    trainer.fit()
    
    # Run Final Evaluation on Test Set
    trainer.evaluate(test_loader)

if __name__ == "__main__":
    main()