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
from src.model import MultiTaskUNet
from src.loss import CombinedLoss
from src.trainer import Trainer

def validate_env():
    print("[INFO] Validating environment...")
    req_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
    setup = EnvironmentSetup(req_path)
    setup.smart_install()
    setup.check_cuda()

def main():
    # 1. Environment
    validate_env()
    
    # 2. Config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")

    # 3. Data Preparation
    data_dir = config['data']['dataset_dir']
    csv_path = os.path.join(data_dir, "metadata_processed.csv")
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] Metadata not found at {csv_path}. Did you attach the dataset?")
        return

    df = pd.read_csv(csv_path)
    train_df = df[df['Split'] == 'train'].reset_index(drop=True)
    val_df = df[df['Split'] == 'val'].reset_index(drop=True)
    
    print(f"[INFO] Train set: {len(train_df)} | Val set: {len(val_df)}")

    train_ds = PneumoDataset(train_df, data_dir, phase='train', config=config)
    val_ds = PneumoDataset(val_df, data_dir, phase='val', config=config)

    train_loader = DataLoader(
        train_ds, 
        batch_size=config['train']['batch_size'], 
        shuffle=True, 
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config['train']['batch_size'], 
        shuffle=False, 
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    # 4. Model & Optimization
    model = MultiTaskUNet(
        encoder_name=config['train']['encoder_name'],
        pretrained=config['train']['pretrained']
    ).to(device)
    
    loss_fn = CombinedLoss(config).to(device)
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
    
    trainer.fit()

if __name__ == "__main__":
    main()