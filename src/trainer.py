import torch
import numpy as np
from tqdm import tqdm

class Trainer:
    """
    Manages training, validation, and testing loops.
    Tracks metrics: Loss, Dice Score, and IoU.
    Pure segmentation version (no classification metrics).
    """
    def __init__(self, model, loaders, optimizer, loss_fn, device, config):
        self.model = model
        self.train_loader = loaders['train']
        self.val_loader = loaders['val']
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        
        # Tracking best performance
        self.best_score = 0.0
        self.best_metrics = {} 

    def fit(self):
        """
        Main training loop.
        """
        epochs = self.config['train']['epochs']
        exp_name = self.config['train']['experiment_name']
        print(f"[INFO] Starting training experiment: {exp_name}")
        print(f"[INFO] Total epochs: {epochs}")
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            
            # 1. Training Phase
            train_metrics = self._train_epoch()
            print(f"TRAIN | Loss: {train_metrics['loss']:.4f}")
            
            # 2. Validation Phase
            val_metrics = self._validate_epoch(self.val_loader)
            print(f"VALID | Loss: {val_metrics['loss']:.4f} | Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}")
            
            # 3. Checkpointing
            if val_metrics['dice'] > self.best_score:
                print(f"[SAVE] New Best Model! Dice improved: {self.best_score:.4f} -> {val_metrics['dice']:.4f}")
                self.best_score = val_metrics['dice']
                self.best_metrics = val_metrics
                self._save_checkpoint()
            else:
                print(f"[INFO] Dice did not improve (Best: {self.best_score:.4f})")

        # End of training summary
        print("\n" + "="*40)
        print("       TRAINING COMPLETE")
        print("="*40)
        print(f"Best Validation Dice: {self.best_metrics.get('dice', 0):.4f}")
        print(f"Best Validation IoU:  {self.best_metrics.get('iou', 0):.4f}")
        print(f"Best Validation Loss: {self.best_metrics.get('loss', 0):.4f}")
        print(f"Best Model saved to:  best_model.pth")
        print("="*40 + "\n")

    def evaluate(self, test_loader):
        """
        Loads the best model weights and evaluates on the Test set.
        """
        print("[INFO] Loading best model for testing...")
        try:
            self.model.load_state_dict(torch.load("best_model.pth"))
            print("[INFO] Weights loaded successfully.")
        except FileNotFoundError:
            print("[WARNING] 'best_model.pth' not found! Testing with current weights.")

        print("\n--- Starting Evaluation on TEST Set ---")
        test_metrics = self._validate_epoch(test_loader)
        
        print("\n" + "="*40)
        print("       TEST SET RESULTS")
        print("="*40)
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Dice: {test_metrics['dice']:.4f}")
        print(f"Test IoU:  {test_metrics['iou']:.4f}")
        print("="*40 + "\n")
        
        return test_metrics

    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        # 'labels' are ignored in pure segmentation
        for images, masks, _ in self.train_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.loss_fn(logits, masks)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
        return {"loss": running_loss / len(self.train_loader)}

    def _validate_epoch(self, loader):
        self.model.eval()
        running_loss = 0.0
        dice_scores = []
        iou_scores = []
        
        with torch.no_grad():
            for images, masks, _ in loader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                logits = self.model(images)
                loss = self.loss_fn(logits, masks)
                running_loss += loss.item()
                
                # Metrics Calculation
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                dice = self._dice_coef(preds, masks)
                iou = self._iou_coef(preds, masks)
                
                dice_scores.append(dice)
                iou_scores.append(iou)
                
        return {
            "loss": running_loss / len(loader),
            "dice": np.mean(dice_scores),
            "iou": np.mean(iou_scores)
        }

    def _dice_coef(self, preds, targets, smooth=1e-6):
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        return ((2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)).item()

    def _iou_coef(self, preds, targets, smooth=1e-6):
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection
        return ((intersection + smooth) / (union + smooth)).item()

    def _save_checkpoint(self):
        torch.save(self.model.state_dict(), "best_model.pth")