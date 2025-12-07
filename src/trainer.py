import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class Trainer:
    """
    Manages the training and validation loops.
    """
    def __init__(self, model, loaders, optimizer, loss_fn, device, config):
        self.model = model
        self.train_loader = loaders['train']
        self.val_loader = loaders['val']
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.best_score = 0.0

    def fit(self):
        """
        Main training loop over epochs.
        """
        epochs = self.config['train']['epochs']
        print(f"[INFO] Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            
            # Train
            train_metrics = self._train_epoch()
            print(f"Train Loss: {train_metrics['loss']:.4f} | Seg: {train_metrics['seg_loss']:.4f} | Cls: {train_metrics['cls_loss']:.4f}")
            
            # Validate
            val_metrics = self._validate_epoch()
            print(f"Val Loss: {val_metrics['loss']:.4f} | Dice: {val_metrics['dice']:.4f} | Acc: {val_metrics['acc']:.4f}")
            
            # Checkpoint
            self._save_checkpoint(val_metrics['dice'], epoch)

    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0
        running_seg = 0.0
        running_cls = 0.0
        
        # Use simple iterator to avoid excessive logging in Kaggle save version
        for images, masks, labels in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward
            seg_logits, cls_logits = self.model(images)
            
            # Calculate Loss
            loss, seg_loss, cls_loss = self.loss_fn(seg_logits, masks, cls_logits, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            running_seg += seg_loss.item()
            running_cls += cls_loss.item()
            
        n = len(self.train_loader)
        return {
            "loss": running_loss / n,
            "seg_loss": running_seg / n,
            "cls_loss": running_cls / n
        }

    def _validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        dice_scores = []
        cls_preds = []
        cls_targets = []
        
        with torch.no_grad():
            for images, masks, labels in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)
                
                seg_logits, cls_logits = self.model(images)
                
                # Loss
                loss, _, _ = self.loss_fn(seg_logits, masks, cls_logits, labels)
                running_loss += loss.item()
                
                # Metrics: Dice (Segmentation)
                probs = torch.sigmoid(seg_logits)
                preds = (probs > 0.5).float()
                dice = self._calculate_dice(preds, masks)
                dice_scores.append(dice)
                
                # Metrics: Accuracy (Classification)
                cls_probs = torch.sigmoid(cls_logits)
                cls_p = (cls_probs > 0.5).float()
                cls_preds.extend(cls_p.cpu().numpy())
                cls_targets.extend(labels.cpu().numpy())
                
        avg_loss = running_loss / len(self.val_loader)
        avg_dice = np.mean(dice_scores)
        acc = accuracy_score(cls_targets, cls_preds)
        
        return {"loss": avg_loss, "dice": avg_dice, "acc": acc}

    def _calculate_dice(self, preds, targets, smooth=1e-6):
        # Flatten
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
        return dice.item()

    def _save_checkpoint(self, current_score, epoch):
        if current_score > self.best_score:
            print(f"[SAVE] Score improved ({self.best_score:.4f} -> {current_score:.4f}). Saving model...")
            self.best_score = current_score
            save_path = "best_model.pth"
            torch.save(self.model.state_dict(), save_path)