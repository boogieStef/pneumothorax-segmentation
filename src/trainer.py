import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

class Trainer:
    """
    Manages training, validation, and testing loops.
    Tracks metrics: Loss, Dice Score, IoU, Sensitivity, Specificity.
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
        
        # --- BRAKUJĄCY ELEMENT (TENSORBOARD) ---
        # Musimy zainicjować writera, żeby móc zapisywać wykresy
        log_dir = os.path.join("runs", config['train']['experiment_name'])
        self.writer = SummaryWriter(log_dir=log_dir)

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
            self.writer.add_scalar("Loss/Train", train_metrics['loss'], epoch)
            
            # 2. Validation
            val_metrics = self._validate_epoch(self.val_loader)
            print(f"VALID | Loss: {val_metrics['loss']:.4f} | Dice: {val_metrics['dice']:.4f}")
            print(f"      | Sens: {val_metrics['sensitivity']:.4f} | Spec: {val_metrics['specificity']:.4f} | Acc: {val_metrics['accuracy']:.4f}")
            
            # Logowanie metryk walidacyjnych
            self.writer.add_scalar("Loss/Validation", val_metrics['loss'], epoch)
            self.writer.add_scalar("Metric/Dice", val_metrics['dice'], epoch)
            self.writer.add_scalar("Metric/IoU", val_metrics['iou'], epoch)
            self.writer.add_scalar("Metric/Sensitivity", val_metrics['sensitivity'], epoch)
            self.writer.add_scalar("Metric/Specificity", val_metrics['specificity'], epoch)
            
            # 3. Checkpointing
            if val_metrics['dice'] > self.best_score:
                print(f"[SAVE] New Best Model! Dice improved: {self.best_score:.4f} -> {val_metrics['dice']:.4f}")
                self.best_score = val_metrics['dice']
                self.best_metrics = val_metrics
                self._save_checkpoint()
            else:
                print(f"[INFO] Dice did not improve (Best: {self.best_score:.4f})")
        
        # Zamknięcie writera po zakończeniu treningu
        self.writer.close()

        # End of training summary
        print("\n" + "="*40)
        print("       TRAINING COMPLETE")
        print("="*40)
        print(f"Best Validation Dice: {self.best_metrics.get('dice', 0):.4f}")
        print(f"Best Validation IoU:  {self.best_metrics.get('iou', 0):.4f}")
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
        print("-" * 20)
        print(f"Sensitivity (Recall): {test_metrics['sensitivity']:.4f}")
        print(f"Specificity:          {test_metrics['specificity']:.4f}")
        print(f"Classification Acc:   {test_metrics['accuracy']:.4f}")
        print("="*40 + "\n")
        
        return test_metrics

    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        for images, masks, _ in self.train_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.loss_fn(logits, masks)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
        return {"loss": running_loss / len(self.train_loader)}

    def _save_checkpoint(self):
        torch.save(self.model.state_dict(), "best_model.pth")

    def _validate_epoch(self, loader):
        self.model.eval()
        running_loss = 0.0
        
        dice_scores = []
        iou_scores = []
        
        tp_total = 0
        tn_total = 0
        fp_total = 0
        fn_total = 0
        
        with torch.no_grad():
            for images, masks, _ in loader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                logits = self.model(images)
                loss = self.loss_fn(logits, masks)
                running_loss += loss.item()
                
                # --- Segmentacja ---
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                d, i = self._calculate_metrics_sample_wise(preds, masks)
                dice_scores.extend(d)
                iou_scores.extend(i)
                
                # --- Detekcja (Klasyfikacja) ---
                batch_cls_stats = self._calculate_classification_stats(preds, masks)
                tp_total += batch_cls_stats['tp']
                tn_total += batch_cls_stats['tn']
                fp_total += batch_cls_stats['fp']
                fn_total += batch_cls_stats['fn']

        mean_dice = np.mean(dice_scores)
        mean_iou = np.mean(iou_scores)
        
        epsilon = 1e-6
        sensitivity = tp_total / (tp_total + fn_total + epsilon)
        specificity = tn_total / (tn_total + fp_total + epsilon)
        accuracy    = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total + epsilon)
        
        return {
            "loss": running_loss / len(loader),
            "dice": mean_dice,
            "iou": mean_iou,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "accuracy": accuracy
        }

    def _calculate_classification_stats(self, preds, targets):
        pixel_thresh = self.config['train'].get('pixel_threshold', 0)
        batch_size = preds.shape[0]
        
        preds_flat = preds.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)
        
        pred_areas = preds_flat.sum(1)
        target_areas = targets_flat.sum(1)
        
        is_positive_pred = (pred_areas > pixel_thresh).float()
        is_positive_true = (target_areas > 0).float()
        
        tp = ((is_positive_pred == 1) & (is_positive_true == 1)).sum().item()
        tn = ((is_positive_pred == 0) & (is_positive_true == 0)).sum().item()
        fp = ((is_positive_pred == 1) & (is_positive_true == 0)).sum().item()
        fn = ((is_positive_pred == 0) & (is_positive_true == 1)).sum().item()
        
        return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

    def _calculate_metrics_sample_wise(self, preds, targets, smooth=1e-6):
        batch_size = preds.shape[0]
        
        preds_flat = preds.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)
        
        intersection = (preds_flat * targets_flat).sum(1)
        
        p_sum = preds_flat.sum(1)
        t_sum = targets_flat.sum(1)
        
        dices = (2. * intersection + smooth) / (p_sum + t_sum + smooth)
        
        union = p_sum + t_sum - intersection
        ious = (intersection + smooth) / (union + smooth)
        
        return dices.cpu().numpy().tolist(), ious.cpu().numpy().tolist()