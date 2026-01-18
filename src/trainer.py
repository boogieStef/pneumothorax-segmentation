import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

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
            
            # 2. Validation
            val_metrics = self._validate_epoch(self.val_loader)
            print(f"VALID | Loss: {val_metrics['loss']:.4f} | Dice: {val_metrics['dice']:.4f}")
            # Dodaj nową linię:
            print(f"      | Sens: {val_metrics['sensitivity']:.4f} | Spec: {val_metrics['specificity']:.4f} | Acc: {val_metrics['accuracy']:.4f}")
            
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
        print("-" * 20)
        print(f"Sensitivity (Recall): {test_metrics['sensitivity']:.4f}")
        print(f"Specificity:          {test_metrics['specificity']:.4f}")
        print(f"Classification Acc:   {test_metrics['accuracy']:.4f}")
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

    def _save_checkpoint(self):
        torch.save(self.model.state_dict(), "best_model.pth")

    def _validate_epoch(self, loader):
        self.model.eval()
        running_loss = 0.0
        
        # Listy na wyniki segmentacji
        dice_scores = []
        iou_scores = []
        
        # Liczniki do klasyfikacji (Detekcja)
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
                
                # Liczymy Dice/IoU per sample (tak jak ustaliliśmy)
                d, i = self._calculate_metrics_sample_wise(preds, masks)
                dice_scores.extend(d)
                iou_scores.extend(i)
                
                # --- Detekcja (Klasyfikacja) ---
                # Wyciągamy statystyki TP, TN, FP, FN dla tego batcha
                batch_cls_stats = self._calculate_classification_stats(preds, masks)
                tp_total += batch_cls_stats['tp']
                tn_total += batch_cls_stats['tn']
                fp_total += batch_cls_stats['fp']
                fn_total += batch_cls_stats['fn']

        # Obliczamy średnie metryki segmentacji
        mean_dice = np.mean(dice_scores)
        mean_iou = np.mean(iou_scores)
        
        # Obliczamy metryki klasyfikacji (zabezpieczenie przed dzieleniem przez 0)
        epsilon = 1e-6
        sensitivity = tp_total / (tp_total + fn_total + epsilon) # Recall
        specificity = tn_total / (tn_total + fp_total + epsilon)
        accuracy    = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total + epsilon)
        
        return {
            "loss": running_loss / len(loader),
            "dice": mean_dice,
            "iou": mean_iou,
            # Nowe metryki
            "sensitivity": sensitivity,
            "specificity": specificity,
            "accuracy": accuracy
        }

    def _calculate_classification_stats(self, preds, targets):
        """
        Ocenia zdolność detekcji modelu (Czy wykrył odmę, czy nie?).
        Używa progu powierzchni (pixel_threshold) z configu.
        """
        # Pobieramy próg z konfigu (np. 100 pikseli), domyślnie 0
        pixel_thresh = self.config['train'].get('pixel_threshold', 0)
        
        batch_size = preds.shape[0]
        
        # Spłaszczamy obrazy do wektora (Batch, Pixels)
        preds_flat = preds.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)
        
        # Zliczamy zapalone piksele dla każdego obrazka
        pred_areas = preds_flat.sum(1)   # Ile pikseli przewidział model
        target_areas = targets_flat.sum(1) # Ile pikseli jest w prawdzie
        
        # Klasyfikacja: 1 (Chory) jeśli obszar > próg, w przeciwnym razie 0 (Zdrowy)
        # Dla Targetu: Chory jeśli ma > 0 pikseli (zakładamy że maska ground truth jest czysta)
        is_positive_pred = (pred_areas > pixel_thresh).float()
        is_positive_true = (target_areas > 0).float()
        
        # Obliczamy TP, TN, FP, FN
        tp = ((is_positive_pred == 1) & (is_positive_true == 1)).sum().item()
        tn = ((is_positive_pred == 0) & (is_positive_true == 0)).sum().item()
        fp = ((is_positive_pred == 1) & (is_positive_true == 0)).sum().item()
        fn = ((is_positive_pred == 0) & (is_positive_true == 1)).sum().item()
        
        return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

    # ... (Metoda _calculate_metrics_sample_wise pozostaje bez zmian z poprzedniej odpowiedzi) ...

    def _calculate_metrics_sample_wise(self, preds, targets, smooth=1e-6):
        """
        Calculates Dice and IoU for each image in the batch separately.
        Returns lists of scores.
        """
        batch_size = preds.shape[0]
        dice_list = []
        iou_list = []
        
        # Spłaszczamy tylko wymiary przestrzenne (H, W), zachowując Batch (N)
        # preds: (N, 1, H, W) -> (N, -1)
        preds_flat = preds.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)
        
        # Liczymy dla każdego obrazka w pętli lub wektorowo
        intersection = (preds_flat * targets_flat).sum(1) # Suma wzdłuż osi pikseli
        
        # Sumy pikseli dla każdego obrazka
        p_sum = preds_flat.sum(1)
        t_sum = targets_flat.sum(1)
        
        # Dice dla każdego obrazka: (2*I + e) / (U + e)
        dices = (2. * intersection + smooth) / (p_sum + t_sum + smooth)
        
        # IoU dla każdego obrazka: (I + e) / (U - I + e)
        union = p_sum + t_sum - intersection
        ious = (intersection + smooth) / (union + smooth)
        
        # Konwersja na listę Pythonową
        return dices.cpu().numpy().tolist(), ious.cpu().numpy().tolist()