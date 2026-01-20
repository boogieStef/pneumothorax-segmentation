import sys
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# Setup ścieżek
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment import EnvironmentSetup
from src.utils import load_config
from src.dataset import PneumoDataset
from src.model import ModelFactory

# --- FUNKCJE POMOCNICZE ---

def calculate_metrics_for_grid(y_probs, y_true, prob_thresh, pixel_thresh):
    """
    Oblicza komplet metryk (Dice, IoU, Sens, Spec, Acc) dla zadanych progów.
    """
    N = y_probs.shape[0]
    epsilon = 1e-6
    
    # 1. Binaryzacja progiem prawdopodobieństwa
    y_pred_raw = (y_probs > prob_thresh).astype(np.uint8)
    
    # 2. Obliczanie obszarów (dla Pixel Threshold)
    pred_areas = y_pred_raw.reshape(N, -1).sum(axis=1)
    true_areas = y_true.reshape(N, -1).sum(axis=1)
    
    # 3. Klasyfikacja (Chory/Zdrowy) - Detekcja
    # Predykcja jest pozytywna (1) TYLKO jeśli obszar > pixel_thresh
    cls_pred = (pred_areas > pixel_thresh).astype(np.uint8)
    cls_true = (true_areas > 0).astype(np.uint8) 
    
    # TP, TN, FP, FN (wektorowo dla klasyfikacji)
    tp = np.sum((cls_pred == 1) & (cls_true == 1))
    tn = np.sum((cls_pred == 0) & (cls_true == 0))
    fp = np.sum((cls_pred == 1) & (cls_true == 0))
    fn = np.sum((cls_pred == 0) & (cls_true == 1))
    
    sens = tp / (tp + fn + epsilon)
    spec = tn / (tn + fp + epsilon)
    acc  = (tp + tn) / (tp + tn + fp + fn + epsilon)
    
    # 4. Metryki Segmentacyjne (Dice & IoU)
    # Filtrowanie: Zerujemy maskę predykcji, jeśli klasyfikator uznał obraz za "Zdrowy" (bo za mały obszar)
    mask_filter = cls_pred.reshape(N, 1, 1, 1) # [N, 1, 1, 1]
    y_pred_filtered = y_pred_raw * mask_filter
    
    # Spłaszczanie do [N, -1]
    flat_pred = y_pred_filtered.reshape(N, -1)
    flat_true = y_true.reshape(N, -1)
    
    intersection = (flat_pred * flat_true).sum(axis=1)
    union_poly = flat_pred.sum(axis=1) + flat_true.sum(axis=1)
    
    # Dice per sample
    dices = (2. * intersection + epsilon) / (union_poly + epsilon)
    mean_dice = np.mean(dices)
    
    # IoU per sample
    # Union dla IoU to (A + B - Intersection)
    union_iou = union_poly - intersection
    ious = (intersection + epsilon) / (union_iou + epsilon)
    mean_iou = np.mean(ious)
    
    return {
        "prob": prob_thresh,
        "pixel": pixel_thresh,
        "dice": mean_dice,
        "iou": mean_iou,
        "sens": sens,
        "spec": spec,
        "acc": acc
    }

def main():
    # --- 1. SETUP ---
    print("[INFO] Starting Threshold Optimization (Full Metrics)...")
    setup = EnvironmentSetup()
    setup.validate_and_install_smp()
    
    config_path = "configs/run_config.yaml"
    if not os.path.exists(config_path):
        config_path = "configs/exp2_unet_aug.yaml"
        
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 2. DANE (Test Set) ---
    data_dir = config['data']['dataset_dir']
    df = pd.read_csv(os.path.join(data_dir, "metadata_processed.csv"))
    test_df = df[df['Split'] == 'test'].reset_index(drop=True)
    
    dataset = PneumoDataset(test_df, data_dir, phase='test', config=config)
    # Większy batch dla przyspieszenia inferencji
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    # --- 3. MODEL ---
    print(f"[INFO] Loading model architecture: {config['train']['architecture']}")
    model = ModelFactory.create(config).to(device)
    
    weights_path = "best_model.pth"
    if not os.path.exists(weights_path):
        print(f"[ERROR] {weights_path} not found.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # --- 4. INFERENCJA (TYLKO RAZ) ---
    print("[INFO] Running Inference on dataset (collecting logits)...")
    
    all_probs = []
    all_masks = []
    
    with torch.no_grad():
        for images, masks, _ in tqdm(loader):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu().numpy())
            all_masks.append(masks.numpy())
            
    y_probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_masks, axis=0)
    
    print(f"[INFO] Inference complete. Collected {y_probs.shape[0]} samples.")
    print("[INFO] Starting Grid Search (CPU optimized)...")

    # --- 5. GRID SEARCH ---
    
    # Zakresy do sprawdzenia
    prob_range = [0.2, 0.3, 0.4, 0.5, 0.6]
    pixel_range = [0, 10, 50, 100, 200, 500]
    
    results = []
    
    # Nagłówek tabeli
    print("\n" + "-"*95)
    print(f"{'Prob':<6} | {'Pixel':<6} | {'Dice':<8} | {'IoU':<8} | {'Sens':<8} | {'Spec':<8} | {'Acc':<8}")
    print("-" * 95)
    
    for p_th in prob_range:
        for pix_th in pixel_range:
            res = calculate_metrics_for_grid(y_probs, y_true, p_th, pix_th)
            results.append(res)
            
            print(f"{res['prob']:<6.2f} | {res['pixel']:<6} | {res['dice']:.4f}   | {res['iou']:.4f}   | {res['sens']:.4f}   | {res['spec']:.4f}   | {res['acc']:.4f}")
            
    print("-" * 95)
    
    # --- 6. PODSUMOWANIE (Best for Each Metric) ---
    print("\n" + "="*40)
    print("       BEST CONFIGURATIONS")
    print("="*40)
    
    metrics_to_check = [
        ('dice', 'Highest Dice Score (Segmentation Quality)'),
        ('iou',  'Highest IoU (Intersection over Union)'),
        ('sens', 'Highest Sensitivity (Recall - Detection)'),
        ('spec', 'Highest Specificity (Low False Alarms)'),
        ('acc',  'Highest Accuracy (Overall Correctness)')
    ]
    
    for metric_key, description in metrics_to_check:
        # Sortujemy malejąco po danej metryce
        best_row = sorted(results, key=lambda x: x[metric_key], reverse=True)[0]
        
        print(f"\n[{metric_key.upper()}] {description}:")
        print(f"   Value: {best_row[metric_key]:.4f}")
        print(f"   Thresholds: Prob > {best_row['prob']}, Area > {best_row['pixel']} px")

    # Opcjonalnie: Kompromis (Dobra czułość przy dobrym Dice)
    print("\n[COMPROMISE] High Sensitivity (>0.70) with Best Specificity:")
    compromise_candidates = [r for r in results if r['sens'] > 0.70]
    if compromise_candidates:
        best_comp = sorted(compromise_candidates, key=lambda x: x['spec'], reverse=True)[0]
        print(f"   Sens: {best_comp['sens']:.4f} | Spec: {best_comp['spec']:.4f} | Dice: {best_comp['dice']:.4f}")
        print(f"   Thresholds: Prob > {best_comp['prob']}, Area > {best_comp['pixel']} px")
    else:
        print("   No configuration met the sensitivity > 0.70 criteria.")
        
    print("="*40 + "\n")

if __name__ == "__main__":
    main()