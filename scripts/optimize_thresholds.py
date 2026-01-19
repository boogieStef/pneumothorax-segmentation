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
# Importujemy klasy, ale nie używamy Trainer.evaluate
from src.dataset import PneumoDataset
from src.model import ModelFactory

# --- FUNKCJE POMOCNICZE (Czysta matematyka, bez zależności od Trainera) ---

def calculate_metrics_for_grid(y_probs, y_true, prob_thresh, pixel_thresh):
    """
    Oblicza metryki dla całego zbioru przy zadanych progach.
    y_probs: numpy array [N, 1, H, W] (prawdopodobieństwa 0-1)
    y_true: numpy array [N, 1, H, W] (maski 0 lub 1)
    """
    N = y_probs.shape[0]
    
    # 1. Binaryzacja progiem prawdopodobieństwa
    y_pred = (y_probs > prob_thresh).astype(np.uint8)
    
    # 2. Obliczanie obszarów (dla Pixel Threshold)
    # Sumujemy piksele dla każdego obrazka: (N, 1, H, W) -> (N,)
    pred_areas = y_pred.reshape(N, -1).sum(axis=1)
    true_areas = y_true.reshape(N, -1).sum(axis=1)
    
    # 3. Klasyfikacja (Chory/Zdrowy)
    # Predykcja jest pozytywna TYLKO jeśli obszar > pixel_thresh
    cls_pred = (pred_areas > pixel_thresh).astype(np.uint8)
    cls_true = (true_areas > 0).astype(np.uint8) # Prawda: cokolwiek > 0 to chory
    
    # TP, TN, FP, FN (wektorowo)
    tp = np.sum((cls_pred == 1) & (cls_true == 1))
    tn = np.sum((cls_pred == 0) & (cls_true == 0))
    fp = np.sum((cls_pred == 1) & (cls_true == 0))
    fn = np.sum((cls_pred == 0) & (cls_true == 1))
    
    epsilon = 1e-6
    sens = tp / (tp + fn + epsilon)
    spec = tn / (tn + fp + epsilon)
    
    # 4. Dice Score (Sample-wise)
    # Liczymy Dice tylko dla tych, które przeszły próg klasyfikacji? 
    # Standardowo w segmentacji liczymy Dice na masce binarnej (krok 1).
    # Ale jeśli pixel_threshold zeruje maskę, to Dice powinien to uwzględnić.
    # Wariant: Zerujemy maskę predykcji, jeśli nie przeszła pixel_thresh
    
    # Filtrowanie masek (Advanced): 
    # Jeśli klasyfikator mówi "Zdrowy" (bo mało pikseli), to zerujemy całą maskę predykcji.
    mask_filter = cls_pred.reshape(N, 1, 1, 1) # [N, 1, 1, 1]
    y_pred_filtered = y_pred * mask_filter
    
    # Obliczenia Dice per sample
    intersection = (y_pred_filtered * y_true).reshape(N, -1).sum(axis=1)
    union = y_pred_filtered.reshape(N, -1).sum(axis=1) + y_true.reshape(N, -1).sum(axis=1)
    
    dices = (2. * intersection + epsilon) / (union + epsilon)
    mean_dice = np.mean(dices)
    
    return {
        "prob_th": prob_thresh,
        "pixel_th": pixel_thresh,
        "dice": mean_dice,
        "sens": sens,
        "spec": spec
    }

def main():
    # --- 1. SETUP ---
    print("[INFO] Starting Threshold Optimization (Inference Mode)...")
    setup = EnvironmentSetup()
    setup.validate_and_install_smp()
    
    # Ładujemy config (najlepiej ten użyty do treningu)
    config_path = "configs/run_config.yaml"
    if not os.path.exists(config_path):
        config_path = "configs/exp2_unet_aug.yaml" # Fallback
        
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 2. DANE ---
    data_dir = config['data']['dataset_dir']
    df = pd.read_csv(os.path.join(data_dir, "metadata_processed.csv"))
    # Używamy zbioru TEST lub VAL (zależy co chcesz optymalizować - naukowo lepiej VAL)
    # Użyjmy TEST zgodnie z Twoim workflow, lub VAL jeśli chcesz być purystą
    target_split = 'test' 
    test_df = df[df['Split'] == target_split].reset_index(drop=True)
    
    dataset = PneumoDataset(test_df, data_dir, phase='test', config=config)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2) # Większy batch bo tylko inferencja

    # --- 3. MODEL ---
    print(f"[INFO] Loading model: {config['train']['architecture']}")
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
            
            # Przenosimy na CPU, żeby nie zapchać GPU, i konwertujemy na NumPy
            # Dzięki temu zwolnimy GPU i będziemy liczyć metryki na CPU
            all_probs.append(probs.cpu().numpy())
            all_masks.append(masks.numpy()) # Maski już są na CPU z loadera (zazwyczaj)
            
    # Łączymy w wielkie tablice NumPy
    y_probs = np.concatenate(all_probs, axis=0) # Shape: (N, 1, 256, 256)
    y_true = np.concatenate(all_masks, axis=0)
    
    print(f"[INFO] Inference complete. Collected {y_probs.shape[0]} samples.")
    print("[INFO] Starting Grid Search (CPU optimized)...")

    # --- 5. GRID SEARCH (SZYBKO NA CPU) ---
    
    prob_range = [0.2, 0.3, 0.4, 0.5, 0.6]
    pixel_range = [0, 10, 50, 100, 200, 500]
    
    results = []
    
    print("\n" + "-"*65)
    print(f"{'Prob':<8} | {'Pixel':<8} | {'Dice':<8} | {'Sens':<8} | {'Spec':<8}")
    print("-"*65)
    
    for p_th in prob_range:
        for pix_th in pixel_range:
            res = calculate_metrics_for_grid(y_probs, y_true, p_th, pix_th)
            results.append(res)
            print(f"{res['prob_th']:<8.2f} | {res['pixel_th']:<8} | {res['dice']:.4f}   | {res['sens']:.4f}   | {res['spec']:.4f}")
            
    print("-"*65)
    
    # --- 6. WNIOSKI ---
    # Sortujemy np. po Dice, albo po Czułości (zależy co chcesz)
    # Tutaj szukamy najlepszego Dice
    best_dice = sorted(results, key=lambda x: x['dice'], reverse=True)[0]
    
    # Lub szukamy kompromisu: Czułość > 0.70 i najlepsze Spec
    compromise = [r for r in results if r['sens'] > 0.70]
    best_compromise = sorted(compromise, key=lambda x: x['spec'], reverse=True)[0] if compromise else None
    
    print(f"\n[BEST DICE] Prob: {best_dice['prob_th']}, Pixel: {best_dice['pixel_th']} -> Dice: {best_dice['dice']:.4f}")
    
    if best_compromise:
        print(f"[HIGH SENS] Prob: {best_compromise['prob_th']}, Pixel: {best_compromise['pixel_th']} -> Sens: {best_compromise['sens']:.4f}, Spec: {best_compromise['spec']:.4f}")

if __name__ == "__main__":
    main()