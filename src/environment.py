import sys
import subprocess
import os
import torch
import importlib.util

class EnvironmentSetup:
    """
    Handles environment verification, surgical package installation,
    and GPU stability checks specifically for Kaggle.
    """

    def __init__(self):
        # Na Kaggle często chcemy wymusić użycie pierwszego GPU, by uniknąć konfliktów DDP
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    def _run_pip_install(self, package_name, no_deps=False):
        """Helper to run pip install via subprocess."""
        cmd = [sys.executable, "-m", "pip", "install", package_name, "-q"]
        if no_deps:
            cmd.append("--no-deps")
        
        print(f"[INSTALL] Installing {package_name} (no_deps={no_deps})...")
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to install {package_name}. Error: {e}")
            raise e

    def validate_and_install_smp(self):
        """
        Performs the 'surgical' installation of segmentation-models-pytorch
        to avoid breaking Kaggle's pre-installed PyTorch/CUDA environment.
        """
        print("\n--- Environment Validation: SMP Surgical Install ---")
        
        # Sprawdzamy czy pakiet już jest (żeby nie instalować bez sensu)
        if importlib.util.find_spec("segmentation_models_pytorch") is None:
            # 1. Kolejność instalacji z Twojego skryptu
            self._run_pip_install("munch")
            self._run_pip_install("pretrainedmodels==0.7.4", no_deps=True)
            self._run_pip_install("efficientnet-pytorch==0.7.1", no_deps=True)
            self._run_pip_install("timm==0.9.2", no_deps=True) # Dodane, bo SMP tego potrzebuje do nowszych encoderów
            self._run_pip_install("segmentation-models-pytorch==0.3.3", no_deps=True)
            print("[INFO] SMP installed successfully.")
        else:
            print("[INFO] SMP already installed. Skipping installation steps.")

        # 2. Test Importu
        try:
            import segmentation_models_pytorch as smp
            print("[CHECK] Import 'segmentation_models_pytorch': SUCCESS")
        except ImportError as e:
            print(f"[FAIL] Import failed: {e}")
            raise e

    def validate_gpu_forward_pass(self):
        """
        Runs a dummy forward pass on GPU to ensure Kernel stability.
        Detects 'Kernel Dead' issues before main training starts.
        """
        print("\n--- Environment Validation: GPU Stability Test ---")
        
        if not torch.cuda.is_available():
            print("[WARNING] CUDA not available! Training will be extremely slow.")
            return

        try:
            import segmentation_models_pytorch as smp
            device = torch.device("cuda")
            
            # Tworzymy lekki model testowy (musi pasować kanałami do configu - zazwyczaj 3)
            print("[TEST] Allocating dummy model on GPU...")
            model = smp.Unet(
                encoder_name="resnet18", # Lekki encoder do testu
                encoder_weights=None,    # Nie pobieramy wag do testu, szkoda czasu
                in_channels=3,           # Ważne: Twój Dataset zwraca RGB (3 kanały)
                classes=1
            ).to(device)

            # Dummy data: Batch=2, Channels=3, 256x256
            dummy_input = torch.randn(2, 3, 256, 256).to(device)

            print("[TEST] Running forward pass...")
            with torch.no_grad():
                output = model(dummy_input)

            print(f"[SUCCESS] Forward pass complete. Output shape: {output.shape}")
            print("--- Environment seems STABLE ---\n")
            
            # Czyścimy pamięć po teście
            del model, dummy_input, output
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[CRITICAL FAIL] GPU Test failed. This would cause a Kernel Crash.")
            print(f"Error: {e}")
            raise e