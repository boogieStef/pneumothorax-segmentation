import sys
import subprocess
import importlib.metadata
from importlib.metadata import PackageNotFoundError

class EnvironmentSetup:
    """
    Handles environment verification and package installation.
    Uses modern importlib.metadata instead of deprecated pkg_resources.
    """

    def __init__(self, requirements_path):
        """
        Initialize the environment setup.

        Args:
            requirements_path (str): Path to the requirements.txt file.
        """
        self.requirements_path = requirements_path

    def _get_base_package_name(self, requirement_string):
        """
        Extracts the base package name from a requirement string.
        Example: 'timm==0.9.2' -> 'timm', 'numpy>=1.20' -> 'numpy'
        
        Args:
            requirement_string (str): A line from requirements.txt
            
        Returns:
            str: The clean package name.
        """
        # Split by common version operators to get the name
        delimiters = ['==', '>=', '<=', '>', '<', '~=', ';']
        cleaned_req = requirement_string
        
        for delimiter in delimiters:
            cleaned_req = cleaned_req.split(delimiter)[0]
            
        return cleaned_req.strip()

    def smart_install(self):
        """
        Reads the requirements file and installs only the missing packages.
        Checks if a package is already installed using importlib.metadata
        to avoid redundant installations and conflicts.
        """
        print(f"--- Smart installation from {self.requirements_path} ---")
        
        try:
            with open(self.requirements_path, 'r') as f:
                # Read lines, ignore comments and empty lines
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        except FileNotFoundError:
            print(f"[ERROR] Requirements file not found at: {self.requirements_path}")
            return

        packages_to_install = []
        
        for req in requirements:
            package_name = self._get_base_package_name(req)
            
            try:
                # modern replacement for pkg_resources.get_distribution()
                installed_version = importlib.metadata.version(package_name)
                print(f"[INFO] {package_name} is already installed (Version: {installed_version}). Skipping.")
            except PackageNotFoundError:
                # Package is strictly missing
                print(f"[INSTALL] {package_name} is missing. Adding to queue.")
                packages_to_install.append(req)
            except Exception as e:
                # Fallback for weird edge cases
                print(f"[WARNING] Could not check status for {package_name}. Adding to queue. Error: {e}")
                packages_to_install.append(req)

        # Install accumulated missing packages
        if packages_to_install:
            print(f"--- Installing missing packages: {', '.join(packages_to_install)} ---")
            try:
                # Use sys.executable to ensure we install in the current python environment
                subprocess.check_call([sys.executable, "-m", "pip", "install", *packages_to_install])
                print("--- Installation complete ---")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Installation failed: {e}")
        else:
            print("--- All requirements are satisfied. No changes made. ---")

    def check_cuda(self):
        """
        Optional: Extensible method to check CUDA availability.
        This allows the class to be expanded for hardware checks.
        """
        try:
            import torch
            if torch.cuda.is_available():
                print(f"[INFO] CUDA is available. Device count: {torch.cuda.device_count()}")
                print(f"[INFO] Current device: {torch.cuda.get_device_name(0)}")
            else:
                print("[WARNING] CUDA is NOT available. Training will be slow on CPU.")
        except ImportError:
            print("[ERROR] PyTorch is not installed. Cannot check CUDA.")