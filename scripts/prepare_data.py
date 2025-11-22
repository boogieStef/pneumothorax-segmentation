import argparse
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_config
from src.preprocessing import DataPreprocessor

def main():
    parser = argparse.ArgumentParser(description="Run offline data preprocessing.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml", 
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        return

    # Initialize and run preprocessor
    preprocessor = DataPreprocessor(config)
    preprocessor.run()

if __name__ == "__main__":
    main()