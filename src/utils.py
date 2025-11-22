import numpy as np
import yaml

def load_config(config_path):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def rle_to_mask(rle_string, height, width):
    """
    Converts a Run-Length Encoding (RLE) string into a binary mask.

    Args:
        rle_string (str): Space-delimited RLE string (e.g., '1 10 20 5').
        height (int): Height of the target mask.
        width (int): Width of the target mask.

    Returns:
        np.ndarray: A binary mask of shape (height, width) with 1s and 0s.
    """
    # Handle cases with no pneumothorax (-1)
    if rle_string == '-1' or rle_string == -1:
        return np.zeros((height, width), dtype=np.uint8)

    # Parse RLE string
    array = np.asarray([int(x) for x in rle_string.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    # Flatten mask for easy assignment
    mask_flattened = np.zeros(height * width, dtype=np.uint8)

    for index, start in enumerate(starts):
        current_position += start
        mask_flattened[current_position : current_position + lengths[index]] = 1
        current_position += lengths[index]

    # Reshape to 2D. Note: RLE in this dataset is column-major (Fortran-style)
    return mask_flattened.reshape(width, height).T

