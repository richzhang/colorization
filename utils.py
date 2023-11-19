import os
from pathlib import Path
import torch

def get_root_dir() -> Path:
    """
    Return the root directory of the project.
    """
    return Path(os.getenv('COLORIZATION_DIR', '.'))


def get_device() -> torch.device:
    """
    Return the device to use for training.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device