import torch
import typing as T
import torch.nn as nn

class ModelConfig:
    def __init__(self, name: str) -> None:
        self.name = name
        raise NotImplementedError
    
    def dump(self, path: str) -> None:
        """
        Save the config to a file.
        """
        raise NotImplementedError


"""
Base model: model in eccv16.py
Model n: Base model but channels times n
n from 1 to 20
Bigger model: Base model + convtranspose2d layers 1-5 extra layers
Dropout: in basic block try 0.1 - 0.8
"""

def generate_model(config: ModelConfig) -> nn.Module:
    """
    Build a model from the given config.
    """
    raise NotImplementedError