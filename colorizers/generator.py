import torch
import typing as T
import torch.nn as nn
import json

class ModelConfig:
    def __init__(self, name: str, dropout: T.List[float], channelMultiplier: int =  1, numExtraConv2DLayers: int = 0) -> None:
        self.name = name
        self.dropoutLayers = dropout
        self.channelMultiplier = channelMultiplier
        self.numExtraConv2DLayers = numExtraConv2DLayers
    
    def dump(self, path: str) -> None:
        """
        Save the config to a file.
        """

        data = {
            'dropoutLayers': self.dropoutLayers,
            'channelMultiplier': self.channelMultiplier,
            'numExtraConv2DLayers': self.numExtraConv2DLayers,
        }

        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

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
    