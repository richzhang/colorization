import torch
import typing as T
import torch.nn as nn
import json
from colorizers.modified import modified_colorizer

# Rudy - I need a little bit of clarification about what the purpose of this class is
class ModelConfig:
    def __init__(self, name: str, dropout: T.List[float], channelMultiplier: int =  1, numExtraConv2DLayers: int = 0) -> None:
        '''
        Need to ensure len(dropout) >= # of layers in base model
        '''
        self.name = name
        self.dropoutLayers = dropout if len(dropout) >= 10 else [0.0] * 10
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
    return modified_colorizer(config)
