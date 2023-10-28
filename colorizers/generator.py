import torch
import typing as T
import torch.nn as nn

# Rudy - I need a little bit of clarification about what the purpose of this class is
class ModelConfig:
    def __init__(self, name: str, dropout: float = 0.0, channelMultiplier: int =  1, numExtraConv2DLayers: int = 0) -> None:
        self.name = name
        self.model = None
        # for dropout, build_basic_block currently doesn't support dropout layers, is it okay to edit to include that? -> if so
        # we'd have to make edits everywhere else we call the function
        raise NotImplementedError
    
    def dump(self, path: str) -> None:
        """
        Save the config to a file.
        """
        # Rudy:
        # Bit unsure of what you mean by config / what format to save it as;
        # I looked through build_layer in layers.py and its given me a slight idea of what
        # the input function expects but I'm still a little lost.
        torch.save(self.model.state_dict(), path)


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