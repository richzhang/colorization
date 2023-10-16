import typing as T
from torch import nn

def build_basic_block(
        channels: T.List[int], kernel_size: T.Union[int, T.List[int]], 
        stride: T.Union[int, T.List[int]] = 1, dilation: T.Union[int, T.List[int]] = 1, 
        padding: T.Union[int, T.List[int]] = 1, bias: bool = True, norm_layer: bool = True, 
        conv_type: T.Union[nn.Module, T.List[nn.Module]] = nn.Conv2d, init_relu: bool = False
    ) -> nn.Sequential:
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * len(channels)
    if isinstance(stride, int):
        stride = [stride] * len(channels)
    if isinstance(dilation, int):
        dilation = [dilation] * len(channels)
    if isinstance(padding, int):
        padding = [padding] * len(channels)
    if not isinstance(conv_type, list):
        conv_type = [conv_type] * len(channels)

    layers = [] if not init_relu else [nn.ReLU(True)]
    for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
        layers.append(conv_type[i](in_channels, out_channels, kernel_size[i], stride[i], padding[i], dilation[i], bias=bias))
        layers.append(nn.ReLU(True))
    if norm_layer:
        layers.append(nn.BatchNorm2d(channels[-1]))
    return nn.Sequential(*layers)

# NOTE(Sebastian) the below function could be used to build a model from a 
# config file. We could do model search (or hyperparameter search) by
# specifying multiple configs and then using this function to build the models.

def build_layer(config: T.Dict) -> nn.Module:
    name = config['name']
    if name == 'BasicBlock':
        return build_basic_block(**config)

    if name == 'Sequential':
        return nn.Sequential(*[build_layer(**l) for l in config['layers']])
    
    module = getattr(nn, name, None)
    if module is None:
        raise ValueError(f'Unknown layer name {name}')
    return module(**config)
    