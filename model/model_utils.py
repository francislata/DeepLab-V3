import torch.nn as nn

def create_convolutional_layer(in_channels, out_channels, kernel_size, dilation=1, add_gap=False, add_bn=False, add_relu=False):
    """Creates a convolutional layer with optional layers before and after the convolution"""
    modules = []

    if add_gap:
        gap = nn.AdaptiveAvgPool2d(1)
        modules.append(gap)

    padding = 0 if dilation == 1 else dilation
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=(not add_bn))
    modules.append(conv)

    if add_bn:
        bn = nn.BatchNorm2d(out_channels)
        modules.append(bn)

    if add_relu:
        relu = nn.ReLU(inplace=True)
        modules.append(relu)

    layer = nn.Sequential(*modules)
    initialize_module_weights(layer)

    return layer

def initialize_module_weights(layer):
    """Initializes of every module weight in for a layer"""

    # Credit from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L148
    for m in layer.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
