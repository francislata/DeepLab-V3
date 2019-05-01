from resnet import resnet101
from aspp import ASPP
import torch.nn as nn

class Encoder(nn.Module):
    """This is the implementation for combining the pre-trained backbone with the ASPP module."""

    def __init__(self, num_classes):
        super(Encoder, self).__init__()

        self.backbone = resnet101(pretrained=True)
        self.aspp = ASPP(2048, 256, [6, 12, 18])
        self.conv = self.create_convolutional_layer(1280, 256, 1)
        self.final_conv = self.create_convolutional_layer(256, num_classes, 1, with_bn=False)

    def forward(self, x):
        size = (x.size(2), x.size(3))
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.conv(x)
        x = self.final_conv(x)
        x = nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=True)

        return x

    def create_convolutional_layer(self, in_channels, out_channels, kernel_size, with_bn=True):
        """Creates a convolutional layer with optional batch normalization"""
        if with_bn:
            convolutional_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            )
        else:
            convolutional_layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size))

        self._initialize_modules(convolutional_layer)
        return convolutional_layer

    def _initialize_modules(self, modules):
        """Initializes the weights and bias of modules"""
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
