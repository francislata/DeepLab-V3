from resnet import resnet101
from aspp import ASPP
from model_utils import create_convolutional_layer
import torch.nn as nn

class Encoder(nn.Module):
    """This is the implementation for combining the pre-trained backbone with the ASPP module."""

    def __init__(self, num_classes):
        super(Encoder, self).__init__()

        self.backbone = resnet101(pretrained=True)
        self.aspp = ASPP(2048, 256, [6, 12, 18])
        self.conv = create_convolutional_layer(1280, 256, 1, add_bn=True, add_relu=True)
        self.final_conv = create_convolutional_layer(256, num_classes, 1)

    def forward(self, x):
        size = (x.size(2), x.size(3))
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.conv(x)
        x = self.final_conv(x)
        x = nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=True)

        return x
