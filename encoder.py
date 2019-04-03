from resnet import resnet101
from aspp import ASPP
import torch.nn as nn

class Encoder(nn.Module):
    """This is the implementation for combining the pre-trained backbone with the ASPP module."""

    def __init__(self, num_classes):
        super(Encoder, self).__init__()

        self.num_classes = num_classes
        self.backbone = resnet101(pretrained=True)
        self.aspp = ASPP(2048, 256, [6, 12, 18])
        self.conv = self.create_convolutional_layer(1280, 256, 1)
        self.final_conv = self.create_final_convolutional_layer(256, 1)

    def forward(self, x):
        upsampling = nn.Upsample(size=(x.size(2), x.size(3)), align_corners=True, mode="bilinear")
        
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.conv(x)
        x = self.final_conv(x)
        x = upsampling(x)

        return x

    def create_convolutional_layer(self, in_channels, out_channels, kernel_size):
        """Creates a convolutional layer with batch normalization"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def create_final_convolutional_layer(self, in_channels, kernel_size):
        """Creates a final convolutional layer"""
        return nn.Conv2d(in_channels, self.num_classes, 1)
