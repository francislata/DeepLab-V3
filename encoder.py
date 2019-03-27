import torch.nn as nn
from resnet import resnet101
from aspp import ASPP

# Constants
IMAGE_SIZE = (224, 224)

class Encoder(nn.Module):
    """This is the implementation for combining the pre-trained backbone with the ASPP module."""

    def __init__(self, num_classes):
        super(Encoder, self).__init__()

        self.num_classes = num_classes
        self.backbone = resnet101(pretrained=True)
        self.aspp = ASPP(2048, 256, [6, 12, 18])
        self.conv = self.create_convolutional_layer(1280, 256, 1)
        self.conv_upsampling = self.create_convolutional_upsampling_layer(256, 1, 16)

    def forward(self, x):
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.conv(x)
        x = self.conv_upsampling(x)

        return x

    def create_convolutional_layer(self, in_channels, out_channels, kernel_size):
        """Creates a convolutional layer with batch normalization"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels)
        )

    def create_convolutional_upsampling_layer(self, in_channels, kernel_size, scale_factor):
        """Creates a convolutional layer with upsampling and softmax"""
        return nn.Sequential(
            nn.Conv2d(in_channels, self.num_classes, 1),
            nn.Upsample(scale_factor=scale_factor, mode="bilinear")
        )
