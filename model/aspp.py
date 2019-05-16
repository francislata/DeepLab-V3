from .model_utils import create_convolutional_layer
import torch.nn as nn
import torch

class ASPP(nn.Module):
    """This is the implementation of the atrous spatial pyramid pooling module of DeepLab v3."""

    def __init__(self, in_channels, out_channels, dilations):
        super(ASPP, self).__init__()

        self.conv1 = create_convolutional_layer(in_channels, out_channels, 1, add_bn=True, add_relu=True)
        self.conv2 = create_convolutional_layer(in_channels, out_channels, 3, dilation=dilations[0], add_bn=True, add_relu=True)
        self.conv3 = create_convolutional_layer(in_channels, out_channels, 3, dilation=dilations[1], add_bn=True, add_relu=True)
        self.conv4 = create_convolutional_layer(in_channels, out_channels, 3, dilation=dilations[2], add_bn=True, add_relu=True)
        self.img_pool = create_convolutional_layer(in_channels, out_channels, 1, add_gap=True, add_bn=True, add_relu=True)

    def forward(self, x):
        conv1_y = self.conv1(x)
        conv2_y = self.conv2(x)
        conv3_y = self.conv3(x)
        conv4_y = self.conv4(x)

        # Bilinearly upsample the features to the height/width of the atrous convoluted feature
        img_pool_y = nn.functional.interpolate(self.img_pool(x), size=(conv1_y.size(2), conv1_y.size(3)), mode="bilinear", align_corners=True)

        return torch.cat([conv1_y, conv2_y, conv3_y, conv4_y, img_pool_y], dim=1)
