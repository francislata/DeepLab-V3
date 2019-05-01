import torch.nn as nn
import torch

class ASPP(nn.Module):
    """This is the implementation of the atrous spatial pyramid pooling module of DeepLab v3."""

    def __init__(self, in_channels, out_channels, dilations):
        super(ASPP, self).__init__()

        self.conv1 = self.create_conv_layer(in_channels, out_channels, 1)
        self.conv2 = self.create_conv_layer(in_channels, out_channels, 3, dilation=dilations[0])
        self.conv3 = self.create_conv_layer(in_channels, out_channels, 3, dilation=dilations[1])
        self.conv4 = self.create_conv_layer(in_channels, out_channels, 3, dilation=dilations[2])
        self.img_pool = self.create_image_pooling_layer(in_channels, out_channels)

    def forward(self, x):
        conv1_y = self.conv1(x)
        conv2_y = self.conv2(x)
        conv3_y = self.conv3(x)
        conv4_y = self.conv4(x)

        # Bilinearly upsample the features to the height/width of the atrous convoluted feature
        img_pool_y = nn.functional.interpolate(self.img_pool(x), size=(conv1_y.size(2), conv1_y.size(3)), mode="bilinear", align_corners=True)

        return torch.cat([conv1_y, conv2_y, conv3_y, conv4_y, img_pool_y], dim=1)

    def create_conv_layer(self, in_channels, out_channels, kernel_size, dilation=1):
        """Creates a convolutional layer with batch normalization"""
        if dilation > 1:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=dilation, dilation=dilation)
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
            
        conv_layer = nn.Sequential(
            conv,
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
        
        self._initialize_modules(conv_layer)
        return conv_layer

    def create_image_pooling_layer(self, in_channels, out_channels):
        """Creates the image pooling layer as input to ASPP module"""
        image_pooling_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
        
        self._initialize_modules(image_pooling_layer)
        return image_pooling_layer
    
    def _initialize_modules(self, modules):
        """Initializes the weights and bias of modules"""
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
