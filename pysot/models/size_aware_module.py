import torch
import torch.nn as nn


"""
这个版本的感受野扩充模块中包含了两个   串行   的空洞卷积层，   膨胀率分别为 2 和 3
"""
class SAM(nn.Module):
    """
    This module contains two types of components:
        - the original FPN lateral convolution layer and fpn convolution layer,
          which are 1x1 conv + 3x3 conv
        - the dilated residual block
    """

    def __init__(self):
        super(SAM, self).__init__()

        self.lateral_conv = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1),
                nn.BatchNorm2d(256),
            )
        self.fpn_conv = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding = 1),
                nn.BatchNorm2d(256),
            )
        self.btnk1 = Bottleneck(dilation = 2)
        self.btnk2 = Bottleneck(dilation = 3)


    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        out = self.lateral_conv(feature)
        out = self.fpn_conv(out)
        out = self.btnk1(out)
        #print("===========>>>>>        out1 size = {}".format(out1.size()))
        out = self.btnk2(out)
        #print("===========>>>>>        out2 size = {}".format(out2.size()))
        #out = out1 + out2
        return out


class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels: int = 256,
                 mid_channels: int = 256,
                 dilation: int = 1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out