import torch
import torch.nn as nn

#torch.set_default_tensor_type(torch.cuda.FloatTensor)


class SubSpace(nn.Module):

    def __init__(self, in_channels: int = 256, out_channels: int = 256):
        super(SubSpace, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride = 1),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(inplace=False),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.conv_block(x)
        out = self.conv_block(out)
        out1 = self.maxpool(out)
        out2 = self.avgpool(out)
        out = out1 + out2
        out = self.softmax(out)
        #print("=============================================================")
        out = torch.mul(out, x)
        out = out + x
        return out

