from torch import nn
from .modules import *


class ConvTransUnet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(ConvTransUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.conv1 = ConvTrans(n_channels, 80)
        self.conv2 = ConvTrans(80, 80)
        self.conv3 = ConvTrans(80, 80)
        self.conv4 = ConvTrans(80, 80)
        self.conv5 = ConvTrans(80, 80)
        # self.conv6 = ConvTrans(self.width, 256, self.width)
        self.outc = OutConv(80, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.conv6(x)
        x = self.outc(x)
        return x