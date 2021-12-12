from torch import nn
from .modules import *


class ConvTrans6(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(ConvTrans6, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.width = 64
        self.conv1 = ConvTrans(n_channels, self.width, self.width)
        self.conv2 = ConvTrans(self.width, self.width, self.width)
        self.conv3 = ConvTrans(self.width, self.width, self.width)
        self.conv4 = ConvTrans(self.width, self.width, self.width)
        self.conv5 = ConvTrans(self.width, self.width, self.width)
        self.conv6 = ConvTrans(self.width, self.width, self.width)
        self.outc = OutConv(self.width, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.outc(x)
        return x