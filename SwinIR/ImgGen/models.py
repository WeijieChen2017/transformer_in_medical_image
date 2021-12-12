from torch import nn
from .modules import *
from .vit import *


class ConvTrans6(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(ConvTrans6, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.conv1 = ConvTrans(n_channels, 256, 256)
        self.conv2 = ConvTrans(256, 256, 256)
        self.conv3 = ConvTrans(256, 256, 256)
        self.conv4 = ConvTrans(256, 256, 256)
        self.conv5 = ConvTrans(256, 256, 256)
        self.conv6 = ConvTrans(256, 256, 256)
        self.outc = OutConv(256, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.outc(x)
        return x