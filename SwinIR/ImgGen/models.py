from torch import nn
from .modules import *


class ConvTransUnet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(ConvTransUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dim = 128
        self.conv1 = ConvTrans(n_channels, self.dim)
        # self.conv2 = ConvTrans(128, 128)
        # self.conv3 = ConvTrans(128, 128)
        # self.conv4 = ConvTrans(32, 64)
        # self.conv5 = ConvTrans(64, 128)
        # self.conv6 = ConvTrans(self.width, 256, self.width)
        self.outc = OutConv_CNRC(self.dim, n_classes)

    def forward(self, x):
        x_res = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        x = self.outc(x_res+x)
        return x