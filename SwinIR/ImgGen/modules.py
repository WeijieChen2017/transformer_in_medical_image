""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from .vit import *

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        # x = self.sigmoid(x)
        x = self.conv1(x)
        return x

class ConvTrans(nn.Module):
    """(convolution => [N] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels, img_size=256, tf_width=256, patch_len=8):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.img_size = img_size
        self.patch_len = patch_len
        self.patch_num = self.img_size // self.patch_len
        self.transformer_width = tf_width

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        patch_dim = out_channels * 4 * self.patch_num * self.patch_num # 1024
        patch_flatten_len = self.patch_len * self.patch_len
        dim = self.transformer_width

        # 10, 1024, 16, 16 -> 10, 256, 1024 -> 10, 256, 1024
        self.embedding = nn.Sequential(
            Rearrange('b c (pnx plx) (pny ply) -> b (plx ply) (pnx pny c)', pnx = self.patch_num, pny = self.patch_num),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, patch_flatten_len, dim))
        self.dropout = nn.Dropout(0.25)

        self.transformer = Transformer(dim=dim, depth=3, heads=64,
                                       dim_head=64, mlp_dim=256, dropout=0.25)

        # image_size = 256,
        # patch_size = 32,
        # num_classes = 1000,
        # dim = 1024,
        # depth = 6,
        # heads = 16,
        # mlp_dim = 2048,
        # dropout = 0.1,
        # emb_dropout = 0.1

        #-->embedding---> torch.Size([10, 256, 1024])
        #-->dropout---> torch.Size([10, 256, 1024])
        #-->Bridge---> torch.Size([10, 256, 1024])

        self.unembedding = nn.Sequential(
            nn.Linear(dim, patch_dim // 4),
            Rearrange(' b (plx ply) (pnx pny c) -> b c (pnx plx) (pny ply)', 
                plx = self.patch_len, ply = self.patch_len,
                pnx = self.patch_num, pny = self.patch_num)
        )


    def forward(self, x):
        x1 = self.conv1(x) # 1*1
        x2 = self.conv2(x) # 3*3
        x3 = self.conv3(x) # 3*3 -> 3*3
        x4 = self.conv4(x) # 3*3 -> 3*3 -> 3*3
        # print(x1.size(), x2.size(), x3.size(), x4.size())
        x1234 = torch.cat([x1, x2, x3, x4], dim=1)
        # print("-->x1234--->", x1234.size())
        x1234 = self.embedding(x1234) + self.pos_embedding
        # print("-->embedding--->", x1234.size())
        x1234 = self.dropout(x1234)
        # print("-->dropout--->", x1234.size())
        x1234 = self.transformer(x1234)
        # print("-->transformer--->", x1234.size())
        x1234 = self.unembedding(x1234)
        # print("-->unembedding--->", x1234.size())
        return x1234