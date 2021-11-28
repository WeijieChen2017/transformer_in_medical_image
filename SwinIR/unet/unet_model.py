""" Full assembly of the parts to form the complete network """
from torch import nn
from .unet_parts import *
from .vit import *
from einops.layers.torch import Rearrange

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet_simple(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_simple, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.hidden_1 = DoubleConv(1024, 1024)
        self.hidden_2 = DoubleConv(1024, 1024)
        self.up1 = Up_simple(1024, 512, bilinear)
        self.up2 = Up_simple(512, 256, bilinear)
        self.up3 = Up_simple(256, 128, bilinear)
        self.up4 = Up_simple(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)
        return logits


class UNet_E(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.hidden = DoubleConv(1024, 1024)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return self.hidden(x)

class UNet_D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.hidden = DoubleConv(1024, 1024)
        self.up1 = SimpleUp(1024, 512, bilinear)
        self.up2 = SimpleUp(512, 256, bilinear)
        self.up3 = SimpleUp(256, 128, bilinear)
        self.up4 = SimpleUp(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet_bridge(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_bridge, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64) # 256
        self.down1 = Down(64, 128) # 256
        self.down2 = Down(128, 256) # 128
        self.down3 = Down(256, 512) # 64
        self.down4 = Down(512, 1024) # 32
        self.hidden_1 = DoubleConv(1024, 1024) # 16

        # -->Input---> torch.Size([10, 3, 256, 256])
        # -->Inc-----> torch.Size([10, 64, 256, 256])
        # -->Down1---> torch.Size([10, 128, 128, 128])
        # -->Down2---> torch.Size([10, 256, 64, 64])
        # -->Down3---> torch.Size([10, 512, 32, 32])
        # -->Down4---> torch.Size([10, 1024, 16, 16])
        # -->Hidden1-> torch.Size([10, 1024, 16, 16])
        # -->Hidden2-> torch.Size([10, 1024, 16, 16])
        # -->Up1-----> torch.Size([10, 512, 32, 32])
        # -->Up2-----> torch.Size([10, 256, 64, 64])
        # -->Up3-----> torch.Size([10, 128, 128, 128])
        # -->Up4-----> torch.Size([10, 64, 256, 256])
        # -->Outc----> torch.Size([10, 1, 256, 256])

        CompFea_lenX, CompFea_lenY = 16, 16
        patch_lenX, patch_lenY = 1, 1
        num_patches = (CompFea_lenX // patch_lenX) * (CompFea_lenY // patch_lenY) # 256
        patch_dim = 1024 * patch_lenX * patch_lenY # 1024
        dim = 1024

        # 10, 1024, 16, 16 -> 10, 256, 1024 -> 10, 256, 1024
        self.embedding = nn.Sequential(
            Rearrange('b c (cfx px) (cfy py) -> b (cfx cfy) (px py c)', px = patch_lenX, py = patch_lenX),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer = vit.Transformer(dim=dim, depth=6, heads=16,
                                           dim_head=64, mlp_dim=1024, dropout=0.1)

        # image_size = 256,
        # patch_size = 32,
        # num_classes = 1000,
        # dim = 1024,
        # depth = 6,
        # heads = 16,
        # mlp_dim = 2048,
        # dropout = 0.1,
        # emb_dropout = 0.1

        self.unembedding = 
        self.hidden_2 = DoubleConv(1024, 1024)
        self.up1 = Up_simple(1024, 512, bilinear)
        self.up2 = Up_simple(512, 256, bilinear)
        self.up3 = Up_simple(256, 128, bilinear)
        self.up4 = Up_simple(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        print()
        print("-->Input--->", x.size())
        x = self.inc(x)
        print("-->Inc--->", x.size())
        x = self.down1(x)
        print("-->Down1--->", x.size())
        x = self.down2(x)
        print("-->Down2--->", x.size())
        x = self.down3(x)
        print("-->Down3--->", x.size())
        x = self.down4(x)
        print("-->Down4--->", x.size())
        x = self.hidden_1(x)
        print("-->Hidden1--->", x.size())

        x = self.embedding(x) + self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        print("-->Bridge--->", x.size())
        exit()


        x = self.hidden_2(x)
        print("-->Hidden2--->", x.size())
        x = self.up1(x)
        print("-->Up1--->", x.size())
        x = self.up2(x)
        print("-->Up2--->", x.size())
        x = self.up3(x)
        print("-->Up3--->", x.size())
        x = self.up4(x)
        print("-->Up4--->", x.size())
        x = self.outc(x)
        print("-->Outc--->", x.size())
        
        exit()
        return x