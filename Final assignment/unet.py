import torch
import torch.nn as nn
import torch.ao.quantization
from torch.ao.quantization import QuantStub, DeQuantStub


class UNet(nn.Module):
    """ 
    A simple U-Net architecture for image segmentation.
    Based on the U-Net architecture from the original paper:
    Olaf Ronneberger et al. (2015), "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf
    """

    def __init__(self, in_channels=3, n_classes=19, quantize=False):
        super(UNet, self).__init__()

        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 512))
        self.up1 = (Up(1024, 256))
        self.up2 = (Up(512, 128))
        self.up3 = (Up(256, 64))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))

        # Quantization
        self.quantize = quantize
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
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
        if self.quantize:
            logits = self.dequant(logits)
        return logits

    def fuse_model(self, is_qat=False):
        fuse_modules = torch.ao.quantization.fuse_modules_qat if is_qat else torch.ao.quantization.fuse_modules

        for name, m in self.named_children():
            if isinstance(m, DoubleConv):
                fuse_modules(m.double_conv, ['0', '1', '2'], inplace=True)
                fuse_modules(m.double_conv, ['3', '4', '5'], inplace=True)
            elif isinstance(m, Down):
                if isinstance(m.double_conv, DoubleConv):
                    fuse_modules(m.double_conv.double_conv, ['0', '1', '2'], inplace=True)
                    fuse_modules(m.double_conv.double_conv, ['3', '4', '5'], inplace=True)
            elif isinstance(m, Up):
                if isinstance(m.double_conv, DoubleConv):
                    fuse_modules(m.double_conv.double_conv, ['0', '1', '2'], inplace=True)
                    fuse_modules(m.double_conv.double_conv, ['3', '4', '5'], inplace=True)
            print(f"Fused {name}")


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.MaxPool2d(2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool_conv(x)
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)