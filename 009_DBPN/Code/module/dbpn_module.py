import torch
import torch.nn as nn
import math

class ConvBlock(nn.Module):
    def __init__(self, input, output, ker=3, stri=1, pad=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input, out_channels=output, kernel_size=ker, stride=stri, padding=pad),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)

class DeconvBlock(nn.Module):
    def __init__(self, input, output, ker=4, stri=2, pad=1):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=input, out_channels=output, kernel_size=ker, stride=stri, padding=pad),
            nn.PReLU()
        )

    def forward(self, x):
        return self.deconv(x)


class UBlock(nn.Module):
    def __init__(self, fil, ker=8, stri=4, pad=2):
        super(UBlock, self).__init__()
        self.U_Deconv1 = DeconvBlock(fil, fil, ker, stri, pad)
        self.U_Conv1 = ConvBlock(fil, fil, ker, stri, pad)
        self.U_Deconv2 = DeconvBlock(fil, fil, ker, stri, pad)

    def forward(self, x):
        x1 = self.U_Deconv1(x) # HR
        x2 = self.U_Conv1(x1) - x # LR
        x3 = self.U_Deconv2(x2) + x1 #HR

        return x3

class DenseUBlock(nn.Module):
    def __init__(self, fil, ker=8, stri=4, pad=2, stage=1):
        super(DenseUBlock, self).__init__()
        self.conv = ConvBlock((fil*stage), fil, 1, 1, 0) # concat으로 인해 filter가 늘어나는 것을 방지

        self.DU_Deconv1 = DeconvBlock(fil, fil, ker, stri, pad)
        self.DU_Conv1 = ConvBlock(fil, fil, ker, stri, pad)
        self.DU_Deconv2 = DeconvBlock(fil, fil, ker, stri, pad)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.DU_Deconv1(x1) # HR
        x3 = self.DU_Conv1(x2) - x1 # LR
        x4 = self.DU_Deconv2(x3) + x2 # HR

        return x4

class DBlock(nn.Module):
    def __init__(self, fil, ker=8, stri=4, pad=2):
        super(DBlock, self).__init__()
        self.D_Conv1 = ConvBlock(fil, fil, ker, stri, pad)
        self.D_Deconv1 = DeconvBlock(fil, fil, ker, stri, pad)
        self.D_Conv2 = ConvBlock(fil, fil, ker, stri, pad)

    def forward(self, x):
        x1 = self.D_Conv1(x)  # LR
        x2 = self.D_Deconv1(x1) - x  # HR
        x3 = self.D_Conv2(x2) + x1  # LR

        return x3

class DenseDBlock(nn.Module):
    def __init__(self, fil, ker=8, stri=4, pad=2, stage=1):
        super(DenseDBlock, self).__init__()
        self.conv = ConvBlock((fil*stage), fil, 1, 1, 0)

        self.DD_Conv1 = ConvBlock(fil, fil, ker, stri, pad)
        self.DD_Deconv1 = DeconvBlock(fil, fil, ker, stri, pad)
        self.DD_Conv2 = ConvBlock(fil, fil, ker, stri, pad)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.DD_Conv1(x1) # LR
        x3 = self.DD_Deconv1(x2) - x1 # HR
        x4 = self.DD_Conv2(x3) + x2 # LR

        return x4