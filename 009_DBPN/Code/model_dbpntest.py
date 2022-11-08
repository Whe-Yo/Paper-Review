import os
import cv2
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import gc
gc.collect()
torch.cuda.empty_cache() # 캐시 제거

from module.dbpn_module import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


################################## DBPN ##################################

class dbpn(nn.Module):
    def __init__(self, cha, fil, fea, stage):
        super(dbpn, self).__init__()

        ker = 12
        stri = 8
        pad = 2

        self.conv1 = ConvBlock(cha, fea, 3, 1, 1)

        self.conv2 = ConvBlock(fea, fil, 1, 1, 0)

        self.up1 = UBlock(fil, ker, stri, pad)

        self.down1 = DBlock(fil, ker, stri, pad)
        self.up2 = UBlock(fil, ker, stri, pad) # 과정 반복

        self.down2 = DenseDBlock(fil, ker, stri, pad, stage=2) # Stage에 따른 feature 수 줄임
        self.up3 = DenseUBlock(fil, ker, stri, pad, stage=2)
        self.down3 = DenseDBlock(fil, ker, stri, pad, stage=3)
        self.up4 = DenseUBlock(fil, ker, stri, pad, stage=3)
        self.down4 = DenseDBlock(fil, ker, stri, pad, stage=4)
        self.up5 = DenseUBlock(fil, ker, stri, pad, stage=4)
        self.down5 = DenseDBlock(fil, ker, stri, pad, stage=5)
        self.up6 = DenseUBlock(fil, ker, stri, pad, stage=5)
        self.down6 = DenseDBlock(fil, ker, stri, pad, stage=6)
        self.up7 = DenseUBlock(fil, ker, stri, pad, stage=6)

        # self.U_Deconv = DeconvBlock(fil, fil, ker, stri, pad)
        # self.U_Conv = ConvBlock(fil, fil, ker, stri, pad)
        # self.U_Deconv = DeconvBlock(fil, fil, ker, stri, pad)
        #
        # self.D_Conv = ConvBlock(fil, fil, ker, stri, pad)
        # self.D_Deconv = DeconvBlock(fil, fil, ker, stri, pad)
        # self.D_Conv = ConvBlock(fil, fil, ker, stri, pad)
        #
        # self.U_Deconv2 = DeconvBlock(fil, fil, ker, stri, pad)
        # self.U_Conv2 = ConvBlock(fil, fil, ker, stri, pad)
        # self.U_Deconv2 = DeconvBlock(fil, fil, ker, stri, pad)
        #
        #
        # self.Dense_conv1 = ConvBlock((fil*stage), fil, 1, 1, 0)
        # self.DD_Conv1 = ConvBlock(fil, fil, ker, stri, pad)
        # self.DD_Deconv1 = DeconvBlock(fil, fil, ker, stri, pad)
        # self.DD_Conv1 = ConvBlock(fil, fil, ker, stri, pad)
        #
        # self.Dense_conv2 = ConvBlock((fil*stage), fil, 1, 1, 0)
        # self.DU_Deconv1 = DeconvBlock(fil, fil, ker, stri, pad)
        # self.DU_Conv1 = ConvBlock(fil, fil, ker, stri, pad)
        # self.DU_Deconv1 = DeconvBlock(fil, fil, ker, stri, pad)
        #
        # self.Dense_conv3 = ConvBlock((fil * stage), fil, 1, 1, 0)
        # self.DD_Conv2 = ConvBlock(fil, fil, ker, stri, pad)
        # self.DD_Deconv2 = DeconvBlock(fil, fil, ker, stri, pad)
        # self.DD_Conv2 = ConvBlock(fil, fil, ker, stri, pad)
        #
        # self.Dense_conv4 = ConvBlock((fil * stage), fil, 1, 1, 0)
        # self.DU_Deconv2 = DeconvBlock(fil, fil, ker, stri, pad)
        # self.DU_Conv2 = ConvBlock(fil, fil, ker, stri, pad)
        # self.DU_Deconv2 = DeconvBlock(fil, fil, ker, stri, pad)
        #
        # self.Dense_conv5 = ConvBlock((fil * stage), fil, 1, 1, 0)
        # self.DD_Conv3 = ConvBlock(fil, fil, ker, stri, pad)
        # self.DD_Deconv3 = DeconvBlock(fil, fil, ker, stri, pad)
        # self.DD_Conv3 = ConvBlock(fil, fil, ker, stri, pad)
        #
        # self.Dense_conv6 = ConvBlock((fil * stage), fil, 1, 1, 0)
        # self.DU_Deconv3 = DeconvBlock(fil, fil, ker, stri, pad)
        # self.DU_Conv3 = ConvBlock(fil, fil, ker, stri, pad)
        # self.DU_Deconv3 = DeconvBlock(fil, fil, ker, stri, pad)
        #
        # self.Dense_conv7 = ConvBlock((fil * stage), fil, 1, 1, 0)
        # self.DD_Conv4 = ConvBlock(fil, fil, ker, stri, pad)
        # self.DD_Deconv4 = DeconvBlock(fil, fil, ker, str, pad)
        # self.DD_Conv4 = ConvBlock(fil, fil, ker, stri, pad)
        #
        # self.Dense_conv8 = ConvBlock((fil * stage), fil, 1, 1, 0)
        # self.DU_Deconv4 = DeconvBlock(fil, fil, ker, stri, pad)
        # self.DU_Conv4 = ConvBlock(fil, fil, ker, stri, pad)
        # self.DU_Deconv4 = DeconvBlock(fil, fil, ker, stri, pad)
        #
        # self.Dense_conv9 = ConvBlock((fil * stage), fil, 1, 1, 0)
        # self.DD_Conv5 = ConvBlock(fil, fil, ker, stri, pad)
        # self.DD_Deconv5 = DeconvBlock(fil, fil, ker, stri, pad)
        # self.DD_Conv5 = ConvBlock(fil, fil, ker, stri, pad)
        #
        # self.Dense_conv10 = ConvBlock((fil * stage), fil, 1, 1, 0)
        # self.DU_Deconv5 = DeconvBlock(fil, fil, ker, stri, pad)
        # self.DU_Conv5 = ConvBlock(fil, fil, ker, stri, pad)
        # self.DU_Deconv5 = DeconvBlock(fil, fil, ker, stri, pad)
        #

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=fil, out_channels=cha, kernel_size=3, stride=1, padding=1),
        )
        #
        # self.init_conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        # self.init_conv2 = nn.Conv2d(3, 64, 1, 1, 1)
        #
        # ## block1_up
        #
        # self.u_conv1 = nn.ConvTranspose2d(64, 64, kernel_size=12, stride=8, padding=2)
        # self.d_conv1 = nn.Conv2d(64, 2, 2, 2)
        # self.u_conv1_2 = nn.ConvTranspose2d(64, 64, kernel_size=12, stride=8, padding=2)

    def forward(self, x):
        # x = 64 channel
        # x1_1 = self.u_conv1(x)
        # x1_2 = self.d_conv1(x1_1) - x
        # x1_3 = self.u_conv1_2(x1_2) + x1_1
        #
        # return x1_3

        x = self.conv1(x)
        x = self.conv2(x)

        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)

        #l2
        concat_h = torch.cat((h2, h1), 1)
        l = self.down2(concat_h)

        #h3
        concat_l = torch.cat((l, l1), 1)
        h = self.up3(concat_l)

        #l3
        concat_h = torch.cat((h, concat_h), 1)
        l = self.down3(concat_h)

        #h4
        concat_l = torch.cat((l, concat_l), 1)
        h = self.up4(concat_l)

        #l4
        concat_h = torch.cat((h, concat_h), 1)
        l = self.down4(concat_h)

        #h5
        concat_l = torch.cat((l, concat_l), 1)
        h = self.up5(concat_l)

        #l5
        concat_h = torch.cat((h, concat_h), 1)
        l = self.down5(concat_h)

        #h6
        concat_l = torch.cat((l, concat_l), 1)
        h = self.up6(concat_l)

        #l6
        concat_h = torch.cat((h, concat_h), 1)
        l = self.down6(concat_h)

        #h7
        concat_l = torch.cat((l, concat_l), 1)
        h = self.up7(concat_l)

        concat_h = torch.cat((h, concat_h), 1)

        x = self.conv3(concat_h)

        return x


################################## UNet ##################################

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        # block1~5 : Contracting Path

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.ReLU(),

            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.ReLU(),

            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.ReLU(),

            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU()
        )

        # block6~9 : Extracting Path

        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.ReLU(),


            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU()
        )

        self.block7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU()
        )

        self.block8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU()
        )

        self.block9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=1, dilation=1)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):

        # block1~5 : Contracting Path
        residual = x
        block1 = self.block1(x)
        block1_pool =  self.pool(block1)
        block2 = self.block2(block1_pool)
        block2_pool = self.pool(block2)
        block3 = self.block3(block2_pool)
        block3_pool = self.pool(block3)
        block4 = self.block4(block3_pool)
        block4_pool = self.pool(block4)
        block5 = self.block5(block4_pool)
        # block1_pool = self.pool(block1)

        # block6~9 : Extracting Path

        ublock5 = block5
        diffX = block4.size()[2] - ublock5.size()[2]
        diffY = block4.size()[3] - ublock5.size()[3]
        ublock5 = F.pad(ublock5, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
        x = torch.cat([block4, ublock5], dim=1)
        block6 = self.block6(x)

        ublock6 = block6
        diffX = block3.size()[2] - ublock6.size()[2]
        diffY = block3.size()[3] - ublock6.size()[3]
        ublock6 = F.pad(ublock6, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
        x = torch.cat([block3, ublock6], dim=1)
        block7 = self.block7(x)

        ublock7 = block7
        diffX = block2.size()[2] - ublock7.size()[2]
        diffY = block2.size()[3] - ublock7.size()[3]
        ublock7 = F.pad(ublock7, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        x = torch.cat([block2, ublock7], dim=1)
        block8 = self.block8(x)

        ublock8 = block8
        diffX = block1.size()[2] - ublock8.size()[2]
        diffY = block1.size()[3] - ublock8.size()[3]
        ublock8 = F.pad(ublock8, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
        x = torch.cat([block1, ublock8], dim=1)
        block9 = self.block9(x)

        return block9

################################## SRGAN ##################################

class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
    
###########################################################################
    
if __name__== "__main__":

################################## DBPN ##################################

    img = cv2.imread('img/input/Lenna.png').astype(np.float32) / 255.
    net = dbpn(cha=3, fil=64, fea=256, stage=7)
    tensor_ = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0)
    out = net(tensor_)
    print(tensor_.size())
    print(out.size())
    
################################## UNet ##################################
    
    # img = cv2.imread('img/input/Lenna.png').astype(np.float32) / 255.
    # net = Unet()
    # tensor_ = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0)
    # out = net(tensor_)
    # print(tensor_.size())
    # print(out.size())

###########################################################################