import math
import torch
from torch import nn
import torch.nn.functional as F

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
if __name__== "__main__":
    import cv2
    import torch
    import numpy as np
    img = cv2.imread('Lenna.png').astype(np.float32) / 255.
    net = Unet()
    tensor_ = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0)
    out = net(tensor_)
    print(tensor_.size())
    print(out.size())