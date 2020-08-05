import torch
import torch.nn as nn
import torch.nn.functional as F

from unet_parts import OutConv, DoubleConv, Down, Up

class UNet(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(UNet, self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.inc = OutConv(channels_in, 8, relu=True)
        self.conv1 = DoubleConv(8, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 128)

        self.up1 = Up(128, 64)
        self.up2 = Up(64, 32)
        self.up3 = Up(32, 32)
        self.conv2 = OutConv(32, 16, relu=True)
        self.conv3 = OutConv(16, 1, relu=False)
    
    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.conv1(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
            nn.LeakyReLU(0.2), 

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=256, out_channels=4, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(125*125, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64,1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return self.net(x).view(batch_size)
