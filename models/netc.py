#https://github.com/roeez/CalcificationDetection/blob/main/core.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBlock, self).__init__()
        assert out_channels%4==0
        self.conv3 = nn.Conv2d(in_channels, out_channels//4, 3, padding=3//2)
        self.conv5 = nn.Conv2d(in_channels, out_channels//4, 5, padding=5//2)
        self.conv7 = nn.Conv2d(in_channels, out_channels//4, 7, padding=7//2)
        self.conv9 = nn.Conv2d(in_channels, out_channels//4, 9, padding=9//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([self.conv3(x), self.conv5(x), self.conv7(x), self.conv9(x)], 1)))

class NetC(nn.Module):

    def __init__(self, kernel_size=9, skip_connections=True, batch_norm=True, kernel_depth_seed=4, network_depth=4, act_func=nn.ReLU(),
                 initializer=None):
        super(NetC, self).__init__()
        self.block1 = CBlock(1*3, 4)
        self.block2 = CBlock(4, 16)
        self.block3 = CBlock(16, 32)
        self.block4 = CBlock(32, 64)
        self.block5 = CBlock(64, 128)
        self.pred = nn.Conv2d(128, 2, 5, padding=5//2)
        self.example_input_array = torch.rand(4, 3, 512, 512)

    def forward(self, x):
        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.block5(x)

        return self.pred(x)

    def for_inference(self, x):
        return nn.Sigmoid()(x)

    def __str__(self):
        return "NetC"
