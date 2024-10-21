#https://github.com/berniwal/swin-transformer-pytorch/blob/master/swin_transformer_pytorch/swin_transformer.py

from .aerialformer_parts import *
from torchvision.models import swin_t
import torch.nn as nn
import torch


class AerialFormer(nn.Module):
    def __init__(self, n_channels=3, embed_dims=96, num_classes=2, pretrained=False):
        super(AerialFormer, self).__init__()
        self.n_channels = n_channels

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.swin = swin_t(num_classes=1000, weights='IMAGENET1K_V1' if pretrained else None)

        self.cnn_stem = CNNStem(n_channels, inplanes=64, embed_dims=embed_dims)

        self.up1 = nn.Sequential(MDCBlock(channels=8*embed_dims, dilatations=(1,2,3), kernel_size=(3,3,3), padding=(1,2,3)), DeconvBlock(8*embed_dims, 4*embed_dims))
        self.up2 = nn.Sequential(MDCBlock(channels=8*embed_dims, dilatations=(1,2,3), kernel_size=(3,3,3), padding=(1,2,3)), DeconvBlock(8*embed_dims, 2*embed_dims))
        self.up3 = nn.Sequential(MDCBlock(channels=4*embed_dims, dilatations=(1,2,3), kernel_size=(3,3,3), padding=(1,2,3)), DeconvBlock(4*embed_dims, 1*embed_dims))
        self.up4 = nn.Sequential(MDCBlock(channels=2*embed_dims, dilatations=(1,1,1), kernel_size=(3,3,3), padding=(1,1,1)), DeconvBlock(2*embed_dims, embed_dims//2))
        self.up5 = nn.Sequential(MDCBlock(channels=1*embed_dims, dilatations=(1,1,1), kernel_size=(1,3,3), padding=(0,1,1)))

        self.down1 = nn.Sequential(self.swin.features[-8], self.swin.features[-7])
        self.down2 = nn.Sequential(self.swin.features[-6], self.swin.features[-5])
        self.down3 = nn.Sequential(self.swin.features[-4], self.swin.features[-3])
        self.down4 = nn.Sequential(self.swin.features[-2], self.swin.features[-1])

        self.outDeconv = DeconvBlock(1*embed_dims, 1*embed_dims)
        self.outConv = nn.Conv2d(1*embed_dims, num_classes, kernel_size=1)


    def forward(self, x):

        x5l = self.cnn_stem(x)

        x4l = self.down1(x)
        x3l = self.down2(x4l)
        x2l = self.down3(x3l)
        x1l = self.down4(x2l)

        x1r = self.up1(x1l.permute((0, 3, 1, 2)))
        x2r = self.up2(torch.cat((x2l.permute((0, 3, 1, 2)), x1r), dim=1))
        x3r = self.up3(torch.cat((x3l.permute((0, 3, 1, 2)), x2r), dim=1))
        x4r = self.up4(torch.cat((x4l.permute((0, 3, 1, 2)), x3r), dim=1))
        x5r = self.up5(torch.cat((x5l, x4r), dim=1))

        return self.outConv(self.outDeconv(x5r))

    def for_inference(self, x):
        return nn.Sigmoid()(x)

    def __str__(self):
        return "aerialformer_pretrained="+str(self.pretrained)
