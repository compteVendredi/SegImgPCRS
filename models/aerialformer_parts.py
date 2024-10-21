import torch
import torch.nn as nn



class CNNStem(nn.Module):
    def __init__(self, in_channels, embed_dims=96, inplanes=64):
        super(CNNStem, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, inplanes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(inplanes),
            nn.GELU(),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inplanes),
            nn.GELU(),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inplanes),
            nn.GELU(),
            nn.Conv2d(inplanes, embed_dims//2, kernel_size=1, stride=1),
            nn.BatchNorm2d(embed_dims//2),
            nn.GELU(),
        )

    def forward(self, x):
        return self.model(x)


class MDCBlock(nn.Module):
    def __init__(self, channels=96, dilatations=(1,2,3), kernel_size=(3,3,3), padding=(1,2,3)):
        super(MDCBlock, self).__init__()
        self.pre_channel = nn.Conv2d(channels, channels, kernel_size=1, stride=1)

        self.DCL3 = nn.Conv2d(channels//3, channels//3, kernel_size=kernel_size[0], stride=1, dilation=dilatations[0], padding=padding[0])
        self.DCL5 = nn.Conv2d(channels//3, channels//3, kernel_size=kernel_size[1], stride=1, dilation=dilatations[1], padding=padding[1])
        self.DCL7 = nn.Conv2d(channels//3, channels//3, kernel_size=kernel_size[2], stride=1, dilation=dilatations[2], padding=padding[2])

        self.post_channel = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),

            )

    def forward(self, x):
        x = self.pre_channel(x)
        x1 = x[:,:x.shape[1]//3, :, :]
        x2 = x[:,x.shape[1]//3:x.shape[1]//3*2, :, :]
        x3 = x[:,x.shape[1]//3*2:, :, :]

        x = torch.cat(
            (self.DCL3(x1), self.DCL5(x2), self.DCL7(x3)),
            dim=1
            )
        return self.post_channel(x)




class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.model = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )


    def forward(self, x):
        return self.model(x)
