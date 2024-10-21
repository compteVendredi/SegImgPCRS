import torch
from torchgeo.models import FarSeg

class CustomFarSeg(torch.nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(CustomFarSeg, self).__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.farseg = FarSeg(backbone='resnet50', classes=num_classes, backbone_pretrained=pretrained)

    def forward(self, x):
        return self.farseg(x)

    def for_inference(self, x):
        return torch.nn.Sigmoid()(x)

    def __str__(self):
        return "FarSeg_pretrained="+str(self.pretrained)

