#https://pytorch.org/vision/main/_modules/torchvision/models/segmentation/deeplabv3.html#deeplabv3_resnet50

import torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, ASPP
import torch.nn as nn

class CustomDeepLabV3(torch.nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(CustomDeepLabV3, self).__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.deeplab = deeplabv3_resnet50(num_classes=21, weights=DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None, weights_backbone=None)
        self.deeplab.classifier = DeepLabHead(2048, num_classes)

    def forward(self, x):
        return self.deeplab(x)['out']

    def for_inference(self, x):
        return torch.nn.Sigmoid()(x)

    def __str__(self):
        return "Deeplabv3_pretrained="+str(self.pretrained)
