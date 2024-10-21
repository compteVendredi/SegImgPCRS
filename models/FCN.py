#https://pytorch.org/vision/main/_modules/torchvision/models/segmentation/deeplabv3.html#DeepLabV3_ResNet50_Weights

import torch
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.segmentation.fcn import FCNHead, _fcn_resnet
from torchvision.models.resnet import resnet50, ResNet50_Weights

class CustomFCN(torch.nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(CustomFCN, self).__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.fcn = fcn_resnet50(num_classes=21, weights=None, weights_backbone=None)
        if pretrained:
            self.fcn = _fcn_resnet(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, replace_stride_with_dilation=[False, True, True]), 21, None)
        self.fcn.classifier = FCNHead(2048, 2)


    def forward(self, x):
        return self.fcn(x)['out']

    def for_inference(self, x):
        return torch.nn.Sigmoid()(x)

    def __str__(self):
        return "FCN_pretrained="+str(self.pretrained)
