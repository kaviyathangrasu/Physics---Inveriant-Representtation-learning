import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class SharedResNet50(nn.Module):

    def __init__(self, pretrained=True):
        super(SharedResNet50, self).__init__()

        if pretrained:
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None

        backbone = resnet50(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.out_channels = 2048

    def forward(self, x):

        # ------------------------------------------
        # Initial features
        # ------------------------------------------

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        # ------------------------------------------
        # ResNet feature hierarchy
        # ------------------------------------------

        feature1 = self.layer1(x)   # 64 × 64
        feature2 = self.layer2(feature1)  # 32 × 32
        feature3 = self.layer3(feature2)  # 16 × 16
        feature4 = self.layer4(feature3)  # 8 × 8

        return {
            "feature1": feature1,
            "feature2": feature2,
            "feature3": feature3,
            "feature4": feature4
        }