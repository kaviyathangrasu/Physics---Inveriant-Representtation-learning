import torch
import torch.nn as nn
from resnet_encoder import SharedResNet50
from decoder import ChangeDetectionDecoder

class BaselineSiameseChangeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SharedResNet50(pretrained=True)
        self.projection = nn.Sequential(
            nn.Conv2d(2048, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decoder = ChangeDetectionDecoder()

    def forward(self, image1, image2):
        f1 = self.encoder(image1)
        f2 = self.encoder(image2)

        d1 = torch.abs(f1["feature1"] - f2["feature1"])
        d2 = torch.abs(f1["feature2"] - f2["feature2"])
        d3 = torch.abs(f1["feature3"] - f2["feature3"])
        d4 = torch.abs(f1["feature4"] - f2["feature4"])

        deep = self.projection(d4)

        logits = self.decoder(deep, d1, d2, d3)

        return {"change_logits": logits}
