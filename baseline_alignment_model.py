import torch
import torch.nn as nn

from resnet_encoder import SharedResNet50
from decoder import ChangeDetectionDecoder


class BaselineFeatureAlignmentChangeDetector(nn.Module):

    def __init__(self):

        super().__init__()

        # ====================================================
        # Shared Siamese ResNet-50 Encoder
        # ====================================================

        self.encoder = SharedResNet50(
            pretrained=True
        )

        # ====================================================
        # Deep Feature Projection
        #
        # ResNet-50 feature4:
        # 2048 channels -> 256 channels
        #
        # ChangeDetectionDecoder expects 256 channels
        # for the deepest input feature.
        # ====================================================

        self.deep_projection = nn.Sequential(

            nn.Conv2d(
                2048,
                256,
                kernel_size=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True)
        )

        # ====================================================
        # Change Detection Decoder
        # ====================================================

        self.decoder = ChangeDetectionDecoder()


    def forward(self, image1, image2):

        # ====================================================
        # Extract multi-scale features
        # ====================================================

        features1 = self.encoder(image1)

        features2 = self.encoder(image2)


        # ====================================================
        # Temporal feature differences
        # ====================================================

        difference1 = torch.abs(
            features1["feature1"]
            -
            features2["feature1"]
        )


        difference2 = torch.abs(
            features1["feature2"]
            -
            features2["feature2"]
        )


        difference3 = torch.abs(
            features1["feature3"]
            -
            features2["feature3"]
        )


        difference4 = torch.abs(
            features1["feature4"]
            -
            features2["feature4"]
        )


        # ====================================================
        # Project deepest temporal difference
        # ====================================================

        deep_difference = self.deep_projection(
            difference4
        )


        # ====================================================
        # Change Detection
        # ====================================================

        change_logits = self.decoder(
            deep_difference,
            difference1,
            difference2,
            difference3
        )


        # ====================================================
        # Return change output + features
        #
        # These features are required by the feature
        # alignment loss during training.
        # ====================================================

        return {

            "change_logits":
                change_logits,

            "feature1_image1":
                features1["feature1"],

            "feature1_image2":
                features2["feature1"],

            "feature2_image1":
                features1["feature2"],

            "feature2_image2":
                features2["feature2"],

            "feature3_image1":
                features1["feature3"],

            "feature3_image2":
                features2["feature3"],

            "feature4_image1":
                features1["feature4"],

            "feature4_image2":
                features2["feature4"]
        }