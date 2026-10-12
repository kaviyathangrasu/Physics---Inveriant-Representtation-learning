import torch
import torch.nn as nn

from resnet_encoder import SharedResNet50
from dual_head import DualHeadRepresentation
from attention_fusion import AttentionFusion
from decoder import ChangeDetectionDecoder


class CompleteChangeDetectionModel(nn.Module):

    def __init__(self, pretrained=True):

        super(CompleteChangeDetectionModel, self).__init__()


        # =====================================================
        # 1. SHARED RESNET-50 ENCODER
        # =====================================================

        self.encoder = SharedResNet50(
            pretrained=pretrained
        )


        # =====================================================
        # 2. DUAL REPRESENTATION HEAD
        #
        # Earth representation
        # Nuisance representation
        # =====================================================

        self.dual_head = DualHeadRepresentation(
            in_channels=2048,
            feature_dim=256
        )


        # =====================================================
        # 3. ATTENTION-GUIDED TEMPORAL FUSION
        # =====================================================

        self.attention_fusion = AttentionFusion(
            feature_dim=256
        )


        # =====================================================
        # 4. TEMPORALLY AWARE DECODER
        # =====================================================

        self.decoder = ChangeDetectionDecoder()


    def forward(
        self,
        image1,
        image2
    ):

        # =====================================================
        # STEP 1
        # Shared encoder for Image 1
        # =====================================================

        features1 = self.encoder(
            image1
        )


        # =====================================================
        # STEP 2
        # Shared encoder for Image 2
        # =====================================================

        features2 = self.encoder(
            image2
        )


        # =====================================================
        # STEP 3
        # Extract Image 1 hierarchical features
        # =====================================================

        feature1 = features1[
            "feature1"
        ]

        feature2 = features1[
            "feature2"
        ]

        feature3 = features1[
            "feature3"
        ]

        feature4 = features1[
            "feature4"
        ]


        # =====================================================
        # STEP 4
        # Extract Image 2 hierarchical features
        # =====================================================

        feature1_t2 = features2[
            "feature1"
        ]

        feature2_t2 = features2[
            "feature2"
        ]

        feature3_t2 = features2[
            "feature3"
        ]

        feature4_t2 = features2[
            "feature4"
        ]


        # =====================================================
        # STEP 5
        # Earth + nuisance representations
        # =====================================================

        earth1, nuisance1 = self.dual_head(
            feature4
        )


        earth2, nuisance2 = self.dual_head(
            feature4_t2
        )


        # =====================================================
        # STEP 6
        # Earth representation difference
        # =====================================================

        earth_difference = torch.abs(
            earth1 - earth2
        )


        # =====================================================
        # STEP 7
        # Attention-guided fusion
        # =====================================================

        fused_feature, attention_map = (
            self.attention_fusion(
                earth1,
                earth2
            )
        )


        # =====================================================
        # STEP 8
        # MULTI-SCALE TEMPORAL SKIP CONNECTIONS
        #
        # Absolute difference between corresponding
        # hierarchical features from the two dates.
        # =====================================================

        feature1_difference = torch.abs(
            feature1 - feature1_t2
        )

        feature2_difference = torch.abs(
            feature2 - feature2_t2
        )

        feature3_difference = torch.abs(
            feature3 - feature3_t2
        )


        # =====================================================
        # STEP 9
        # Temporally aware decoder
        # =====================================================

        change_logits = self.decoder(
            fused_feature,
            feature1_difference,
            feature2_difference,
            feature3_difference
        )


        # =====================================================
        # STEP 10
        # RETURN OUTPUTS
        # =====================================================

        return {

            "change_logits": change_logits,

            "earth1": earth1,

            "earth2": earth2,

            "nuisance1": nuisance1,

            "nuisance2": nuisance2,

            "earth_difference": earth_difference,

            "fused_feature": fused_feature,

            "attention_map": attention_map,

            "temporal_feature1_difference":
                feature1_difference,

            "temporal_feature2_difference":
                feature2_difference,

            "temporal_feature3_difference":
                feature3_difference,

            "encoder_features1": features1,

            "encoder_features2": features2
        }