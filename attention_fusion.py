import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# ATTENTION FUSION MODULE
# ==========================================

class AttentionFusion(nn.Module):

    def __init__(self, feature_dim=256):

        super(AttentionFusion, self).__init__()

        # Learn attention from the temporal difference
        self.attention = nn.Sequential(

            nn.Conv2d(
                feature_dim,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                feature_dim,
                kernel_size=1
            ),

            nn.Sigmoid()
        )


    def forward(
        self,
        feature1,
        feature2
    ):

        # --------------------------------------
        # Temporal feature difference
        # --------------------------------------

        difference = torch.abs(
            feature1 - feature2
        )


        # --------------------------------------
        # Generate attention map
        # --------------------------------------

        attention_map = self.attention(
            difference
        )


        # --------------------------------------
        # Fuse features
        # --------------------------------------

        fused_feature = (
            attention_map * difference
        )


        return fused_feature, attention_map