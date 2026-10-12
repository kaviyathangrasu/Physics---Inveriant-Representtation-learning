import torch
import torch.nn as nn
import torch.nn.functional as F


class ChangeDetectionDecoder(nn.Module):

    def __init__(self):

        super(ChangeDetectionDecoder, self).__init__()

        # =====================================================
        # TEMPORAL SKIP 3
        # Input:
        # fused high-level feature = 256 channels
        # feature3 temporal difference = 1024 channels
        #
        # Output = 256 channels
        # =====================================================

        self.block1 = nn.Sequential(

            nn.Conv2d(
                256 + 1024,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True)
        )


        # =====================================================
        # TEMPORAL SKIP 2
        #
        # Input:
        # decoder = 256 channels
        # feature2 temporal difference = 512 channels
        #
        # Output = 128 channels
        # =====================================================

        self.block2 = nn.Sequential(

            nn.Conv2d(
                256 + 512,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True)
        )


        # =====================================================
        # TEMPORAL SKIP 1
        #
        # Input:
        # decoder = 128 channels
        # feature1 temporal difference = 256 channels
        #
        # Output = 64 channels
        # =====================================================

        self.block3 = nn.Sequential(

            nn.Conv2d(
                128 + 256,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True)
        )


        # =====================================================
        # FINAL UPSAMPLING BLOCK 1
        # =====================================================

        self.block4 = nn.Sequential(

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True)
        )


        # =====================================================
        # FINAL UPSAMPLING BLOCK 2
        # =====================================================

        self.block5 = nn.Sequential(

            nn.Conv2d(
                32,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(inplace=True)
        )


        # =====================================================
        # FINAL CHANGE DETECTION OUTPUT
        # =====================================================

        self.output_layer = nn.Conv2d(
            16,
            1,
            kernel_size=1
        )


    def forward(
        self,
        fused_feature,
        feature1_difference,
        feature2_difference,
        feature3_difference
    ):

        # =====================================================
        # STEP 1
        # Upsample fused 8 × 8 feature
        # to feature3 resolution: 16 × 16
        # =====================================================

        x = F.interpolate(
            fused_feature,
            size=feature3_difference.shape[-2:],
            mode="bilinear",
            align_corners=False
        )


        # =====================================================
        # STEP 2
        # Temporal skip from feature3
        # =====================================================

        x = torch.cat(
            [
                x,
                feature3_difference
            ],
            dim=1
        )


        x = self.block1(x)


        # =====================================================
        # STEP 3
        # Upsample to feature2 resolution
        # 16 × 16 → 32 × 32
        # =====================================================

        x = F.interpolate(
            x,
            size=feature2_difference.shape[-2:],
            mode="bilinear",
            align_corners=False
        )


        # =====================================================
        # STEP 4
        # Temporal skip from feature2
        # =====================================================

        x = torch.cat(
            [
                x,
                feature2_difference
            ],
            dim=1
        )


        x = self.block2(x)


        # =====================================================
        # STEP 5
        # Upsample to feature1 resolution
        # 32 × 32 → 64 × 64
        # =====================================================

        x = F.interpolate(
            x,
            size=feature1_difference.shape[-2:],
            mode="bilinear",
            align_corners=False
        )


        # =====================================================
        # STEP 6
        # Temporal skip from feature1
        # =====================================================

        x = torch.cat(
            [
                x,
                feature1_difference
            ],
            dim=1
        )


        x = self.block3(x)


        # =====================================================
        # STEP 7
        # 64 × 64 → 128 × 128
        # =====================================================

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )


        x = self.block4(x)


        # =====================================================
        # STEP 8
        # 128 × 128 → 256 × 256
        # =====================================================

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )


        x = self.block5(x)


        # =====================================================
        # STEP 9
        # Final change logits
        # =====================================================

        change_logits = self.output_layer(x)


        return change_logits