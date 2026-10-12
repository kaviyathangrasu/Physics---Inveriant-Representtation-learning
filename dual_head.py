import torch
import torch.nn as nn
import torch.nn.functional as F


class EarthHead(nn.Module):
    def __init__(self, in_channels=2048, feature_dim=256):
        super(EarthHead, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.network(x)


class NuisanceHead(nn.Module):
    def __init__(self, in_channels=2048, feature_dim=256):
        super(NuisanceHead, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.network(x)


class DualHeadRepresentation(nn.Module):
    def __init__(self, in_channels=2048, feature_dim=256):
        super(DualHeadRepresentation, self).__init__()

        self.earth_head = EarthHead(
            in_channels,
            feature_dim
        )

        self.nuisance_head = NuisanceHead(
            in_channels,
            feature_dim
        )

    def forward(self, x):

        earth_feature = self.earth_head(x)
        nuisance_feature = self.nuisance_head(x)

        return earth_feature, nuisance_feature


class DisentanglementLoss(nn.Module):
    """
    Encourages Earth and Nuisance representations
    to contain less redundant information.

    This is implemented as a cross-correlation
    / orthogonality constraint.
    """

    def __init__(self):
        super(DisentanglementLoss, self).__init__()

    def forward(self, earth_feature, nuisance_feature):

        # Normalize along feature dimension
        earth = F.normalize(
            earth_feature,
            p=2,
            dim=1
        )

        nuisance = F.normalize(
            nuisance_feature,
            p=2,
            dim=1
        )

        # Correlation between Earth and Nuisance
        correlation = (earth * nuisance).sum(
            dim=1
        )

        # Penalize correlation
        loss = correlation.pow(2).mean()

        return loss