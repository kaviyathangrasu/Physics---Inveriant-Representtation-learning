import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConsistencyLoss(nn.Module):
    """
    Encourages the Earth representation to remain consistent
    in regions where no actual ground change has occurred.
    """

    def __init__(self):
        super(TemporalConsistencyLoss, self).__init__()

    def forward(self, feature1, feature2, change_mask):

        # Resize change mask to feature resolution
        mask = F.interpolate(
            change_mask,
            size=feature1.shape[-2:],
            mode="nearest"
        )

        # Unchanged regions = 1
        unchanged_mask = 1.0 - mask

        # Feature difference
        difference = torch.abs(feature1 - feature2)

        # Keep only unchanged regions
        consistency_difference = difference * unchanged_mask

        # Normalize using number of unchanged pixels
        denominator = (
            unchanged_mask.sum() * feature1.shape[1]
            + 1e-8
        )

        loss = consistency_difference.sum() / denominator

        return loss