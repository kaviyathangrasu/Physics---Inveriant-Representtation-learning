import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAlignmentLoss(nn.Module):
    """
    Aligns Earth representations from the two temporal images
    in regions where no ground change has occurred.

    Cosine similarity is used after L2 normalization.
    """

    def __init__(self):
        super(FeatureAlignmentLoss, self).__init__()

    def forward(self, feature1, feature2, change_mask):

        # Resize change mask to feature resolution
        mask = F.interpolate(
            change_mask,
            size=feature1.shape[-2:],
            mode="nearest"
        )

        # Unchanged regions
        unchanged_mask = 1.0 - mask

        # Normalize feature vectors
        feature1 = F.normalize(feature1, p=2, dim=1)
        feature2 = F.normalize(feature2, p=2, dim=1)

        # Cosine similarity
        similarity = (feature1 * feature2).sum(
            dim=1,
            keepdim=True
        )

        # Alignment error
        alignment_error = 1.0 - similarity

        # Only unchanged regions contribute
        alignment_error = alignment_error * unchanged_mask

        # Normalize by number of unchanged spatial locations
        loss = alignment_error.sum() / (
            unchanged_mask.sum() + 1e-8
        )

        return loss