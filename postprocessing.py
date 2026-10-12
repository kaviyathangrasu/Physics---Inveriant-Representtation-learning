import torch


class ChangePostProcessor:

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def forward(self, logits):

        # Convert logits to probability
        probability = torch.sigmoid(logits)

        # Convert probability to binary mask
        binary_mask = (probability >= self.threshold).float()

        return probability, binary_mask

    def calculate_change_ratio(self, binary_mask):

        # Number of changed pixels
        changed_pixels = binary_mask.sum(dim=(1, 2, 3))

        # Total pixels
        total_pixels = (
            binary_mask.shape[2] *
            binary_mask.shape[3]
        )

        # Percentage of changed area
        change_ratio = (
            changed_pixels / total_pixels
        ) * 100

        return change_ratio