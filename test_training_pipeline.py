import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import S2LookingDataset
from complete_model import CompleteChangeDetectionModel
from physics_loss import TemporalConsistencyLoss
from feature_alignment import FeatureAlignmentLoss
from dual_head import DisentanglementLoss


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# 2. DATASET
# ============================================================

DATASET_PATH = r"K:\S2Looking"

dataset = S2LookingDataset(
    base_path=DATASET_PATH,
    split="train",
    image_size=256
)

print("\nDataset length:", len(dataset))


# ============================================================
# 3. DATALOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

print("DataLoader created successfully.")


# ============================================================
# 4. GET ONE REAL BATCH
# ============================================================

image1, image2, label = next(iter(loader))

image1 = image1.to(device)
image2 = image2.to(device)
label = label.to(device)

print("\nReal S2Looking batch:")
print("Image1 :", image1.shape)
print("Image2 :", image2.shape)
print("Label  :", label.shape)
print("Label values:", torch.unique(label))


# ============================================================
# 5. CREATE MODEL
# ============================================================

model = CompleteChangeDetectionModel(
    pretrained=True
)

model = model.to(device)

model.train()

print("\nComplete model created successfully.")


# ============================================================
# 6. DEFINE LOSSES
# ============================================================

# ------------------------------------------------------------
# Change detection loss
# Weighted BCE + Dice
# ------------------------------------------------------------

class WeightedBCEDiceLoss(nn.Module):

    def __init__(
        self,
        positive_weight=4.0,
        dice_weight=1.0
    ):

        super().__init__()

        self.positive_weight = positive_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):

        # Weighted BCE
        pos_weight = torch.tensor(
            self.positive_weight,
            device=logits.device
        )

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight
        )

        # Convert logits to probability
        probability = torch.sigmoid(logits)

        # Flatten spatial dimensions
        probability_flat = probability.view(
            probability.size(0),
            -1
        )

        target_flat = target.view(
            target.size(0),
            -1
        )

        # Dice coefficient
        intersection = (
            probability_flat * target_flat
        ).sum(dim=1)

        denominator = (
            probability_flat.sum(dim=1)
            + target_flat.sum(dim=1)
            + 1e-8
        )

        dice = (
            2.0 * intersection
            + 1e-8
        ) / denominator

        dice_loss = 1.0 - dice.mean()

        # Combined loss
        total_loss = (
            bce
            + self.dice_weight * dice_loss
        )

        return total_loss


change_loss_fn = WeightedBCEDiceLoss(
    positive_weight=4.0,
    dice_weight=1.0
)


# ------------------------------------------------------------
# Temporal consistency
# ------------------------------------------------------------

temporal_loss_fn = TemporalConsistencyLoss()


# ------------------------------------------------------------
# Feature alignment
# ------------------------------------------------------------

alignment_loss_fn = FeatureAlignmentLoss()


# ------------------------------------------------------------
# Disentanglement
# ------------------------------------------------------------

disentanglement_loss_fn = DisentanglementLoss()


print("\nAll loss functions created successfully.")


# ============================================================
# 7. FORWARD PASS
# ============================================================

print("\nRunning forward pass...")

outputs = model(
    image1,
    image2
)

print("Forward pass successful.")


# ============================================================
# 8. EXTRACT OUTPUTS
# ============================================================

change_logits = outputs[
    "change_logits"
]

earth1 = outputs[
    "earth1"
]

earth2 = outputs[
    "earth2"
]

nuisance1 = outputs[
    "nuisance1"
]

nuisance2 = outputs[
    "nuisance2"
]


# ============================================================
# 9. EXTRACT EARTH FEATURE DIFFERENCE
# ============================================================

earth_difference = outputs[
    "earth_difference"
]


# ============================================================
# 10. CALCULATE CHANGE LOSS
# ============================================================

change_loss = change_loss_fn(
    change_logits,
    label
)


# ============================================================
# 11. CALCULATE TEMPORAL CONSISTENCY LOSS
# ============================================================

temporal_loss = temporal_loss_fn(
    earth1,
    earth2,
    label
)


# ============================================================
# 12. CALCULATE FEATURE ALIGNMENT LOSS
# ============================================================

alignment_loss = alignment_loss_fn(
    earth1,
    earth2,
    label
)


# ============================================================
# 13. CALCULATE DISENTANGLEMENT LOSS
# ============================================================

disentanglement_loss = disentanglement_loss_fn(
    earth1,
    nuisance1
)


# ============================================================
# 14. LOSS WEIGHTS
# ============================================================

TEMPORAL_WEIGHT = 0.1
ALIGNMENT_WEIGHT = 0.1
DISENTANGLEMENT_WEIGHT = 0.1


# ============================================================
# 15. TOTAL LOSS
# ============================================================

total_loss = (
    change_loss
    + TEMPORAL_WEIGHT * temporal_loss
    + ALIGNMENT_WEIGHT * alignment_loss
    + DISENTANGLEMENT_WEIGHT * disentanglement_loss
)


# ============================================================
# 16. PRINT LOSSES
# ============================================================

print("\n======================================")
print("LOSS VALUES")
print("======================================")

print(
    f"Change Loss          : "
    f"{change_loss.item():.6f}"
)

print(
    f"Temporal Loss        : "
    f"{temporal_loss.item():.6f}"
)

print(
    f"Alignment Loss       : "
    f"{alignment_loss.item():.6f}"
)

print(
    f"Disentanglement Loss : "
    f"{disentanglement_loss.item():.6f}"
)

print(
    f"Total Loss           : "
    f"{total_loss.item():.6f}"
)


# ============================================================
# 17. CHECK LOSS VALUES
# ============================================================

if not torch.isfinite(total_loss):

    raise RuntimeError(
        "ERROR: Total loss contains NaN or Infinity."
    )

print("\nLoss values are finite.")


# ============================================================
# 18. CREATE OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

print("Optimizer created successfully.")


# ============================================================
# 19. BACKWARD PASS
# ============================================================

print("\nRunning backward pass...")

optimizer.zero_grad()

total_loss.backward()

print("Backward pass successful.")


# ============================================================
# 20. CHECK GRADIENTS
# ============================================================

gradient_count = 0
invalid_gradient = False

for name, parameter in model.named_parameters():

    if parameter.grad is not None:

        gradient_count += 1

        if not torch.isfinite(
            parameter.grad
        ).all():

            print(
                f"ERROR: Invalid gradient in {name}"
            )

            invalid_gradient = True


print(
    "\nParameters with gradients:",
    gradient_count
)


if invalid_gradient:

    raise RuntimeError(
        "ERROR: Invalid gradients detected."
    )


# ============================================================
# 21. OPTIMIZER STEP
# ============================================================

optimizer.step()

print("Optimizer step successful.")


# ============================================================
# 22. FINAL RESULT
# ============================================================

print("\n======================================")

if (
    torch.isfinite(total_loss)
    and gradient_count > 0
    and not invalid_gradient
):

    print(
        "     TRAINING PIPELINE TEST PASSED"
    )

else:

    print(
        "     TRAINING PIPELINE TEST FAILED"
    )

print("======================================")