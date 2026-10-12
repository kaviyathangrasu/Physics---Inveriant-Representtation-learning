import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import S2LookingDataset
from complete_model import CompleteChangeDetectionModel
from physics_loss import TemporalConsistencyLoss
from feature_alignment import FeatureAlignmentLoss
from dual_head import DisentanglementLoss


# ============================================================
# CONFIGURATION
# ============================================================

BASE_PATH = r"K:\S2Looking"

IMAGE_SIZE = 256
BATCH_SIZE = 2

NUM_SAMPLES = 10
EPOCHS = 10

LEARNING_RATE = 1e-4

LAMBDA_TEMPORAL = 0.1
LAMBDA_ALIGNMENT = 0.1
LAMBDA_DISENTANGLE = 0.1

POSITIVE_WEIGHT = 4.0
DICE_WEIGHT = 1.0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("======================================")
print("        10-SAMPLE OVERFIT TEST")
print("======================================")
print("Device :", device)


# ============================================================
# DATASET
# ============================================================

dataset = S2LookingDataset(
    base_path=BASE_PATH,
    split="train",
    image_size=IMAGE_SIZE
)

dataset = Subset(
    dataset,
    range(NUM_SAMPLES)
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print("Samples :", len(dataset))


# ============================================================
# MODEL
# ============================================================

model = CompleteChangeDetectionModel(
    pretrained=True
)

model = model.to(device)

print("Model loaded successfully.")


# ============================================================
# CHANGE LOSS
# ============================================================

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

        pos_weight = torch.tensor(
            self.positive_weight,
            device=logits.device
        )

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight
        )

        probabilities = torch.sigmoid(logits)

        smooth = 1e-6

        intersection = (
            probabilities * target
        ).sum(dim=(1, 2, 3))

        denominator = (
            probabilities.sum(dim=(1, 2, 3))
            + target.sum(dim=(1, 2, 3))
        )

        dice = (
            2.0 * intersection + smooth
        ) / (
            denominator + smooth
        )

        dice_loss = 1.0 - dice.mean()

        return (
            bce
            + self.dice_weight * dice_loss
        )


# ============================================================
# LOSSES
# ============================================================

change_loss_fn = WeightedBCEDiceLoss(
    positive_weight=POSITIVE_WEIGHT,
    dice_weight=DICE_WEIGHT
)

temporal_loss_fn = TemporalConsistencyLoss()

alignment_loss_fn = FeatureAlignmentLoss()

disentangle_loss_fn = DisentanglementLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    total_epoch_loss = 0.0

    for image1, image2, label in loader:

        image1 = image1.to(device)
        image2 = image2.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        outputs = model(
            image1,
            image2
        )

        change_logits = outputs["change_logits"]

        earth1 = outputs["earth1"]
        earth2 = outputs["earth2"]

        nuisance1 = outputs["nuisance1"]
        nuisance2 = outputs["nuisance2"]

        # ------------------------------------------
        # Losses
        # ------------------------------------------

        change_loss = change_loss_fn(
            change_logits,
            label
        )

        temporal_loss = temporal_loss_fn(
            earth1,
            earth2,
            label
        )

        alignment_loss = alignment_loss_fn(
            earth1,
            earth2,
            label
        )

        disentangle_loss = (
            disentangle_loss_fn(
                earth1,
                nuisance1
            )
            +
            disentangle_loss_fn(
                earth2,
                nuisance2
            )
        ) / 2.0

        # ------------------------------------------
        # Total loss
        # ------------------------------------------

        total_loss = (
            change_loss
            + LAMBDA_TEMPORAL * temporal_loss
            + LAMBDA_ALIGNMENT * alignment_loss
            + LAMBDA_DISENTANGLE * disentangle_loss
        )

        total_loss.backward()

        optimizer.step()

        total_epoch_loss += total_loss.item()

    average_loss = (
        total_epoch_loss / len(loader)
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {average_loss:.4f}"
    )


# ============================================================
# FINAL EVALUATION ON THE SAME 10 SAMPLES
# ============================================================

model.eval()

tp = 0
fp = 0
fn = 0
tn = 0

with torch.no_grad():

    for image1, image2, label in loader:

        image1 = image1.to(device)
        image2 = image2.to(device)
        label = label.to(device)

        outputs = model(
            image1,
            image2
        )

        logits = outputs["change_logits"]

        predictions = (
            torch.sigmoid(logits) >= 0.5
        ).float()

        tp += (
            (predictions == 1) &
            (label == 1)
        ).sum().item()

        fp += (
            (predictions == 1) &
            (label == 0)
        ).sum().item()

        fn += (
            (predictions == 0) &
            (label == 1)
        ).sum().item()

        tn += (
            (predictions == 0) &
            (label == 0)
        ).sum().item()


epsilon = 1e-8

precision = tp / (tp + fp + epsilon)

recall = tp / (tp + fn + epsilon)

f1 = (
    2 * precision * recall /
    (precision + recall + epsilon)
)

iou = (
    tp /
    (tp + fp + fn + epsilon)
)

accuracy = (
    (tp + tn) /
    (tp + tn + fp + fn + epsilon)
)


# ============================================================
# RESULTS
# ============================================================

print("\n======================================")
print("       OVERFIT TEST RESULTS")
print("======================================")

print("Accuracy  :", f"{accuracy:.4f}")
print("Precision :", f"{precision:.4f}")
print("Recall    :", f"{recall:.4f}")
print("F1 Score  :", f"{f1:.4f}")
print("IoU       :", f"{iou:.4f}")

print("\nConfusion Matrix:")
print("TP :", tp)
print("FP :", fp)
print("FN :", fn)
print("TN :", tn)

print("\n======================================")
print("             TEST COMPLETE")
print("======================================")