import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset

from dataset import S2LookingDataset
from complete_model import CompleteChangeDetectionModel
from physics_loss import TemporalConsistencyLoss
from feature_alignment import FeatureAlignmentLoss
from dual_head import DisentanglementLoss


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = r"K:\S2Looking"

IMAGE_SIZE = 256

BATCH_SIZE = 2

EPOCHS = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

TRAIN_SAMPLES = 3500

VAL_SAMPLES = 500

# Current best candidate from previous experiment
POSITIVE_WEIGHT = 2.0

DICE_WEIGHT = 1.0

TEMPORAL_WEIGHT = 0.1

ALIGNMENT_WEIGHT = 0.1

DISENTANGLEMENT_WEIGHT = 0.1

SEED = 42

# New checkpoint because decoder architecture changed
CHECKPOINT_PATH = (
    "best_temporal_decoder_pw2_500x10_sched.pth"
)

# Learning-rate scheduling and early stopping
LR_FACTOR = 0.5
LR_PATIENCE = 2
EARLY_STOPPING_PATIENCE = 4
MIN_DELTA = 1e-4


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)

torch.manual_seed(SEED)


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("======================================")
print("DEVICE")
print("======================================")

print("Device:", device)


# ============================================================
# 4. CHANGE DETECTION LOSS
#    WEIGHTED BCE + DICE
# ============================================================

class WeightedBCEDiceLoss(nn.Module):

    def __init__(
        self,
        positive_weight=2.0,
        dice_weight=1.0
    ):

        super().__init__()

        self.positive_weight = positive_weight

        self.dice_weight = dice_weight


    def forward(self, logits, target):

        # ----------------------------------------------------
        # Weighted BCE
        # ----------------------------------------------------

        pos_weight = torch.tensor(
            self.positive_weight,
            device=logits.device
        )

        bce_loss = (
            F.binary_cross_entropy_with_logits(
                logits,
                target,
                pos_weight=pos_weight
            )
        )


        # ----------------------------------------------------
        # Dice Loss
        # ----------------------------------------------------

        probability = torch.sigmoid(
            logits
        )


        probability_flat = probability.view(
            probability.size(0),
            -1
        )


        target_flat = target.view(
            target.size(0),
            -1
        )


        intersection = (
            probability_flat
            * target_flat
        ).sum(dim=1)


        denominator = (
            probability_flat.sum(dim=1)
            + target_flat.sum(dim=1)
            + 1e-8
        )


        dice_score = (
            2.0 * intersection
            + 1e-8
        ) / denominator


        dice_loss = (
            1.0 - dice_score.mean()
        )


        # ----------------------------------------------------
        # Combined loss
        # ----------------------------------------------------

        total_loss = (
            bce_loss
            + self.dice_weight * dice_loss
        )


        return total_loss


# ============================================================
# 5. LOAD TRAIN DATASET
# ============================================================

print("\n======================================")
print("LOADING DATASET")
print("======================================")


full_train_dataset = S2LookingDataset(
    base_path=DATASET_PATH,
    split="train",
    image_size=IMAGE_SIZE
)


# ============================================================
# 6. LOAD VALIDATION DATASET
# ============================================================

full_val_dataset = S2LookingDataset(
    base_path=DATASET_PATH,
    split="val",
    image_size=IMAGE_SIZE
)


# ============================================================
# 7. CREATE PILOT SUBSETS
# ============================================================

train_count = min(
    TRAIN_SAMPLES,
    len(full_train_dataset)
)


val_count = min(
    VAL_SAMPLES,
    len(full_val_dataset)
)


train_indices = list(
    range(train_count)
)


val_indices = list(
    range(val_count)
)


train_dataset = Subset(
    full_train_dataset,
    train_indices
)


val_dataset = Subset(
    full_val_dataset,
    val_indices
)


print(
    "\nTraining samples       :",
    len(train_dataset)
)


print(
    "Validation samples     :",
    len(val_dataset)
)


# ============================================================
# 8. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print(
    "\nDataLoaders created successfully."
)


# ============================================================
# 9. CREATE MODEL
# ============================================================

print("\n======================================")
print("CREATING MODEL")
print("======================================")


model = CompleteChangeDetectionModel(
    pretrained=True
)


model = model.to(device)


print(
    "Complete model created successfully."
)


# ============================================================
# 10. CREATE LOSSES
# ============================================================

change_loss_fn = WeightedBCEDiceLoss(
    positive_weight=POSITIVE_WEIGHT,
    dice_weight=DICE_WEIGHT
)


temporal_loss_fn = TemporalConsistencyLoss()


alignment_loss_fn = FeatureAlignmentLoss()


disentanglement_loss_fn = (
    DisentanglementLoss()
)


print(
    "All loss functions created successfully."
)


# ============================================================
# 11. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# Reduce the learning rate when validation F1 stops improving.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=LR_FACTOR,
    patience=LR_PATIENCE
)


# ============================================================
# 12. METRIC FUNCTION
# ============================================================

def calculate_metrics(
    true_positive,
    false_positive,
    false_negative,
    true_negative
):

    epsilon = 1e-8


    accuracy = (
        true_positive
        + true_negative
    ) / (
        true_positive
        + true_negative
        + false_positive
        + false_negative
        + epsilon
    )


    precision = (
        true_positive
    ) / (
        true_positive
        + false_positive
        + epsilon
    )


    recall = (
        true_positive
    ) / (
        true_positive
        + false_negative
        + epsilon
    )


    f1 = (
        2.0
        * precision
        * recall
    ) / (
        precision
        + recall
        + epsilon
    )


    iou = (
        true_positive
    ) / (
        true_positive
        + false_positive
        + false_negative
        + epsilon
    )


    false_change_rate = (
        false_positive
    ) / (
        false_positive
        + true_negative
        + epsilon
    )


    return (
        accuracy,
        precision,
        recall,
        f1,
        iou,
        false_change_rate
    )


# ============================================================
# 13. BEST F1
# ============================================================

best_f1 = -1.0

epochs_without_improvement = 0


# ============================================================
# 14. TRAINING LOOP
# ============================================================

for epoch in range(
    1,
    EPOCHS + 1
):


    # ========================================================
    # TRAIN MODE
    # ========================================================

    model.train()


    running_total_loss = 0.0

    running_change_loss = 0.0

    running_temporal_loss = 0.0

    running_alignment_loss = 0.0

    running_disentanglement_loss = 0.0


    print("\n")

    print(
        "======================================"
    )

    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print(
        "======================================"
    )


    # ========================================================
    # TRAIN BATCHES
    # ========================================================

    for batch_index, (
        image1,
        image2,
        label
    ) in enumerate(
        train_loader,
        start=1
    ):


        image1 = image1.to(device)

        image2 = image2.to(device)

        label = label.to(device)


        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(
            image1,
            image2
        )


        # ----------------------------------------------------
        # Extract required outputs
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Change detection loss
        # ----------------------------------------------------

        change_loss = change_loss_fn(
            change_logits,
            label
        )


        # ----------------------------------------------------
        # Temporal consistency
        # ----------------------------------------------------

        temporal_loss = temporal_loss_fn(
            earth1,
            earth2,
            label
        )


        # ----------------------------------------------------
        # Feature alignment
        # ----------------------------------------------------

        alignment_loss = alignment_loss_fn(
            earth1,
            earth2,
            label
        )


        # ----------------------------------------------------
        # Disentanglement
        # ----------------------------------------------------

        disentanglement_loss = (
            disentanglement_loss_fn(
                earth1,
                nuisance1
            )
        )


        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------

        total_loss = (

            change_loss

            + TEMPORAL_WEIGHT
            * temporal_loss

            + ALIGNMENT_WEIGHT
            * alignment_loss

            + DISENTANGLEMENT_WEIGHT
            * disentanglement_loss

        )


        # ----------------------------------------------------
        # Check loss
        # ----------------------------------------------------

        if not torch.isfinite(
            total_loss
        ):

            raise RuntimeError(
                "Non-finite loss detected."
            )


        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        total_loss.backward()


        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # Accumulate losses
        # ----------------------------------------------------

        running_total_loss += (
            total_loss.item()
        )


        running_change_loss += (
            change_loss.item()
        )


        running_temporal_loss += (
            temporal_loss.item()
        )


        running_alignment_loss += (
            alignment_loss.item()
        )


        running_disentanglement_loss += (
            disentanglement_loss.item()
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if batch_index % 10 == 0:

            print(
                f"Batch {batch_index:3d}/"
                f"{len(train_loader):3d} | "
                f"Total: "
                f"{total_loss.item():.4f} | "
                f"Change: "
                f"{change_loss.item():.4f}"
            )


    # ========================================================
    # 15. AVERAGE TRAINING LOSSES
    # ========================================================

    num_train_batches = len(
        train_loader
    )


    avg_total_loss = (
        running_total_loss
        / num_train_batches
    )


    avg_change_loss = (
        running_change_loss
        / num_train_batches
    )


    avg_temporal_loss = (
        running_temporal_loss
        / num_train_batches
    )


    avg_alignment_loss = (
        running_alignment_loss
        / num_train_batches
    )


    avg_disentanglement_loss = (
        running_disentanglement_loss
        / num_train_batches
    )


    # ========================================================
    # 16. VALIDATION
    # ========================================================

    model.eval()


    validation_loss = 0.0

    validation_batches = 0


    true_positive = 0

    false_positive = 0

    false_negative = 0

    true_negative = 0


    with torch.no_grad():

        for (
            image1,
            image2,
            label
        ) in val_loader:


            image1 = image1.to(device)

            image2 = image2.to(device)

            label = label.to(device)


            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            outputs = model(
                image1,
                image2
            )


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


            # ------------------------------------------------
            # Validation losses
            # ------------------------------------------------

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


            disentanglement_loss = (
                disentanglement_loss_fn(
                    earth1,
                    nuisance1
                )
            )


            total_loss = (

                change_loss

                + TEMPORAL_WEIGHT
                * temporal_loss

                + ALIGNMENT_WEIGHT
                * alignment_loss

                + DISENTANGLEMENT_WEIGHT
                * disentanglement_loss

            )


            validation_loss += (
                total_loss.item()
            )


            validation_batches += 1


            # ------------------------------------------------
            # Probability
            # ------------------------------------------------

            probability = torch.sigmoid(
                change_logits
            )


            # ------------------------------------------------
            # Standard threshold = 0.5
            # ------------------------------------------------

            prediction = (
                probability >= 0.5
            )


            # ------------------------------------------------
            # Confusion matrix
            # ------------------------------------------------

            true_positive += (
                (
                    prediction
                    & (label == 1)
                )
                .sum()
                .item()
            )


            false_positive += (
                (
                    prediction
                    & (label == 0)
                )
                .sum()
                .item()
            )


            false_negative += (
                (
                    (~prediction)
                    & (label == 1)
                )
                .sum()
                .item()
            )


            true_negative += (
                (
                    (~prediction)
                    & (label == 0)
                )
                .sum()
                .item()
            )


    # ========================================================
    # 17. VALIDATION METRICS
    # ========================================================

    validation_loss /= (
        validation_batches
    )


    (
        accuracy,
        precision,
        recall,
        f1,
        iou,
        false_change_rate
    ) = calculate_metrics(
        true_positive,
        false_positive,
        false_negative,
        true_negative
    )


    # ========================================================
    # 18. PRINT RESULTS
    # ========================================================

    print("\n--------------------------------------")


    print(
        f"Train Total Loss       : "
        f"{avg_total_loss:.6f}"
    )


    print(
        f"Train Change Loss      : "
        f"{avg_change_loss:.6f}"
    )


    print(
        f"Train Temporal Loss    : "
        f"{avg_temporal_loss:.6f}"
    )


    print(
        f"Train Alignment Loss   : "
        f"{avg_alignment_loss:.6f}"
    )


    print(
        f"Train Disentangle Loss : "
        f"{avg_disentanglement_loss:.6f}"
    )


    print(
        f"Validation Loss        : "
        f"{validation_loss:.6f}"
    )


    print("\nValidation Metrics:")


    print(
        f"Accuracy          : "
        f"{accuracy:.6f}"
    )


    print(
        f"Precision         : "
        f"{precision:.6f}"
    )


    print(
        f"Recall            : "
        f"{recall:.6f}"
    )


    print(
        f"F1 Score          : "
        f"{f1:.6f}"
    )


    print(
        f"IoU               : "
        f"{iou:.6f}"
    )


    print(
        f"False Change Rate : "
        f"{false_change_rate:.6f}"
    )


    print("\nConfusion Matrix:")


    print(
        f"TP : {true_positive}"
    )


    print(
        f"FP : {false_positive}"
    )


    print(
        f"FN : {false_negative}"
    )


    print(
        f"TN : {true_negative}"
    )


    # ========================================================
    # ========================================================
# 19. SAVE BEST MODEL + EARLY STOPPING
# ========================================================

    print(
        f"Learning Rate        : {optimizer.param_groups[0]['lr']:.8f}"
    )

    if f1 > best_f1 + MIN_DELTA:

        best_f1 = f1
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "validation_f1": f1,
                "validation_iou": iou,
                "validation_precision": precision,
                "validation_recall": recall,
                "validation_accuracy": accuracy,
                "false_change_rate": false_change_rate,
                "positive_weight": POSITIVE_WEIGHT,
                "temporal_weight": TEMPORAL_WEIGHT,
                "alignment_weight": ALIGNMENT_WEIGHT,
                "disentanglement_weight": DISENTANGLEMENT_WEIGHT
            },
            CHECKPOINT_PATH
        )

        print("\nBest model saved:")
        print(CHECKPOINT_PATH)

    else:

        epochs_without_improvement += 1

        print(
            f"No F1 improvement for "
            f"{epochs_without_improvement} epoch(s)."
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

            print("\nEarly stopping triggered.")
            break


# 20. FINAL RESULT
# ============================================================

print("\n======================================")

print(
    "FULL TRAINING COMPLETED"
)

print(
    "======================================"
)


print(
    f"Positive Weight    : "
    f"{POSITIVE_WEIGHT}"
)


print(
    f"Best Validation F1 : "
    f"{best_f1:.6f}"
)


print(
    f"Checkpoint         : "
    f"{CHECKPOINT_PATH}"
)


print(
    "======================================"
)