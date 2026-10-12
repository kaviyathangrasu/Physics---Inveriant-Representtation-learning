import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import S2LookingDataset
from baseline_alignment_model import BaselineFeatureAlignmentChangeDetector
from feature_alignment import FeatureAlignmentLoss


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = r"K:\S2Looking"

IMAGE_SIZE = 256
BATCH_SIZE = 2

EPOCHS = 10

TRAIN_SAMPLES = 3500
VAL_SAMPLES = 500

POSITIVE_WEIGHT = 2.0

# Weight for feature alignment loss
ALIGNMENT_WEIGHT = 0.1

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

CHECKPOINT_PATH = "best_baseline_alignment.pth"

SEED = 42


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("======================================")
print("BASELINE + FEATURE ALIGNMENT TRAINING")
print("======================================")

print("Device :", device)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading datasets...")


train_full = S2LookingDataset(
    DATASET_PATH,
    "train",
    IMAGE_SIZE
)


val_full = S2LookingDataset(
    DATASET_PATH,
    "val",
    IMAGE_SIZE
)


# ============================================================
# SELECT DATA
# ============================================================

train_dataset = Subset(
    train_full,
    range(
        min(
            TRAIN_SAMPLES,
            len(train_full)
        )
    )
)


val_dataset = Subset(
    val_full,
    range(
        min(
            VAL_SAMPLES,
            len(val_full)
        )
    )
)


# ============================================================
# DATA LOADERS
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


print("\nDataset information:")

print(
    f"Training samples   : {len(train_dataset)}"
)

print(
    f"Validation samples : {len(val_dataset)}"
)


# ============================================================
# MODEL
# ============================================================

print("\nCreating model...")


model = BaselineFeatureAlignmentChangeDetector().to(device)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    threshold=1e-4
)


# ============================================================
# BCE LOSS
# ============================================================

pos_weight = torch.tensor(
    [POSITIVE_WEIGHT],
    device=device
)


bce = torch.nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)


# ============================================================
# FEATURE ALIGNMENT LOSS
# ============================================================

alignment_loss_function = FeatureAlignmentLoss()


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(logits, target):

    probability = torch.sigmoid(logits)

    smooth = 1.0

    intersection = (
        probability * target
    ).sum()

    dice = (
        2.0 * intersection + smooth
    ) / (
        probability.sum()
        + target.sum()
        + smooth
    )

    return 1.0 - dice


# ============================================================
# BEST VALIDATION F1
# ============================================================

best_f1 = 0.0


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    # ========================================================
    # TRAINING MODE
    # ========================================================

    model.train()

    train_total = 0.0
    train_change_loss = 0.0
    train_alignment_loss = 0.0


    # ========================================================
    # TRAIN BATCHES
    # ========================================================

    for image1, image2, label in train_loader:

        image1 = image1.to(device)

        image2 = image2.to(device)

        label = label.to(device).float()


        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        output = model(
            image1,
            image2
        )


        logits = output[
            "change_logits"
        ]


        # ----------------------------------------------------
        # Change detection loss
        # ----------------------------------------------------

        loss_bce = bce(
            logits,
            label
        )


        loss_dice = dice_loss(
            logits,
            label
        )


        change_loss = (
            loss_bce +
            loss_dice
        )


        # ----------------------------------------------------
        # Unchanged region mask
        # ----------------------------------------------------

        unchanged_mask = (
            1.0 - label
        )


        # ----------------------------------------------------
        # Multi-scale feature alignment
        # ----------------------------------------------------

        alignment1 = alignment_loss_function(
            output["feature1_image1"],
            output["feature1_image2"],
            unchanged_mask
        )


        alignment2 = alignment_loss_function(
            output["feature2_image1"],
            output["feature2_image2"],
            unchanged_mask
        )


        alignment3 = alignment_loss_function(
            output["feature3_image1"],
            output["feature3_image2"],
            unchanged_mask
        )


        alignment4 = alignment_loss_function(
            output["feature4_image1"],
            output["feature4_image2"],
            unchanged_mask
        )


        # ----------------------------------------------------
        # Average alignment loss
        # ----------------------------------------------------

        alignment_loss = (
            alignment1
            + alignment2
            + alignment3
            + alignment4
        ) / 4.0


        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------

        total_loss = (
            change_loss
            +
            ALIGNMENT_WEIGHT * alignment_loss
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        total_loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # Store losses
        # ----------------------------------------------------

        train_total += total_loss.item()

        train_change_loss += change_loss.item()

        train_alignment_loss += alignment_loss.item()


    # ========================================================
    # VALIDATION MODE
    # ========================================================

    model.eval()

    val_total = 0.0
    val_change_loss = 0.0
    val_alignment_loss = 0.0


    # Confusion matrix
    tp = 0
    fp = 0
    fn = 0
    tn = 0


    # ========================================================
    # VALIDATION
    # ========================================================

    with torch.no_grad():

        for image1, image2, label in val_loader:

            image1 = image1.to(device)

            image2 = image2.to(device)

            label = label.to(device).float()


            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            output = model(
                image1,
                image2
            )


            logits = output[
                "change_logits"
            ]


            # ------------------------------------------------
            # Change loss
            # ------------------------------------------------

            loss_bce = bce(
                logits,
                label
            )


            loss_dice = dice_loss(
                logits,
                label
            )


            change_loss = (
                loss_bce +
                loss_dice
            )


            # ------------------------------------------------
            # Unchanged mask
            # ------------------------------------------------

            unchanged_mask = (
                1.0 - label
            )


            # ------------------------------------------------
            # Alignment losses
            # ------------------------------------------------

            alignment1 = alignment_loss_function(
                output["feature1_image1"],
                output["feature1_image2"],
                unchanged_mask
            )


            alignment2 = alignment_loss_function(
                output["feature2_image1"],
                output["feature2_image2"],
                unchanged_mask
            )


            alignment3 = alignment_loss_function(
                output["feature3_image1"],
                output["feature3_image2"],
                unchanged_mask
            )


            alignment4 = alignment_loss_function(
                output["feature4_image1"],
                output["feature4_image2"],
                unchanged_mask
            )


            alignment_loss = (
                alignment1
                + alignment2
                + alignment3
                + alignment4
            ) / 4.0


            # ------------------------------------------------
            # Total validation loss
            # ------------------------------------------------

            total_loss = (
                change_loss
                +
                ALIGNMENT_WEIGHT * alignment_loss
            )


            val_total += total_loss.item()

            val_change_loss += change_loss.item()

            val_alignment_loss += alignment_loss.item()


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            probability = torch.sigmoid(
                logits
            )


            prediction = (
                probability >= 0.5
            ).long()


            ground_truth = (
                label >= 0.5
            ).long()


            prediction = prediction.view(-1)

            ground_truth = ground_truth.view(-1)


            # ------------------------------------------------
            # TP
            # ------------------------------------------------

            tp += (
                (prediction == 1)
                &
                (ground_truth == 1)
            ).sum().item()


            # ------------------------------------------------
            # FP
            # ------------------------------------------------

            fp += (
                (prediction == 1)
                &
                (ground_truth == 0)
            ).sum().item()


            # ------------------------------------------------
            # FN
            # ------------------------------------------------

            fn += (
                (prediction == 0)
                &
                (ground_truth == 1)
            ).sum().item()


            # ------------------------------------------------
            # TN
            # ------------------------------------------------

            tn += (
                (prediction == 0)
                &
                (ground_truth == 0)
            ).sum().item()


    # ========================================================
    # METRICS
    # ========================================================

    eps = 1e-8

    total_pixels = (
        tp + fp + fn + tn
    )


    accuracy = (
        tp + tn
    ) / (
        total_pixels + eps
    )


    precision = tp / (
        tp + fp + eps
    )


    recall = tp / (
        tp + fn + eps
    )


    f1 = (
        2.0 * precision * recall
    ) / (
        precision + recall + eps
    )


    iou = tp / (
        tp + fp + fn + eps
    )


    false_change_rate = fp / (
        fp + tn + eps
    )


    # ========================================================
    # AVERAGE LOSSES
    # ========================================================

    train_average = (
        train_total /
        len(train_loader)
    )


    train_change_average = (
        train_change_loss /
        len(train_loader)
    )


    train_alignment_average = (
        train_alignment_loss /
        len(train_loader)
    )


    val_average = (
        val_total /
        len(val_loader)
    )


    val_change_average = (
        val_change_loss /
        len(val_loader)
    )


    val_alignment_average = (
        val_alignment_loss /
        len(val_loader)
    )


    # ========================================================
    # LEARNING RATE UPDATE
    # ========================================================

    scheduler.step(f1)


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # ========================================================
    # PRINT EPOCH RESULTS
    # ========================================================

    print("\n======================================")

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print("======================================")


    print(
        f"Train Total Loss       : "
        f"{train_average:.6f}"
    )


    print(
        f"Train Change Loss      : "
        f"{train_change_average:.6f}"
    )


    print(
        f"Train Alignment Loss   : "
        f"{train_alignment_average:.6f}"
    )


    print(
        f"Validation Total Loss  : "
        f"{val_average:.6f}"
    )


    print(
        f"Validation Change Loss : "
        f"{val_change_average:.6f}"
    )


    print(
        f"Validation Alignment   : "
        f"{val_alignment_average:.6f}"
    )


    print(
        f"Accuracy               : "
        f"{accuracy:.6f}"
    )


    print(
        f"Precision              : "
        f"{precision:.6f}"
    )


    print(
        f"Recall                 : "
        f"{recall:.6f}"
    )


    print(
        f"F1 Score               : "
        f"{f1:.6f}"
    )


    print(
        f"IoU                    : "
        f"{iou:.6f}"
    )


    print(
        f"False Change Rate      : "
        f"{false_change_rate:.6f}"
    )


    print(
        f"Learning Rate          : "
        f"{current_lr:.8f}"
    )


    # ========================================================
    # SAVE BEST CHECKPOINT
    # ========================================================

    if f1 > best_f1:

        best_f1 = f1


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "epoch":
                    epoch + 1,

                "best_f1":
                    best_f1
            },
            CHECKPOINT_PATH
        )


        print(
            "Best Baseline + Feature "
            "Alignment checkpoint saved."
        )


# ============================================================
# COMPLETED
# ============================================================

print("\n======================================")

print(
    "BASELINE + FEATURE ALIGNMENT "
    "TRAINING COMPLETED"
)

print("======================================")

print(
    f"Best Validation F1 : "
    f"{best_f1:.6f}"
)

print(
    f"Checkpoint          : "
    f"{CHECKPOINT_PATH}"
)

print("======================================")