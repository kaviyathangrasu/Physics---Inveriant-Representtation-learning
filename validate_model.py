import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import S2LookingDataset
from complete_model import CompleteChangeDetectionModel


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"K:\S2Looking"

CHECKPOINT_PATH = "best_temporal_decoder_pw2_500x10_sched.pth"

IMAGE_SIZE = 256
BATCH_SIZE = 2
THRESHOLD = 0.50

NUM_VISUALIZATIONS = 10
RESULTS_DIR = "results"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("======================================")
print("OFFICIAL TEST SET VALIDATION")
print("======================================")
print("Device    :", device)
print("Checkpoint:", CHECKPOINT_PATH)
print("Threshold :", THRESHOLD)


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading official test dataset...")

test_dataset = S2LookingDataset(
    base_path=DATASET_PATH,
    split="test",
    image_size=IMAGE_SIZE
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test samples:", len(test_dataset))


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

model = CompleteChangeDetectionModel().to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully!")


# ============================================================
# METRICS
# ============================================================

tp = 0
fp = 0
fn = 0
tn = 0


# ============================================================
# RESULTS FOLDER
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# IMAGE CONVERSION FUNCTION
# ============================================================

def tensor_to_image(tensor):

    image = (
        tensor
        .detach()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    # Undo ImageNet normalization
    mean = np.array(
        [0.485, 0.456, 0.406]
    )

    std = np.array(
        [0.229, 0.224, 0.225]
    )

    image = image * std + mean

    image = np.clip(
        image,
        0,
        1
    )

    return image


# ============================================================
# TEST LOOP
# ============================================================

print("\nRunning inference on all test samples...\n")


with torch.no_grad():

    for batch_index, batch in enumerate(test_loader):

        # ----------------------------------------------------
        # YOUR DATASET RETURNS:
        # image1, image2, label
        # ----------------------------------------------------

        image1, image2, label = batch

        image1 = image1.to(device)
        image2 = image2.to(device)
        label = label.to(device).float()


        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        output = model(
            image1,
            image2
        )

        logits = output["change_logits"]


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        probability = torch.sigmoid(
            logits
        )


        # ----------------------------------------------------
        # BINARY PREDICTION
        # ----------------------------------------------------

        prediction = (
            probability >= THRESHOLD
        ).float()


        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        pred_flat = (
            prediction
            .view(-1)
            .long()
        )

        label_flat = (
            label
            .view(-1)
            .long()
        )


        tp += (
            (pred_flat == 1) &
            (label_flat == 1)
        ).sum().item()


        fp += (
            (pred_flat == 1) &
            (label_flat == 0)
        ).sum().item()


        fn += (
            (pred_flat == 0) &
            (label_flat == 1)
        ).sum().item()


        tn += (
            (pred_flat == 0) &
            (label_flat == 0)
        ).sum().item()


        # ====================================================
        # SAVE VISUALIZATIONS
        # ====================================================

        start_index = (
            batch_index * BATCH_SIZE
        )


        for j in range(
            image1.size(0)
        ):

            sample_index = (
                start_index + j
            )


            if sample_index >= NUM_VISUALIZATIONS:
                continue


            img1 = tensor_to_image(
                image1[j]
            )

            img2 = tensor_to_image(
                image2[j]
            )


            gt = (
                label[j]
                .detach()
                .cpu()
                .squeeze()
                .numpy()
            )


            pred = (
                prediction[j]
                .detach()
                .cpu()
                .squeeze()
                .numpy()
            )


            # ------------------------------------------------
            # CREATE FIGURE
            # ------------------------------------------------

            fig, axes = plt.subplots(
                1,
                4,
                figsize=(16, 4)
            )


            axes[0].imshow(img1)
            axes[0].set_title(
                "Image 1"
            )
            axes[0].axis("off")


            axes[1].imshow(img2)
            axes[1].set_title(
                "Image 2"
            )
            axes[1].axis("off")


            axes[2].imshow(
                gt,
                cmap="gray"
            )
            axes[2].set_title(
                "Ground Truth"
            )
            axes[2].axis("off")


            axes[3].imshow(
                pred,
                cmap="gray"
            )
            axes[3].set_title(
                "Prediction"
            )
            axes[3].axis("off")


            plt.tight_layout()


            # ------------------------------------------------
            # SAVE IMAGE
            # ------------------------------------------------

            output_path = os.path.join(
                RESULTS_DIR,
                f"test_sample_{sample_index + 1:03d}.png"
            )


            plt.savefig(
                output_path,
                dpi=150,
                bbox_inches="tight"
            )


            plt.close(fig)


        # ====================================================
        # PROGRESS
        # ====================================================

        if (batch_index + 1) % 100 == 0:

            processed = min(
                (batch_index + 1) * BATCH_SIZE,
                len(test_dataset)
            )

            print(
                f"Processed "
                f"{processed}/"
                f"{len(test_dataset)} samples"
            )


# ============================================================
# CALCULATE METRICS
# ============================================================

eps = 1e-8

total = (
    tp +
    fp +
    fn +
    tn
)


accuracy = (
    (tp + tn) /
    (total + eps)
)


precision = (
    tp /
    (tp + fp + eps)
)


recall = (
    tp /
    (tp + fn + eps)
)


f1 = (
    2 *
    precision *
    recall /
    (precision + recall + eps)
)


iou = (
    tp /
    (tp + fp + fn + eps)
)


false_change_rate = (
    fp /
    (fp + tn + eps)
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n======================================")
print("OFFICIAL TEST RESULTS")
print("======================================")

print(
    f"Accuracy          : {accuracy:.6f}"
)

print(
    f"Precision         : {precision:.6f}"
)

print(
    f"Recall            : {recall:.6f}"
)

print(
    f"F1 Score          : {f1:.6f}"
)

print(
    f"IoU               : {iou:.6f}"
)

print(
    f"False Change Rate : {false_change_rate:.6f}"
)


print("\nConfusion Matrix:")

print(
    "TP :",
    tp
)

print(
    "FP :",
    fp
)

print(
    "FN :",
    fn
)

print(
    "TN :",
    tn
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = os.path.join(
    RESULTS_DIR,
    "test_metrics.txt"
)


with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "OFFICIAL S2Looking TEST SET RESULTS\n"
    )

    f.write(
        "======================================\n"
    )

    f.write(
        f"Checkpoint         : "
        f"{CHECKPOINT_PATH}\n"
    )

    f.write(
        f"Test samples       : "
        f"{len(test_dataset)}\n"
    )

    f.write(
        f"Threshold          : "
        f"{THRESHOLD:.2f}\n\n"
    )

    f.write(
        f"Accuracy           : "
        f"{accuracy:.6f}\n"
    )

    f.write(
        f"Precision          : "
        f"{precision:.6f}\n"
    )

    f.write(
        f"Recall             : "
        f"{recall:.6f}\n"
    )

    f.write(
        f"F1 Score           : "
        f"{f1:.6f}\n"
    )

    f.write(
        f"IoU                : "
        f"{iou:.6f}\n"
    )

    f.write(
        f"False Change Rate  : "
        f"{false_change_rate:.6f}\n\n"
    )

    f.write(
        "Confusion Matrix\n"
    )

    f.write(
        f"TP : {tp}\n"
    )

    f.write(
        f"FP : {fp}\n"
    )

    f.write(
        f"FN : {fn}\n"
    )

    f.write(
        f"TN : {tn}\n"
    )


# ============================================================
# FINISHED
# ============================================================

print(
    "\nSaved metrics to:",
    metrics_path
)

print(
    "Saved visualizations to:",
    RESULTS_DIR
)

print(
    "======================================"
)

print(
    "TEST VALIDATION COMPLETED"
)

print(
    "======================================"
)