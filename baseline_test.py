import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import S2LookingDataset
from baseline_model import BaselineSiameseChangeDetector

DATASET_PATH = r"K:\S2Looking"
CHECKPOINT_PATH = "best_baseline_siamese.pth"
IMAGE_SIZE = 256
BATCH_SIZE = 2
THRESHOLD = 0.50
NUM_VISUALIZATIONS = 10
RESULTS_DIR = "results_baseline"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("======================================")
print("BASELINE OFFICIAL TEST VALIDATION")
print("======================================")
print("Device    :", device)
print("Checkpoint:", CHECKPOINT_PATH)
print("Threshold :", THRESHOLD)

print("\nLoading official test dataset...")
test_dataset = S2LookingDataset(DATASET_PATH, "test", IMAGE_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print("Test samples:", len(test_dataset))

print("\nLoading baseline model...")
model = BaselineSiameseChangeDetector().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()
print("Baseline model loaded successfully!")

tp = fp = fn = tn = 0
os.makedirs(RESULTS_DIR, exist_ok=True)

def tensor_to_image(tensor):
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return np.clip(image * std + mean, 0, 1)

print("\nRunning baseline inference on all test samples...\n")
with torch.no_grad():
    for batch_index, batch in enumerate(test_loader):
        image1, image2, label = batch
        image1 = image1.to(device)
        image2 = image2.to(device)
        label = label.to(device).float()
        logits = model(image1, image2)["change_logits"]
        probability = torch.sigmoid(logits)
        prediction = (probability >= THRESHOLD).float()

        pred_flat = prediction.view(-1).long()
        label_flat = label.view(-1).long()
        tp += ((pred_flat == 1) & (label_flat == 1)).sum().item()
        fp += ((pred_flat == 1) & (label_flat == 0)).sum().item()
        fn += ((pred_flat == 0) & (label_flat == 1)).sum().item()
        tn += ((pred_flat == 0) & (label_flat == 0)).sum().item()

        start_index = batch_index * BATCH_SIZE
        for j in range(image1.size(0)):
            sample_index = start_index + j
            if sample_index >= NUM_VISUALIZATIONS:
                continue
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(tensor_to_image(image1[j])); axes[0].set_title("Image 1"); axes[0].axis("off")
            axes[1].imshow(tensor_to_image(image2[j])); axes[1].set_title("Image 2"); axes[1].axis("off")
            axes[2].imshow(label[j].detach().cpu().squeeze().numpy(), cmap="gray"); axes[2].set_title("Ground Truth"); axes[2].axis("off")
            axes[3].imshow(prediction[j].detach().cpu().squeeze().numpy(), cmap="gray"); axes[3].set_title("Baseline Prediction"); axes[3].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"test_sample_{sample_index + 1:03d}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

        if (batch_index + 1) % 100 == 0:
            processed = min((batch_index + 1) * BATCH_SIZE, len(test_dataset))
            print(f"Processed {processed}/{len(test_dataset)} samples")

eps = 1e-8
total = tp + fp + fn + tn
accuracy = (tp + tn) / (total + eps)
precision = tp / (tp + fp + eps)
recall = tp / (tp + fn + eps)
f1 = 2 * precision * recall / (precision + recall + eps)
iou = tp / (tp + fp + fn + eps)
false_change_rate = fp / (fp + tn + eps)

print("\n======================================")
print("BASELINE OFFICIAL TEST RESULTS")
print("======================================")
print(f"Accuracy          : {accuracy:.6f}")
print(f"Precision         : {precision:.6f}")
print(f"Recall            : {recall:.6f}")
print(f"F1 Score          : {f1:.6f}")
print(f"IoU               : {iou:.6f}")
print(f"False Change Rate : {false_change_rate:.6f}")
print("\nConfusion Matrix:")
print("TP :", tp); print("FP :", fp); print("FN :", fn); print("TN :", tn)

metrics_path = os.path.join(RESULTS_DIR, "baseline_test_metrics.txt")
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write("S2Looking BASELINE OFFICIAL TEST RESULTS\n")
    f.write("======================================\n")
    f.write(f"Checkpoint         : {CHECKPOINT_PATH}\n")
    f.write(f"Test samples       : {len(test_dataset)}\n")
    f.write(f"Threshold          : {THRESHOLD:.2f}\n\n")
    f.write(f"Accuracy           : {accuracy:.6f}\n")
    f.write(f"Precision          : {precision:.6f}\n")
    f.write(f"Recall             : {recall:.6f}\n")
    f.write(f"F1 Score           : {f1:.6f}\n")
    f.write(f"IoU                : {iou:.6f}\n")
    f.write(f"False Change Rate  : {false_change_rate:.6f}\n\n")
    f.write("Confusion Matrix\n")
    f.write(f"TP : {tp}\nFP : {fp}\nFN : {fn}\nTN : {tn}\n")
print("\nSaved metrics to:", metrics_path)
print("Saved visualizations to:", RESULTS_DIR)
print("======================================")
print("BASELINE TEST VALIDATION COMPLETED")
print("======================================")
