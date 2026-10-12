import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from dataset import S2LookingDataset
from baseline_model import BaselineSiameseChangeDetector

DATASET_PATH = r"K:\S2Looking"
IMAGE_SIZE = 256
BATCH_SIZE = 2
EPOCHS = 10
TRAIN_SAMPLES = 3500
VAL_SAMPLES = 500
POSITIVE_WEIGHT = 2.0
CHECKPOINT_PATH = "best_baseline_siamese.pth"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("======================================")
print("BASELINE SIAMESE TRAINING")
print("======================================")
print("Device :", device)

train_full = S2LookingDataset(DATASET_PATH, "train", IMAGE_SIZE)
val_full = S2LookingDataset(DATASET_PATH, "val", IMAGE_SIZE)

train_dataset = Subset(train_full, range(min(TRAIN_SAMPLES, len(train_full))))
val_dataset = Subset(val_full, range(min(VAL_SAMPLES, len(val_full))))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = BaselineSiameseChangeDetector().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2, threshold=1e-4
)
pos_weight = torch.tensor([POSITIVE_WEIGHT], device=device)
bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def dice_loss(logits, target):
    prob = torch.sigmoid(logits)
    smooth = 1.0
    inter = (prob * target).sum()
    return 1.0 - (2 * inter + smooth) / (prob.sum() + target.sum() + smooth)

best_f1 = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_total = 0.0

    for image1, image2, label in train_loader:
        image1, image2, label = image1.to(device), image2.to(device), label.to(device).float()
        optimizer.zero_grad()
        logits = model(image1, image2)["change_logits"]
        loss = bce(logits, label) + dice_loss(logits, label)
        loss.backward()
        optimizer.step()
        train_total += loss.item()

    model.eval()
    tp = fp = fn = tn = 0
    val_total = 0.0

    with torch.no_grad():
        for image1, image2, label in val_loader:
            image1, image2, label = image1.to(device), image2.to(device), label.to(device).float()
            logits = model(image1, image2)["change_logits"]
            val_total += (bce(logits, label) + dice_loss(logits, label)).item()

            pred = (torch.sigmoid(logits) >= 0.5).long().view(-1)
            gt = label.long().view(-1)
            tp += ((pred == 1) & (gt == 1)).sum().item()
            fp += ((pred == 1) & (gt == 0)).sum().item()
            fn += ((pred == 0) & (gt == 1)).sum().item()
            tn += ((pred == 0) & (gt == 0)).sum().item()

    eps = 1e-8
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    fcr = fp / (fp + tn + eps)

    scheduler.step(f1)
    lr = optimizer.param_groups[0]["lr"]

    print("\n======================================")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("======================================")
    print(f"Train Loss        : {train_total / len(train_loader):.6f}")
    print(f"Validation Loss   : {val_total / len(val_loader):.6f}")
    print(f"Accuracy          : {accuracy:.6f}")
    print(f"Precision         : {precision:.6f}")
    print(f"Recall            : {recall:.6f}")
    print(f"F1 Score          : {f1:.6f}")
    print(f"IoU               : {iou:.6f}")
    print(f"False Change Rate : {fcr:.6f}")
    print(f"Learning Rate     : {lr:.8f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch + 1,
            "best_f1": best_f1
        }, CHECKPOINT_PATH)
        print("Best baseline checkpoint saved.")

print("\n======================================")
print("BASELINE TRAINING COMPLETED")
print("======================================")
print(f"Best Validation F1 : {best_f1:.6f}")
print(f"Checkpoint          : {CHECKPOINT_PATH}")
print("======================================")
