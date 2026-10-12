import torch
from torch.utils.data import DataLoader, Subset
from dataset import S2LookingDataset
from complete_model import CompleteChangeDetectionModel

DATASET_PATH = r"K:\S2Looking"
IMAGE_SIZE = 256
BATCH_SIZE = 2
VAL_SAMPLES = 100
CHECKPOINT_PATH = "best_temporal_decoder_pw2_500x10.pth"
THRESHOLDS = [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("======================================")
print("THRESHOLD ANALYSIS")
print("======================================")
print("Device:", device)
print("Checkpoint:", CHECKPOINT_PATH)

print("\n======================================")
print("LOADING VALIDATION DATASET")
print("======================================")
full_val_dataset = S2LookingDataset(base_path=DATASET_PATH, split="val", image_size=IMAGE_SIZE)
val_count = min(VAL_SAMPLES, len(full_val_dataset))
val_dataset = Subset(full_val_dataset, range(val_count))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print("\nValidation samples:", len(val_dataset))

print("\n======================================")
print("LOADING MODEL")
print("======================================")
model = CompleteChangeDetectionModel(pretrained=True)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
model = model.to(device)
model.eval()
print("Checkpoint loaded successfully.")

print("\n======================================")
print("RUNNING MODEL ON VALIDATION SET")
print("======================================")
all_probabilities, all_labels = [], []
with torch.no_grad():
    for batch_index, (image1, image2, label) in enumerate(val_loader, start=1):
        image1, image2, label = image1.to(device), image2.to(device), label.to(device)
        outputs = model(image1, image2)
        all_probabilities.append(torch.sigmoid(outputs["change_logits"]).cpu())
        all_labels.append(label.cpu())
        if batch_index % 25 == 0:
            print(f"Processed {batch_index}/{len(val_loader)} batches")
probabilities = torch.cat(all_probabilities, dim=0)
labels = torch.cat(all_labels, dim=0)
print("\nTotal validation pixels:", labels.numel())

def calculate_metrics(prediction, target):
    eps = 1e-8
    tp = (prediction & (target == 1)).sum().item()
    fp = (prediction & (target == 0)).sum().item()
    fn = ((~prediction) & (target == 1)).sum().item()
    tn = ((~prediction) & (target == 0)).sum().item()
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    fcr = fp / (fp + tn + eps)
    return accuracy, precision, recall, f1, iou, fcr, tp, fp, fn, tn

results = []
print("\n======================================")
print("THRESHOLD RESULTS")
print("======================================")
print(f"{'Threshold':<12}{'Precision':<14}{'Recall':<14}{'F1':<14}{'IoU':<14}{'Accuracy':<14}{'False Change':<14}")
for threshold in THRESHOLDS:
    prediction = probabilities >= threshold
    accuracy, precision, recall, f1, iou, fcr, tp, fp, fn, tn = calculate_metrics(prediction, labels)
    results.append(dict(threshold=threshold, accuracy=accuracy, precision=precision, recall=recall, f1=f1, iou=iou, false_change_rate=fcr, tp=tp, fp=fp, fn=fn, tn=tn))
    print(f"{threshold:<12.2f}{precision:<14.6f}{recall:<14.6f}{f1:<14.6f}{iou:<14.6f}{accuracy:<14.6f}{fcr:<14.6f}")

best_f1 = max(results, key=lambda x: x["f1"])
best_iou = max(results, key=lambda x: x["iou"])
print("\n======================================")
print("BEST THRESHOLD BY F1")
print("======================================")
for k, label in [("threshold","Threshold"),("precision","Precision"),("recall","Recall"),("f1","F1 Score"),("iou","IoU"),("accuracy","Accuracy"),("false_change_rate","False Change Rate")]:
    print(f"{label:<19}: {best_f1[k]:.6f}")
print("\nConfusion Matrix:")
print(f"TP : {best_f1['tp']}\nFP : {best_f1['fp']}\nFN : {best_f1['fn']}\nTN : {best_f1['tn']}")
print("\n======================================")
print("BEST THRESHOLD BY IoU")
print("======================================")
print(f"Threshold : {best_iou['threshold']:.2f}")
print(f"IoU       : {best_iou['iou']:.6f}")
print(f"F1        : {best_iou['f1']:.6f}")
print("\n======================================")
print("THRESHOLD ANALYSIS COMPLETED")
print("======================================")
print("No training was performed.")
print("No model weights were modified.")
print("No checkpoint was overwritten.")
print("======================================")
