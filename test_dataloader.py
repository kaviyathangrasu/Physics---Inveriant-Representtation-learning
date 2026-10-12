import torch
from torch.utils.data import DataLoader
from dataset import S2LookingDataset

BASE_PATH = r"K:\S2Looking"

# Load dataset
dataset = S2LookingDataset(
    base_path=BASE_PATH,
    split="train",
    image_size=256
)

# Create DataLoader
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

print("\n======================================")
print("        DATALOADER TEST")
print("======================================")

image1, image2, label = next(iter(loader))

print("Image1 batch shape :", image1.shape)
print("Image2 batch shape :", image2.shape)
print("Label batch shape  :", label.shape)

print("Image1 dtype :", image1.dtype)
print("Image2 dtype :", image2.dtype)
print("Label dtype  :", label.dtype)

print("Label unique values :", torch.unique(label))

# Expected shapes
expected_image_shape = (4, 3, 256, 256)
expected_label_shape = (4, 1, 256, 256)

if (
    image1.shape == expected_image_shape
    and image2.shape == expected_image_shape
    and label.shape == expected_label_shape
):
    print("\nDATA LOADER TEST PASSED")
else:
    print("\nDATA LOADER TEST FAILED")