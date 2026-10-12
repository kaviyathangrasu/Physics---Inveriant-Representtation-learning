import torch
from dataset import S2LookingDataset


BASE_PATH = r"K:\S2Looking"


dataset = S2LookingDataset(
    base_path=BASE_PATH,
    split="train",
    image_size=256
)


print()
print("Dataset length:", len(dataset))


image1, image2, label = dataset[0]


print()
print("======================================")
print("       FIRST SAMPLE TEST")
print("======================================")

print("Image1 shape :", image1.shape)
print("Image2 shape :", image2.shape)
print("Label shape  :", label.shape)

print()
print("Image1 dtype :", image1.dtype)
print("Image2 dtype :", image2.dtype)
print("Label dtype  :", label.dtype)

print()
print("Image1 min   :", image1.min().item())
print("Image1 max   :", image1.max().item())

print("Image2 min   :", image2.min().item())
print("Image2 max   :", image2.max().item())

print("Label values :", torch.unique(label))

print()
print("======================================")

if (
    image1.shape == (3, 256, 256)
    and image2.shape == (3, 256, 256)
    and label.shape == (1, 256, 256)
):
    print("DATASET TEST PASSED")
else:
    print("DATASET TEST FAILED")

print("======================================")