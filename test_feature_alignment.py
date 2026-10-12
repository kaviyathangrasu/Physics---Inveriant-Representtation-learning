import torch
from feature_alignment import FeatureAlignmentLoss


print("======================================")
print("   FEATURE ALIGNMENT LOSS TEST")
print("======================================")

feature1 = torch.randn(4, 256, 8, 8)
feature2 = torch.randn(4, 256, 8, 8)

change_mask = torch.randint(
    0,
    2,
    (4, 1, 256, 256)
).float()

loss_fn = FeatureAlignmentLoss()

loss = loss_fn(
    feature1,
    feature2,
    change_mask
)

print("Feature1 shape :", feature1.shape)
print("Feature2 shape :", feature2.shape)
print("Mask shape     :", change_mask.shape)
print("Loss value     :", loss.item())

if torch.isfinite(loss):
    print("\nFEATURE ALIGNMENT LOSS TEST PASSED")
else:
    print("\nFEATURE ALIGNMENT LOSS TEST FAILED")