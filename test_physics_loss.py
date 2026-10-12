import torch
from physics_loss import TemporalConsistencyLoss


print("======================================")
print("     PHYSICS LOSS TEST")
print("======================================")

# Simulated feature maps
feature1 = torch.randn(4, 256, 8, 8)
feature2 = torch.randn(4, 256, 8, 8)

# Binary change mask
change_mask = torch.randint(
    0,
    2,
    (4, 1, 256, 256)
).float()

# Create loss
loss_fn = TemporalConsistencyLoss()

# Calculate loss
loss = loss_fn(
    feature1,
    feature2,
    change_mask
)

print("Feature1 shape :", feature1.shape)
print("Feature2 shape :", feature2.shape)
print("Mask shape     :", change_mask.shape)
print("Loss value     :", loss.item())

# Check validity
if torch.isfinite(loss):
    print("\nPHYSICS LOSS TEST PASSED")
else:
    print("\nPHYSICS LOSS TEST FAILED")