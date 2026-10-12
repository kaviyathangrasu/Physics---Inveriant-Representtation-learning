import torch
from dual_head import DisentanglementLoss


print("======================================")
print("    DISENTANGLEMENT LOSS TEST")
print("======================================")

earth_feature = torch.randn(
    4, 256, 8, 8
)

nuisance_feature = torch.randn(
    4, 256, 8, 8
)

loss_fn = DisentanglementLoss()

loss = loss_fn(
    earth_feature,
    nuisance_feature
)

print("Earth feature shape    :", earth_feature.shape)
print("Nuisance feature shape :", nuisance_feature.shape)
print("Loss value             :", loss.item())

if torch.isfinite(loss):
    print("\nDISENTANGLEMENT LOSS TEST PASSED")
else:
    print("\nDISENTANGLEMENT LOSS TEST FAILED")