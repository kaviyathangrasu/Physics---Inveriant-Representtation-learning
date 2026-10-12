import torch

from dual_head import DualHeadRepresentation


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ==========================================
# CREATE DUAL HEAD
# ==========================================

dual_head = DualHeadRepresentation(
    in_channels=2048,
    feature_dim=256
).to(device)

print()
print("Dual-head module created successfully.")


# ==========================================
# SIMULATE RESNET-50 FEATURES
# ==========================================

shared_feature = torch.randn(
    4,
    2048,
    8,
    8
).to(device)


# ==========================================
# FORWARD PASS
# ==========================================

with torch.no_grad():

    earth_feature, nuisance_feature = dual_head(
        shared_feature
    )


# ==========================================
# PRINT SHAPES
# ==========================================

print()
print("Shared feature shape   :", shared_feature.shape)
print("Earth feature shape    :", earth_feature.shape)
print("Nuisance feature shape :", nuisance_feature.shape)


# ==========================================
# SUCCESS CHECK
# ==========================================

if (
    earth_feature.shape == (4, 256, 8, 8)
    and
    nuisance_feature.shape == (4, 256, 8, 8)
):

    print()
    print("======================================")
    print("       DUAL-HEAD TEST PASSED")
    print("======================================")

else:

    print()
    print("DUAL-HEAD OUTPUT SHAPE IS INCORRECT")