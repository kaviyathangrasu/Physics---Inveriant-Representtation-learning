import torch

from attention_fusion import AttentionFusion


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ==========================================
# CREATE MODULE
# ==========================================

fusion = AttentionFusion(
    feature_dim=256
).to(device)

print()
print("Attention fusion module created successfully.")


# ==========================================
# SAMPLE FEATURES
# ==========================================

feature1 = torch.randn(
    4,
    256,
    8,
    8
).to(device)

feature2 = torch.randn(
    4,
    256,
    8,
    8
).to(device)


# ==========================================
# FORWARD PASS
# ==========================================

with torch.no_grad():

    fused_feature, attention_map = fusion(
        feature1,
        feature2
    )


# ==========================================
# PRINT SHAPES
# ==========================================

print()
print("Feature1 shape      :", feature1.shape)
print("Feature2 shape      :", feature2.shape)

print(
    "Attention map shape :",
    attention_map.shape
)

print(
    "Fused feature shape :",
    fused_feature.shape
)


# ==========================================
# ATTENTION RANGE
# ==========================================

print()
print(
    "Attention minimum   :",
    attention_map.min().item()
)

print(
    "Attention maximum   :",
    attention_map.max().item()
)


# ==========================================
# SUCCESS CHECK
# ==========================================

if (
    fused_feature.shape == (4, 256, 8, 8)
    and
    attention_map.shape == (4, 256, 8, 8)
):

    print()
    print("======================================")
    print("    ATTENTION FUSION TEST PASSED")
    print("======================================")

else:

    print()
    print("ATTENTION FUSION OUTPUT IS INCORRECT")