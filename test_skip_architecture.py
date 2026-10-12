import torch

from resnet_encoder import SharedResNet50
from decoder import ChangeDetectionDecoder


print("======================================")
print("      SKIP ARCHITECTURE TEST")
print("======================================")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device :", device)


# ==========================================
# Encoder
# ==========================================

encoder = SharedResNet50(
    pretrained=True
).to(device)

x = torch.randn(
    2,
    3,
    256,
    256
).to(device)

features = encoder(x)

feature1 = features["feature1"]
feature2 = features["feature2"]
feature3 = features["feature3"]
feature4 = features["feature4"]


print("\nEncoder outputs:")

print("Feature1 :", feature1.shape)
print("Feature2 :", feature2.shape)
print("Feature3 :", feature3.shape)
print("Feature4 :", feature4.shape)


# ==========================================
# Simulated fused feature
# ==========================================

fused_feature = torch.randn(
    2,
    256,
    8,
    8
).to(device)


# ==========================================
# Decoder
# ==========================================

decoder = ChangeDetectionDecoder().to(device)

output = decoder(
    fused_feature,
    feature1,
    feature2,
    feature3
)


print("\nDecoder output:")
print("Change map :", output.shape)


# ==========================================
# Test
# ==========================================

expected = (
    2,
    1,
    256,
    256
)

if output.shape == expected:

    print("\nSKIP ARCHITECTURE TEST PASSED")

else:

    print("\nSKIP ARCHITECTURE TEST FAILED")