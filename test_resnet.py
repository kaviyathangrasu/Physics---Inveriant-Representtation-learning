import torch

from resnet_encoder import SharedResNet50


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ==========================================
# CREATE SHARED RESNET-50
# ==========================================

model = SharedResNet50(
    pretrained=True
).to(device)


print()
print("Shared ResNet-50 created successfully.")


# ==========================================
# CREATE TWO SAMPLE IMAGES
# ==========================================

image1 = torch.randn(
    4,
    3,
    256,
    256
).to(device)


image2 = torch.randn(
    4,
    3,
    256,
    256
).to(device)


# ==========================================
# EXTRACT FEATURES
# ==========================================

with torch.no_grad():

    feature1 = model(image1)

    feature2 = model(image2)


# ==========================================
# PRINT SHAPES
# ==========================================

print()
print("Image1 shape :", image1.shape)
print("Image2 shape :", image2.shape)

print()
print("Feature1 shape :", feature1.shape)
print("Feature2 shape :", feature2.shape)


# ==========================================
# CHECK SHARED FEATURES
# ==========================================

difference = torch.abs(
    feature1 - feature2
)

print()
print("Difference shape :", difference.shape)


# ==========================================
# SUCCESS CHECK
# ==========================================

if (
    feature1.shape == (4, 2048, 8, 8)
    and
    feature2.shape == (4, 2048, 8, 8)
):

    print()
    print("======================================")
    print("     RESNET-50 TEST PASSED")
    print("======================================")

else:

    print()
    print("RESNET-50 OUTPUT SHAPE IS INCORRECT")