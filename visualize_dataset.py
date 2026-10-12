from PIL import Image
import matplotlib.pyplot as plt
import os

# ==========================================
# DATASET PATH
# ==========================================

base = r"K:\S2Looking\train"

# Sample filename
filename = "1.png"


# ==========================================
# LOAD IMAGES
# ==========================================

image1 = Image.open(
    os.path.join(base, "Image1", filename)
)

image2 = Image.open(
    os.path.join(base, "Image2", filename)
)

label = Image.open(
    os.path.join(base, "label", filename)
)

label1 = Image.open(
    os.path.join(base, "label1", filename)
)

label2 = Image.open(
    os.path.join(base, "label2", filename)
)


# ==========================================
# DISPLAY DATASET
# ==========================================

plt.figure(figsize=(15, 8))


# Image 1
plt.subplot(2, 3, 1)
plt.imshow(image1)
plt.title("Image 1 - Before")
plt.axis("off")


# Image 2
plt.subplot(2, 3, 2)
plt.imshow(image2)
plt.title("Image 2 - After")
plt.axis("off")


# Ground Truth Label
plt.subplot(2, 3, 3)
plt.imshow(label, cmap="gray")
plt.title("Label - Ground Truth")
plt.axis("off")


# Label 1
plt.subplot(2, 3, 4)
plt.imshow(label1)
plt.title("Label 1")
plt.axis("off")


# Label 2
plt.subplot(2, 3, 5)
plt.imshow(label2)
plt.title("Label 2")
plt.axis("off")


# ==========================================
# SHOW
# ==========================================

plt.tight_layout()
plt.show()