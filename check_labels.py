import os
import numpy as np
from PIL import Image

BASE_PATH = r"K:\S2Looking\train"

LABEL_DIR = os.path.join(BASE_PATH, "label")
LABEL1_DIR = os.path.join(BASE_PATH, "label1")
LABEL2_DIR = os.path.join(BASE_PATH, "label2")

# Check first 20 common files
label_files = sorted([
    f for f in os.listdir(LABEL_DIR)
    if f.endswith(".png")
])

common_files = [
    f for f in label_files
    if os.path.exists(os.path.join(LABEL1_DIR, f))
    and os.path.exists(os.path.join(LABEL2_DIR, f))
]

print("Total label files:", len(label_files))
print("Common files:", len(common_files))

print()
print("Checking first 20 files...")
print("--------------------------------------")

all_match = True

for filename in common_files[:20]:

    label = np.array(
        Image.open(os.path.join(LABEL_DIR, filename)).convert("L")
    )

    label1 = np.array(
        Image.open(os.path.join(LABEL1_DIR, filename)).convert("L")
    )

    label2 = np.array(
        Image.open(os.path.join(LABEL2_DIR, filename)).convert("L")
    )

    # Convert masks to binary
    label_binary = label > 0
    label1_binary = label1 > 0
    label2_binary = label2 > 0

    # Combine label1 and label2
    combined = label1_binary | label2_binary

    match = np.array_equal(label_binary, combined)

    print(f"{filename:10s} -> {'MATCH' if match else 'NOT MATCH'}")

    if not match:
        all_match = False

print()
print("======================================")

if all_match:
    print("RESULT: LABEL = LABEL1 OR LABEL2")
    print("The three label folders are consistent.")
else:
    print("RESULT: SOME LABELS DO NOT MATCH")
    print("We need to investigate the label definitions.")

print("======================================")