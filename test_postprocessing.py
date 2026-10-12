import torch

from postprocessing import ChangePostProcessor


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# Create post-processor
processor = ChangePostProcessor(
    threshold=0.5
)

print()
print("Post-processing module created successfully.")

# Simulated decoder output
logits = torch.randn(
    4,
    1,
    256,
    256
).to(device)

print()
print("Input logits shape:", logits.shape)

# Post-processing
probability, binary_mask = processor.forward(logits)

print()
print("Probability shape :", probability.shape)
print("Binary mask shape :", binary_mask.shape)

print()
print("Probability minimum:", probability.min().item())
print("Probability maximum:", probability.max().item())

# Calculate change ratio
change_ratio = processor.calculate_change_ratio(
    binary_mask
)

print()
print("Change ratio for each image:")

for i, ratio in enumerate(change_ratio):

    print(
        f"Image {i + 1}: "
        f"{ratio.item():.2f}%"
    )

# Check output
if (
    probability.shape == (4, 1, 256, 256)
    and
    binary_mask.shape == (4, 1, 256, 256)
):

    print()
    print("======================================")
    print("   POST-PROCESSING TEST PASSED")
    print("======================================")

else:

    print()
    print("POST-PROCESSING OUTPUT IS INCORRECT")