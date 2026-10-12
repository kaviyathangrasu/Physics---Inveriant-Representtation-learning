import torch

from decoder import ChangeDetectionDecoder


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# Create decoder
decoder = ChangeDetectionDecoder(
    input_channels=256
).to(device)

print()
print("Decoder created successfully.")

# Simulated fused feature
fused_feature = torch.randn(
    4,
    256,
    8,
    8
).to(device)

print()
print("Input feature shape :", fused_feature.shape)

# Forward pass
with torch.no_grad():

    output = decoder(fused_feature)

print()
print("Decoder output shape:", output.shape)

# Convert logits to probability
probability = torch.sigmoid(output)

print()
print("Probability minimum :", probability.min().item())
print("Probability maximum :", probability.max().item())

# Verify
if output.shape == (4, 1, 256, 256):

    print()
    print("======================================")
    print("       DECODER TEST PASSED")
    print("======================================")

else:

    print()
    print("DECODER OUTPUT IS INCORRECT")