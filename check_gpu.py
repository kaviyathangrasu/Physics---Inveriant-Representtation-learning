import torch

print("======================================")
print("             GPU CHECK")
print("======================================")

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU count       :", torch.cuda.device_count())
    print("GPU name        :", torch.cuda.get_device_name(0))
    print("CUDA version    :", torch.version.cuda)

    print("\nGPU IS AVAILABLE ✅")
else:
    print("\nGPU IS NOT AVAILABLE ⚠️")