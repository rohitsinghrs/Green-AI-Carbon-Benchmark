import torch

print("=" * 60)
print("PYTORCH HARDWARE CHECK")
print("=" * 60)

print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print("CUDA available: YES")
    print("GPU:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    print("CUDA available: NO")
    print("Using CPU")
    device = torch.device("cpu")

print("Selected device:", device)

print("=" * 60)