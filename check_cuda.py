#!/usr/bin/env python3
"""Run in the same env as training. Prints whether PyTorch can use your GPU."""
import sys
try:
    import torch
except ImportError:
    print("torch not installed. Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)
cuda_ok = torch.cuda.is_available()
print("PyTorch version:", torch.__version__)
print("CUDA available:", cuda_ok)
if cuda_ok:
    print("Device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}:", torch.cuda.get_device_name(i))
else:
    print("\nYour PyTorch build is CPU-only. To use GPU, reinstall with CUDA support:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print("(Use cu124 for CUDA 12.4 if you prefer. Your driver supports CUDA 13.0.)")
