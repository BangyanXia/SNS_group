import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")

import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


x = torch.randn(10000, 10000)


start = time.time()
y_cpu = x.matmul(x)
print(f"CPU time: {time.time() - start:.4f} seconds")


x = x.to(device)
start = time.time()
y_gpu = x.matmul(x)
print(f"GPU time: {time.time() - start:.4f} seconds")
