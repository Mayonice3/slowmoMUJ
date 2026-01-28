import torch
import sys

path = "./src/engine/rife/flownet.pkl"
try:
    checkpoint = torch.load(path, map_location='cpu')
    keys = list(checkpoint.keys())
    print(f"Number of keys: {len(keys)}")
    print("First 20 keys:")
    for k in keys[:20]:
        print(f"  {k}")
except Exception as e:
    print(f"Error loading pbar: {e}")
