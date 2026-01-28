import torch
import sys

path = "./data/train_log/flownet.pkl"
try:
    checkpoint = torch.load(path, map_location='cpu')
    print(f"Type of checkpoint: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        keys = list(checkpoint.keys())
        print(f"Number of keys: {len(keys)}")
        print("First 20 keys:")
        for k in keys[:20]:
            print(f"  {k}")
    else:
        print("Checkpoint is not a dictionary.")
        
except Exception as e:
    print(f"Error loading pkl: {e}")
