import torch
import os

model_path = "/Users/mayur/repos/slowmoMUJ/src/engine/rife/flownet.pkl"

try:
    print(f"Loading {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    keys = []
    # Handle state_dict vs full model
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            keys = list(checkpoint['state_dict'].keys())
        else:
            keys = list(checkpoint.keys())
    else:
        keys = list(checkpoint.state_dict().keys())
        
    print(f"\nTotal keys: {len(keys)}")
    
    # Check for specific modules
    has_contextnet = any('contextnet' in k for k in keys)
    has_unet = any('unet' in k for k in keys)
    has_encode = any('encode' in k for k in keys)
    
    print(f"\n--- Feature Check ---")
    print(f"  Has 'encode' (Head): {has_encode}")
    print(f"  Has 'contextnet':    {has_contextnet}")
    print(f"  Has 'unet':          {has_unet}")
    
    # Print sample keys
    print(f"\nSample keys:")
    for k in keys[:5]:
        print(f"  {k}")
        
except Exception as e:
    print(f"Error: {e}")
