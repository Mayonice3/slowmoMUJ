import cv2
import numpy as np
import torch

def interpolate_frames_motion(t1, t2):
    # Convert tensors back to Numpy/CV2 format for Optical Flow
    img1 = t1.permute(1, 2, 0).cpu().numpy()
    img2 = t2.permute(1, 2, 0).cpu().numpy()
    
    # 1. Calculate Optical Flow (Farneback)
    # This finds the 'displacement' of every pixel
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # 2. THE FIX: Warp the pixels halfway (t=0.5)
    h, w = flow.shape[:2]
    flow_half = flow * 0.5 # We move the pixels only 50% of the way
    
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow_half[..., 0]).astype(np.float32)
    map_y = (grid_y + flow_half[..., 1]).astype(np.float32)
    
    # Warp Frame 1 forward by 50%
    mid_frame = cv2.remap(img1, map_x, map_y, cv2.INTER_LINEAR)
    
    # 3. Blending (Optional: Mix a bit of Frame 2 to reduce 'ghosting')
    # If this isn't here, you get duplicates.
    mid_tensor = torch.from_numpy(mid_frame).permute(2, 0, 1)
    
    return mid_tensor