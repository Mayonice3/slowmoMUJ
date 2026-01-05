import cv2
import numpy as np
import torch

def interpolate_frames_motion(frame1_tensor, frame2_tensor):
    # Convert Tensors to Numpy (RGB)
    f1 = frame1_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    f2 = frame2_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Grayscale for Optical Flow calculation
    gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    
    # Calculate motion vectors (Flow)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    h, w = gray1.shape
    mid_flow = flow * 0.5  # Move halfway
    
    # Create the meshgrid for remapping
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    
    # Shift coordinates by the flow vectors
    map_x = (xx + mid_flow[..., 0]).astype(np.float32)
    map_y = (yy + mid_flow[..., 1]).astype(np.float32)
    
    # Remap the first frame to the new halfway position
    mid_frame = cv2.remap(f1, map_x, map_y, interpolation=cv2.INTER_LANCZOS4)
    
    # Back to Tensor
    mid_tensor = torch.from_numpy(mid_frame).permute(2, 0, 1).unsqueeze(0)
    return mid_tensor