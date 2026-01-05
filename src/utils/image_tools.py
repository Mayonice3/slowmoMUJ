import torch
import cv2
import numpy as np

def load_image_as_tensor(image_path):
    """Loads an image from disk and converts it to a PyTorch Tensor."""
    # Read image using OpenCV (BGR format)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to Tensor: (H, W, C) -> (C, H, W) and normalize to [0, 1]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img_tensor.unsqueeze(0)  # Add batch dimension: (1, C, H, W)

def save_tensor_as_image(tensor, output_path):
    """Converts a PyTorch Tensor back to an image and saves it to disk."""
    # Remove batch dimension and move to CPU
    img = tensor.squeeze().cpu().numpy()
    
    # (C, H, W) -> (H, W, C)
    img = np.transpose(img, (1, 2, 0))
    
    # Clip values to [0, 1] and scale to [0, 255]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    
    # Convert RGB back to BGR for OpenCV saving
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img)