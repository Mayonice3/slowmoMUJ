"""
RIFE Processor - Wrapper for RIFE HDv3 Model
Provides a clean interface for video frame interpolation
"""

import torch
import cv2
import numpy as np
import torch.nn.functional as F
import os


class RIFEProcessor:
    """Wrapper for RIFE HDv3 frame interpolation model"""
    
    def __init__(self, model_dir=None, device=None):
        """
        Initialize RIFE processor
        
        Args:
            model_dir: Directory containing flownet.pkl weights
            device: 'mps', 'cuda', 'cpu', or None for auto-detection
        """
        # Auto-detect device if not specified
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"[RIFE] Initializing on device: {self.device}")
        
        # Set model directory
        if model_dir is None:
            # Default to rife directory in engine
            model_dir = os.path.join(os.path.dirname(__file__), 'rife')
        
        self.model_dir = model_dir
        
        # Import and initialize model with device parameter
        from .rife.RIFE_HDv3 import Model
        self.model = Model(device_override=self.device)
        
        # Load weights
        try:
            self.model.load_model(model_dir, rank=0)
            print(f"[RIFE] Successfully loaded weights from {model_dir}/flownet.pkl")
        except Exception as e:
            print(f"[RIFE] WARNING: Failed to load weights: {e}")
            print(f"[RIFE] Using randomly initialized weights (results will be poor)")
        
        # Enable FP16 (Half Precision) for massive speedup on Apple Silicon / NVIDIA
        self.fp16 = self.device.type in ['mps', 'cuda']
        if self.fp16:
            print(f"[RIFE] Enabling FP16 Half-Precision mode for optimization")
            self.model.flownet.half()
            
        # Set to eval mode (model should already be on correct device)
        self.model.eval()
        
        print(f"[RIFE] Model ready for inference")
    
    def process_pair(self, img0_bgr, img1_bgr, scale=1.0, timestep=0.5):
        """
        Generate interpolated frame between two input frames
        
        Args:
            img0_bgr: First frame as numpy array (H, W, 3) in BGR format
            img1_bgr: Second frame as numpy array (H, W, 3) in BGR format
            scale: Scale factor for processing (1.0=full res, 0.5=half res for speed)
            timestep: Time position of interpolated frame (0.5 = middle)
            
        Returns:
            Interpolated frame as numpy array (H, W, 3) in BGR format
        """
        try:
            # Get original dimensions
            h, w = img0_bgr.shape[:2]
            
            # Convert BGR to RGB and normalize to [0, 1]
            img0_rgb = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)
            img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor [1, 3, H, W]
            I0 = torch.from_numpy(np.transpose(img0_rgb, (2, 0, 1))).unsqueeze(0).float() / 255.0
            I1 = torch.from_numpy(np.transpose(img1_rgb, (2, 0, 1))).unsqueeze(0).float() / 255.0
            
            # Move to device
            I0 = I0.to(self.device)
            I1 = I1.to(self.device)
            
            # Use FP16 if enabled
            if self.fp16:
                I0 = I0.half()
                I1 = I1.half()
            
            # Pad to multiples of 32 (required by architecture)
            tmp = max(32, int(32 / scale))
            ph = ((h - 1) // tmp + 1) * tmp
            pw = ((w - 1) // tmp + 1) * tmp
            padding = (0, pw - w, 0, ph - h)
            
            I0_padded = F.pad(I0, padding)
            I1_padded = F.pad(I1, padding)
            
            # Run inference
            with torch.no_grad():
                middle = self.model.inference(I0_padded, I1_padded, scale=scale)
            
            # Unpad to original size
            middle = middle[:, :, :h, :w]
            
            # Ensure back to float for post-processing if it was half
            if self.fp16:
                middle = middle.float()
            
            # Convert back to numpy BGR
            middle_np = (middle[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)
            middle_bgr = cv2.cvtColor(middle_np, cv2.COLOR_RGB2BGR)
            
            return middle_bgr
            
        except Exception as e:
            print(f"[RIFE] ERROR during inference: {e}")
            raise


if __name__ == "__main__":
    # Simple test
    print("Testing RIFEProcessor...")
    processor = RIFEProcessor()
    
    # Create dummy frames
    frame0 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test interpolation
    result = processor.process_pair(frame0, frame1)
    print(f"Input shape: {frame0.shape}")
    print(f"Output shape: {result.shape}")
    print("✅ Test passed!")
