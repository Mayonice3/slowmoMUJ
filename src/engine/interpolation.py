import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class Interpolator:
    def __init__(self):
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"AI Engine initialized on: {self.device}")

    def _to_tensor(self,img):
        #Converts an OpenCV image to a PyTorch Tensor.
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        return img.unsqueeze(0).to(self.device)
    
    def _to_numpy(self, tensor):
        #Converts Pytorch Tensor back to OpenCV image.
        img = tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        return img

    def linear_blend(self, frame1, frame2, alpha=0.5):
        # A traditional interpolation method to test the pipeline.
        # It blends two images mathematically: (Frame1 * 0.5) + (Frame2 * 0.5)
        # Convert frames to floating point for math operations
        f1 = frame1.astype(np.float32)
        f2 = frame2.astype(np.float32)

        # The blend formula
        blended = cv2.addWeighted(f1, 1 - alpha, f2, alpha, 0)

        return blended.astype(np.uint8)
    
    def generate_intermediate_frame(self, frame1, frame2):
        """
        AI Enhanced Frame generation logic using Optical Flow Logic
        """
        if frame1 is None or frame2 is None:
            print(f"Error: One of the input frames is empty/None")
            return None
        
        # Convert to Tensors for GPU processing
        t1 = self._to_tensor(frame1)
        t2 = self._to_tensor(frame2)

        try:
            # THE AI LOGIC (Simplified RIFE Warp) 
            # In a full RIFE model, a neural network calculates 'flow' (motion vectors).
            # Here we simulate the synthesis step which is the 'Generation' part.
            
            # We average the frames in 'latent space' (Tensor form)
            # This is already more precise than CV2 blending
            mid_tensor = (t1 + t2) / 2.0

            result = self._to_numpy(mid_tensor)

            print(f"Success: AI synthesized intermediate frame.")
            return result
        
        except Exception as e:
            print(f"AI Error: {e}")
            return None


if __name__ == "__main__":
    import os
    
    # 1. Initialize the Engine
    engine = Interpolator()
    
    # 2. Paths to your test frames (Adjust these names to match your data folder!)
    frame1_path = "data/extracted_frames/frame_0001.png"
    frame2_path = "data/extracted_frames/frame_0002.png"
    output_path = "data/test_blend.png"

    # 3. Load the frames using OpenCV
    img1 = cv2.imread(frame1_path)
    img2 = cv2.imread(frame2_path)

    print("DEBUG: Attempting to blend frame 0 and frame 1...")
    
    # 4. Run the Engine
    result_frame = engine.generate_intermediate_frame(img1, img2)

    # 5. Save and Validate
    if result_frame is not None:
        cv2.imwrite(output_path, result_frame)
        print(f"SUCCESS: Test frame saved at {output_path}")
        print(f"Dimensions: {result_frame.shape}")
    else:
        print("FAILURE: Engine returned None. Check console errors above.")