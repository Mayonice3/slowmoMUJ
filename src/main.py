import os
import cv2
import torch
import sys
from utils.video_tools import extract_frames, combine_frames_to_video
from engine.interpolation import Interpolator

def check_hardware():
    """Validates the Mac GPU environment before starting."""
    print("--- Pre-Runtime System Check ---")
    print(f"Python: {sys.version.split()[0]} | PyTorch: {torch.__version__}")
    
    if torch.backends.mps.is_available():
        print("Hardware Acceleration: Apple Silicon GPU (MPS) DETECTED.")
        return True
    else:
        print("Hardware Acceleration: NOT DETECTED. Using CPU (Warning: Processing will be slow).")
        return False

def run_upsampling_pipeline(video_name):
    # 1. Hardware Check
    has_gpu = check_hardware()
    
    # 2. Setup Paths
    project_root = os.getcwd()
    input_path = os.path.join(project_root, "data", video_name)
    frames_dir = os.path.join(project_root, "data", "extracted_frames")
    upsampled_dir = os.path.join(project_root, "data", "upsampled_frames")
    output_video = os.path.join(project_root, "data", "output_60fps.mp4")

    if not os.path.exists(upsampled_dir):
        os.makedirs(upsampled_dir)

    # 3. Extract original frames
    fps = extract_frames(input_path, frames_dir)
    if fps is None:
        print("Pipeline aborted: Extraction failed.")
        return
    
    engine = Interpolator()

    # 5. Process Frames
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    print(f"\nStarting interpolation of {len(frame_files)} frames...")

    for i in range(len(frame_files) - 1):
        img1 = cv2.imread(os.path.join(frames_dir, frame_files[i]))
        img2 = cv2.imread(os.path.join(frames_dir, frame_files[i+1]))

        # Original
        cv2.imwrite(os.path.join(upsampled_dir, f"out_{i*2:04d}.png"), img1)

        # Intermediate
        mid_frame = engine.generate_intermediate_frame(img1, img2)
        if mid_frame is not None:
            cv2.imwrite(os.path.join(upsampled_dir, f"out_{i*2+1:04d}.png"), mid_frame)
        
        if i % 10 == 0:
            print(f"Progress: {(i/len(frame_files))*100:.1f}% complete...")

    # 6. Final Video (Slow Motion Mode)
    # We use the ORIGINAL 'fps' here so the video plays it back 
    # the doubled frame count at the original speed.
    combine_frames_to_video(upsampled_dir, output_video, fps) 
    print(f"\n--- Pipeline Complete: Video is now {len(frame_files)*2/fps:.2f} seconds long ---")

if __name__ == "__main__":
    run_upsampling_pipeline("test_video.mp4")