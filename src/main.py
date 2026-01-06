import sys
import os
import shutil
import torch
import time
from tqdm import tqdm # The progress bar library

# 1. System Path Setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.video_tools import extract_frames, combine_frames_to_video
from utils.image_tools import load_image_as_tensor, save_tensor_as_image
from engine.interpolation import interpolate_frames_motion

# 2. Global Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EXTRACTED_DIR = os.path.join(DATA_DIR, "extracted_frames")
UPSAMPLED_DIR = os.path.join(DATA_DIR, "upsampled_frames")

# 3. Hardware Selection (Apple Silicon Support)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def run_pipeline(video_filename):
    start_time = time.time()
    print(f"--- Starting Slow-Mo Pipeline on {device} ---")
    
    # A. CLEANUP: Fresh start to prevent 'jumpy' frames from old runs
    for folder in [EXTRACTED_DIR, UPSAMPLED_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    video_path = os.path.join(BASE_DIR, video_filename)
    
    # B. EXTRACTION
    fps = extract_frames(video_path, EXTRACTED_DIR)
    frame_files = sorted([f for f in os.listdir(EXTRACTED_DIR) if f.endswith('.png')])
    
    if not frame_files:
        print(f"Error: No frames extracted. Check if {video_filename} exists in the root.")
        return

    # C. INTERPOLATION LOOP WITH PROGRESS BAR
    # tqdm wraps the range to create a visual bar in the terminal
    print(f"\nAI Motion Interpolation in progress...")
    for i in tqdm(range(len(frame_files) - 1), desc="Processing Frames", unit="interval"):
        f1_path = os.path.join(EXTRACTED_DIR, frame_files[i])
        f2_path = os.path.join(EXTRACTED_DIR, frame_files[i+1])

        t1 = load_image_as_tensor(f1_path).to(device)
        t2 = load_image_as_tensor(f2_path).to(device)

        # Save Original (_0) and AI-generated Midpoint (_5)
        save_tensor_as_image(t1, os.path.join(UPSAMPLED_DIR, f"frame_{i:04d}_0.png"))
        
        mid_tensor = interpolate_frames_motion(t1, t2)
        save_tensor_as_image(mid_tensor, os.path.join(UPSAMPLED_DIR, f"frame_{i:04d}_5.png"))

    # D. SAVE THE FINAL FRAME
    last_idx = len(frame_files) - 1
    last_f = os.path.join(EXTRACTED_DIR, frame_files[-1])
    last_t = load_image_as_tensor(last_f).to(device)
    save_tensor_as_image(last_t, os.path.join(UPSAMPLED_DIR, f"frame_{last_idx:04d}_0.png"))

    # E. FINAL AUDIT
    upsampled_count = len([f for f in os.listdir(UPSAMPLED_DIR) if f.endswith('.png')])
    expected_count = (len(frame_files) * 2) - 1
    
    print(f"\n--- AUDIT REPORT ---")
    print(f"Expected: {expected_count} | Actual: {upsampled_count}")
    
    if expected_count == upsampled_count:
        print("Result: PASS - Frame sequence is consistent.")
    else:
        print("Result: FAIL - Sequence mismatch detected.")

    # F. REBUILD
    output_path = os.path.join(DATA_DIR, "output_slowmo.mp4")
    combine_frames_to_video(UPSAMPLED_DIR, output_path, fps)
    
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"\n--- Pipeline Complete! ---")
    print(f"Total Processing Time: {total_duration:.2f} seconds")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    run_pipeline("test_video.mp4")