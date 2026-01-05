import sys
import os
import os
import cv2
import torch

# Add the current directory to the system path so local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your custom functions from the other files
from utils.video_tools import extract_frames, combine_frames_to_video
from utils.image_tools import load_image_as_tensor, save_tensor_as_image
from engine.interpolation import interpolate_frames_motion

# --- DEFINE DIRECTORIES ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EXTRACTED_DIR = os.path.join(DATA_DIR, "extracted_frames")
UPSAMPLED_DIR = os.path.join(DATA_DIR, "upsampled_frames")

# --- SET DEVICE ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

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

def run_interpolation(input_video_path):
    # 1. Prep folders
    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    os.makedirs(UPSAMPLED_DIR, exist_ok=True)

    # 2. Extract frames and get original FPS
    fps = extract_frames(input_video_path, EXTRACTED_DIR)
    frame_files = sorted([f for f in os.listdir(EXTRACTED_DIR) if f.endswith('.png')])

    print(f"Processing {len(frame_files)} frames...")

    # 3. Process the sequence
    for i in range(len(frame_files) - 1):
        f1_path = os.path.join(EXTRACTED_DIR, frame_files[i])
        f2_path = os.path.join(EXTRACTED_DIR, frame_files[i+1])
        
        t1 = load_image_as_tensor(f1_path).to(device)
        t2 = load_image_as_tensor(f2_path).to(device)

        # Save Original (naming it _0)
        save_tensor_as_image(t1, os.path.join(UPSAMPLED_DIR, f"frame_{i:04d}_0.png"))

        # Generate and Save Motion Interpolated (naming it _5)
        mid_tensor = interpolate_frames_motion(t1, t2)
        save_tensor_as_image(mid_tensor, os.path.join(UPSAMPLED_DIR, f"frame_{i:04d}_5.png"))

    # Save the very final frame
    last_f = os.path.join(EXTRACTED_DIR, frame_files[-1])
    last_t = load_image_as_tensor(last_f).to(device)
    save_tensor_as_image(last_t, os.path.join(UPSAMPLED_DIR, f"frame_{len(frame_files)-1:04d}_0.png"))

    # 4. Rebuild Video
    output_path = os.path.join(DATA_DIR, "output_slowmo.mp4")
    combine_frames_to_video(UPSAMPLED_DIR, output_path, fps)
    print(f"\n--- Pipeline Complete: Video is now {len(frame_files)*2/fps:.2f} seconds long ---")
        
    if i % 10 == 0:
        print(f"Progress: {(i/len(frame_files))*100:.1f}% complete...")


if __name__ == "__main__":
    # This points to /Users/mayur/repos/slowmoMUJ/test_video.mp4
    target_video = os.path.join(BASE_DIR, "test_video.mp4")
    
    # Safety Check: Print the path so you can see where it's looking
    print(f"Targeting Video at: {target_video}")
    
    if not os.path.exists(target_video):
        print(f"CRITICAL ERROR: The file '{target_video}' was not found.")
        print("Please ensure your video is in the main 'slowmoMUJ' folder.")
    else:
        run_interpolation(target_video)