import sys
import os
import time

# 1. System Path Setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.rife_processor import RIFEProcessor
from engine.interpolation import process_video_streaming

# 2. Global Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORT_FRAMES_DIR = os.path.join(DATA_DIR, "report_frames")  # For first 20 frames

# 3. Model paths
RIFE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "engine", "rife")


def run_pipeline(video_filename, save_report_frames=True):
    """
    RIFE Interpolation Pipeline with Streaming Assembly
    
    Args:
        video_filename: Name of video file in the project root
        save_report_frames: If True, saves first 20 frames as PNGs for reporting
    """
    start_time = time.time()
    
    print("\n" + "="*60)
    print("   RIFE SLOW-MOTION PIPELINE")
    print("="*60)
    
    # Setup paths
    video_path = os.path.join(BASE_DIR, video_filename)
    output_path = os.path.join(DATA_DIR, "output_slowmo.mp4")
    
    # Verify input exists
    if not os.path.exists(video_path):
        print(f"\n❌ ERROR: Video file not found: {video_path}")
        print(f"   Please ensure '{video_filename}' exists in the project root.")
        return
    
    try:
        # STEP 1: Initialize RIFE Processor
        print("\n[STEP 1/3] Initializing RIFE Model...")
        print("-" * 60)
        
        processor = RIFEProcessor(model_dir=RIFE_MODEL_DIR, device=None)  # Auto-detect device (now fixed for GPU)
        
        # STEP 2: Setup output directories
        print("\n[STEP 2/3] Setting up output directories...")
        print("-" * 60)
        
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            print(f"  ✓ Created data directory: {DATA_DIR}")
        
        # Clean up old report frames if saving new ones
        if save_report_frames:
            if os.path.exists(REPORT_FRAMES_DIR):
                import shutil
                shutil.rmtree(REPORT_FRAMES_DIR)
            os.makedirs(REPORT_FRAMES_DIR)
            print(f"  ✓ Prepared report frames directory: {REPORT_FRAMES_DIR}")
        
        # STEP 3: Process with streaming assembly
        print("\n[STEP 3/3] Processing Video with Streaming Assembly...")
        print("-" * 60)
        
        stats = process_video_streaming(
            input_path=video_path,
            output_path=output_path,
            processor=processor,
            target_fps_multiplier=2,
            save_png_count=20 if save_report_frames else 0,
            png_output_dir=REPORT_FRAMES_DIR if save_report_frames else None
        )
        
        # Final summary
        end_time = time.time()
        total_duration = end_time - start_time
        
        print("\n" + "="*60)
        print("   PIPELINE COMPLETE")
        print("="*60)
        print(f"  ⏱️  Total Processing Time: {total_duration:.2f} seconds")
        print(f"  📊 Input Frames:  {stats['input_frames']}")
        print(f"  📊 Output Frames: {stats['output_frames']}")
        print(f"  📊 Interpolated:  {stats['interpolated_frames']}")
        print(f"  🎬 Input FPS:  {stats['input_fps']:.2f}")
        print(f"  🎬 Output FPS: {stats['output_fps']:.2f}")
        
        if save_report_frames:
            print(f"  📸 Report PNGs:   {stats['png_saved']} frames saved to {REPORT_FRAMES_DIR}")
        
        print(f"\n  ✅ Output saved to: {output_path}")
        print("="*60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found")
        print(f"   {e}")
        print(f"   Please check the file path and try again.")
        
    except RuntimeError as e:
        print(f"\n❌ ERROR: Runtime error during processing")
        print(f"   {e}")
        print(f"   This might be a model or hardware compatibility issue.")
        
    except Exception as e:
        print(f"\n❌ ERROR: Unexpected error")
        print(f"   {e}")
        import traceback
        print(f"\n   Full traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    # Update this to match your test video filename
    run_pipeline("video.mp4", save_report_frames=True)