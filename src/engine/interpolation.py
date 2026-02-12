"""
Video Interpolation with RIFE - Streaming Assembly
Processes video frames with RIFE interpolation and writes directly to output video
Supports hybrid mode: saves first N frames as PNGs for inspection/reporting
"""

import cv2
import os
import numpy as np
from tqdm import tqdm


def process_video_streaming(
    input_path,
    output_path,
    processor,
    target_fps_multiplier=2,
    output_fps=30.0,
    scale=1.0,  # 1.0 = normal, 2.0 = high quality (2x slower)
    ensemble=False,  # True = TTA Ensemble (2x slower, better quality)
    stop_event=None,
    save_png_count=20,
    png_output_dir=None,
    progress_callback=None
):
    """
    Process video with RIFE interpolation using streaming assembly
    Supports 2x, 4x, 8x, etc. (powers of 2)
    
    Args:
        scale: RIFE scale parameter (1.0=normal, 2.0=high quality for fast motion)
        ensemble: If True, enable TTA ensemble (flip augmentation)
        stop_event: threading.Event to signal cancellation
    """
    
    def get_interpolated_frames(img0, img1, multiplier):
        """Recursive helper to generate power-of-2 intermediate frames"""
        if multiplier <= 1:
            return []
        
        # Check stop signal deep in recursion too?
        if stop_event and stop_event.is_set():
            return []
        
        try:
            # Use standard process_pair with scale parameter
            mid = processor.process_pair(img0, img1, scale=scale, ensemble=ensemble)
        except Exception as e:
            print(f"\n[FALLBACK] Model Failed! Using Blend. Error: {e}")
            mid = cv2.addWeighted(img0, 0.5, img1, 0.5, 0)
            
        # Recursive calls for 4x, 8x, etc.
        # multiplier // 2 frames from left side, then mid, then multiplier // 2 frames from right side
        return get_interpolated_frames(img0, mid, multiplier // 2) + [mid] + get_interpolated_frames(mid, img1, multiplier // 2)

    try:
        # ... (setup code omitted) ...
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {input_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n[VIDEO INFO]")
        print(f"  Input: {input_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Input FPS: {fps:.2f}")
        print(f"  Total Frames: {total_frames}")
        
        # Use provided output_fps
        out_fps = output_fps
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
        
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer for: {output_path}")
        
        print(f"\n[OUTPUT INFO]")
        print(f"  Output: {output_path}")
        print(f"  Target Playback FPS: {out_fps:.2f}")
        print(f"  Multiplier: {target_fps_multiplier}x")
        print(f"  RIFE Scale: {scale}x")
        print(f"  Ensemble: {'ON' if ensemble else 'OFF'}")
        print(f"  Expected frames: ~{total_frames * target_fps_multiplier}")
        
        # Setup PNG export
        png_counter = 0
        if save_png_count > 0:
            if not os.path.exists(png_output_dir):
                os.makedirs(png_output_dir)
            print(f"\n[PNG EXPORT]")
            print(f"  Saving first {save_png_count} frames to: {png_output_dir}")
        
        # Read first frame
        ret, last_frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame from video")
        
        # Write first frame
        writer.write(last_frame)
        if save_png_count > 0 and png_counter < save_png_count:
            cv2.imwrite(os.path.join(png_output_dir, f"frame_{png_counter:04d}_original.png"), last_frame)
            png_counter += 1
        
        # Statistics
        frames_written = 1
        frames_interpolated = 0
        
        # Main processing loop
        print(f"\n[INTERPOLATION] Processing...")
        pbar = tqdm(total=total_frames-1, desc="Interpolating", unit="pair")
        
        while True:
            # Check for stop signal
            if stop_event and stop_event.is_set():
                print("\n[STOP] Processing cancelled by user.")
                break
                
            ret, current_frame = cap.read()
            if not ret:
                break
            
            # Generate multiple interpolated frames (recursive)
            intermediate_frames = get_interpolated_frames(last_frame, current_frame, target_fps_multiplier)
            
            # If interrupted during recursion
            if stop_event and stop_event.is_set():
                print("\n[STOP] Processing cancelled by user.")
                break
            
            # Write intermediate frames
            for i, frame in enumerate(intermediate_frames):
                writer.write(frame)
                frames_written += 1
                frames_interpolated += 1
                
                # Save PNG if within limit
                if save_png_count > 0 and png_counter < save_png_count:
                    # Calculate timestamp for filename (e.g., 0.25, 0.5, 0.75)
                    ts = (i + 1) / target_fps_multiplier
                    png_path = os.path.join(png_output_dir, f"idx{png_counter:04d}_time{ts:.3f}_interpolated.png")
                    cv2.imwrite(png_path, frame)
                    png_counter += 1
            
            # Write the next original frame
            writer.write(current_frame)
            frames_written += 1
            if save_png_count > 0 and png_counter < save_png_count:
                png_path = os.path.join(png_output_dir, f"idx{png_counter:04d}_time1.000_original.png")
                cv2.imwrite(png_path, current_frame)
                png_counter += 1
            
            # Update
            last_frame = current_frame
            pbar.update(1)
            
            if progress_callback:
                percent = pbar.n / pbar.total
                progress_callback(percent)
        
        # Final 100% signal (only if finished naturally)
        if progress_callback and not (stop_event and stop_event.is_set()):
            progress_callback(1.0)
        
        # Cleanup
        pbar.close()
        cap.release()
        writer.release()
        
        if stop_event and stop_event.is_set():
            print(f"\n[CANCELLED]")
            print(f"  ✓ Partial file saved")
            return {'success': False, 'cancelled': True}
        
        print(f"\n[COMPLETE]")
        print(f"  ✓ Frames interpolated: {frames_interpolated}")
        print(f"  ✓ Total frames written: {frames_written}")
        if save_png_count > 0:
            print(f"  ✓ PNG frames saved: {png_counter}")
        print(f"  ✓ Output saved to: {output_path}")
        
        return {
            'input_frames': total_frames,
            'interpolated_frames': frames_interpolated,
            'output_frames': frames_written,
            'input_fps': fps,
            'output_fps': out_fps,
            'png_saved': png_counter,
            'success': True
        }
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        raise
    except RuntimeError as e:
        print(f"\n[ERROR] Runtime error: {e}")
        raise
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during video processing: {e}")
        raise
    finally:
        # Ensure resources are released
        try:
            cap.release()
            writer.release()
        except:
            pass