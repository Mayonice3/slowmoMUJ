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
    save_png_count=20,
    png_output_dir=None,
    progress_callback=None  # New parameter for UI integration
):
    """
    Process video with RIFE interpolation using streaming assembly
    
    Args:
        input_path: Path to input video file
        output_path: Path to output video file
        processor: RIFEProcessor instance
        target_fps_multiplier: Multiplier for output FPS (2 = double framerate)
        save_png_count: Number of interpolated frames to save as PNG (0 to disable)
        png_output_dir: Directory to save PNG frames (required if save_png_count > 0)
        
    Returns:
        dict: Statistics about the processing (frame_count, fps, etc.)
    """
    
    try:
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
        print(f"  FPS: {fps:.2f}")
        print(f"  Total Frames: {total_frames}")
        
        # Force output FPS to 30 for super slow-motion playback effect
        out_fps = 30.0
        
        # Setup video writer IMMEDIATELY (streaming pattern)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
        
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer for: {output_path}")
        
        print(f"\n[OUTPUT INFO]")
        print(f"  Output: {output_path}")
        print(f"  FPS: {out_fps:.2f} ({target_fps_multiplier})")
        print(f"  Expected frames: ~{total_frames * target_fps_multiplier}")
        
        # Setup PNG export if requested
        png_counter = 0
        if save_png_count > 0:
            if png_output_dir is None:
                raise ValueError("png_output_dir must be specified when save_png_count > 0")
            
            if not os.path.exists(png_output_dir):
                os.makedirs(png_output_dir)
            
            print(f"\n[PNG EXPORT]")
            print(f"  Saving first {save_png_count} frames to: {png_output_dir}")
        
        # Read first frame
        ret, last_frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame from video")
        
        # Write first frame immediately
        writer.write(last_frame)
        
        # Save first frame as PNG if enabled
        if save_png_count > 0 and png_counter < save_png_count:
            png_path = os.path.join(png_output_dir, f"frame_{png_counter:04d}_original.png")
            cv2.imwrite(png_path, last_frame)
            png_counter += 1
        
        # Statistics
        frames_written = 1
        frames_interpolated = 0
        
        # Main processing loop with progress bar
        print(f"\n[INTERPOLATION] Processing...")
        pbar = tqdm(total=total_frames-1, desc="Interpolating", unit="pair")
        
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break
            
            # 1. Generate interpolated middle frame
            try:
                mid_frame = processor.process_pair(last_frame, current_frame, scale=1.0)
                frames_interpolated += 1
            except Exception as e:
                print(f"\n[WARNING] Interpolation failed, using blend fallback: {e}")
                # Fallback: simple blend if RIFE fails
                mid_frame = cv2.addWeighted(last_frame, 0.5, current_frame, 0.5, 0)
            
            # 2. WRITE IMMEDIATELY (Streaming Assembly - Critical!)
            writer.write(mid_frame)
            writer.write(current_frame)
            frames_written += 2
            
            # 3. Save as PNG if within limit
            if save_png_count > 0 and png_counter < save_png_count:
                png_path = os.path.join(png_output_dir, f"frame_{png_counter:04d}_interpolated.png")
                cv2.imwrite(png_path, mid_frame)
                png_counter += 1
                
                if png_counter < save_png_count:
                    png_path = os.path.join(png_output_dir, f"frame_{png_counter:04d}_original.png")
                    cv2.imwrite(png_path, current_frame)
                    png_counter += 1
            
            # Update for next iteration
            last_frame = current_frame
            pbar.update(1)
            
            # Send progress update to UI based on pbar count
            if progress_callback:
                percent = pbar.n / pbar.total
                progress_callback(percent)
        
        # Final 100% signal
        if progress_callback:
            progress_callback(1.0)
        
        # Cleanup
        pbar.close()
        cap.release()
        writer.release()
        
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


# Keep legacy function for backward compatibility (deprecated)
def interpolate_frames_rife(t1, t2, timestep=0.5):
    """
    DEPRECATED: Legacy tensor-based interpolation
    This is kept for backward compatibility but should not be used in new code
    Use RIFEProcessor.process_pair() instead
    """
    print("[WARNING] Using deprecated interpolate_frames_rife(). Consider using RIFEProcessor instead.")
    
    # Simplified implementation - just return average
    # Real implementation would need a global processor instance
    return (t1 + t2) / 2