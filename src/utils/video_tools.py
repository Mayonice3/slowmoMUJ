import cv2
import os

def extract_frames(video_path, output_folder):
    #Extracts every frame from a video file and saves them as PNGs.
    #Returns the original FPS of the video.
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open the video file at {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"VIDEO INFO: {fps} FPS; {frame_count} Frames.")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save each frame with 4-digit padding (frame_0000.png)
        frame_name = os.path.join(output_folder, f"frame_{count:04d}.png")
        cv2.imwrite(frame_name, frame)
        count += 1

    cap.release()
    print(f"Success: {count} frames saved to {output_folder}")
    return fps


def combine_frames_to_video(input_folder, output_video_path, fps):
    """
    Compiles PNG images into a video.
    Uses target_fps = fps to ensure a smooth slow-motion effect.
    """
    if not os.path.exists(input_folder):
        print(f"Error: Folder {input_folder} does not exist.")
        return

    # Sort images alphabetically to maintain temporal order
    images = sorted([img for img in os.listdir(input_folder) if img.endswith(".png")])

    if not images:
        print(f"Error: No images found in {input_folder}")
        return

    # Get dimensions from the first frame
    first_frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, _ = first_frame.shape
    
    #Change this parameter to adjust fps of output video
    target_fps = fps 
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(output_video_path, fourcc, target_fps, (width, height))

    print(f"Rebuilding Video: {len(images)} frames at {target_fps} FPS...")

    for img_name in images:
        frame = cv2.imread(os.path.join(input_folder, img_name))
        if frame is None:
            continue

        # Safety Resize: Ensures every frame matches the video container size
        if (frame.shape[1] != width) or (frame.shape[0] != height):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)

        video.write(frame)

    video.release()
    print(f"Success! Final video compiled at: {output_video_path}")


if __name__ == "__main__":
    test_vid = "data/test_video.mp4"
    test_out = "data/extracted_frames"
    extract_frames(test_vid, test_out)