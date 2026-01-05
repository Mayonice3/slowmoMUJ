import cv2
import os

#Function to take a video file and saves every image as a png image file

def extract_frames(video_path, output_folder):
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open the Video{video_path}")
        return
    # Getting the metadata from the Video for future reference

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"VIDEO INFO: {fps} fps; {frame_count} Frames.")

    # Creating an output directory if one isn't already made

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print_interval = int(fps) if fps > 0 else 30
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save each Frame as a PNG File

        frame_name = os.path.join(output_folder, f"frame_{count:04d}.png")
        cv2.imwrite(frame_name, frame)

        count += 1

        if count % print_interval == 0:
            percentage = (count/frame_count) * 100
            print(f"Progress: {count}/{frame_count} frames ({percentage:1f}%) ")

    cap.release()
    print(f"Success!!! All Frames saved to {output_folder}")
    return fps

def combine_frames_to_video(input_folder, output_video_path, fps):
   def combine_frames_to_video(input_folder, output_video_path, fps):
    """
    Takes a folder of PNG images and compiles them into a single video file.
    Includes a safety resize to prevent FFmpeg write errors.
    """
    # 1. Get and SORT the images
    if not os.path.exists(input_folder):
        print(f"Error: Folder {input_folder} does not exist.")
        return

    images = [img for img in os.listdir(input_folder) if img.endswith(".png")]
    images.sort()

    if not images:
        print(f"Error: No images found in {input_folder}")
        return

    # 2. Determine target size from the VERY FIRST frame
    first_frame_path = os.path.join(input_folder, images[0])
    first_frame = cv2.imread(first_frame_path)
    
    if first_frame is None:
        print("Error: Could not read the first frame.")
        return
        
    height, width, _ = first_frame.shape
    
    # Use 'mp4v' for Mac compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Rebuilding video: {width}x{height} at {fps} FPS...")

    # 3. The Loop with Safety Resize
    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        frame = cv2.imread(img_path)

        if frame is None:
            continue

        # Resize if there is a dimensional mismatch
        if (frame.shape[1] != width) or (frame.shape[0] != height):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)

        video.write(frame)

    video.release()
    print(f"Success! Video saved to {output_video_path}")

    video.release()
    print(f"Success! Video saved to {output_video_path}")
    
if __name__ == "__main__":
    # Define paths
    input_vid = "data/test_video.mp4" # Ensure this file exists in your data folder!
    output_dir = "data/extracted_frames"
    
    print(f"DEBUG: Starting extraction from {input_vid}...")
    
    # Run the function and capture the return value
    returned_fps = extract_frames(input_vid, output_dir)
    
    if returned_fps:
        print(f"DEBUG: Successfully extracted at {returned_fps} FPS.")
        # Check how many files were actually created
        num_files = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
        print(f"DEBUG: Total files in output folder: {num_files}")
    else:
        print("DEBUG: Extraction failed or returned nothing.")