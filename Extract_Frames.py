import cv2
import os

def extract_frames(video_path, output_folder, seconds_per_frame=1):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    frame_interval = fps * seconds_per_frame  # Capture every N seconds

    count = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
        count += 1

    cap.release()
    print(f"Extracted {frame_count} frames in '{output_folder}'.")

# Example Usage
video_path = r"C:\Users\tejan\OneDrive\Desktop\Deefake_Detection_Model\deepfake_test.mp4"
output_folder = r"C:\Users\tejan\OneDrive\Desktop\Deefake_Detection_Model\test_frames"
extract_frames(video_path, output_folder, seconds_per_frame=1)
