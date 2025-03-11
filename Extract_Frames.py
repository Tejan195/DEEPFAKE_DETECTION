import cv2
import os
import numpy as np
import torch
import timm
import zipfile

def extract_frames(video_path, output_folder, seconds_per_frame=1):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return []
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps * seconds_per_frame
    
    count = 0
    frame_count = 0
    frame_paths = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_paths.append(frame_filename)
            frame_count += 1
        count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames from video {video_path}.")
    return frame_paths

def extract_zip(zip_path, extract_to):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"{zip_path} does not exist.")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted dataset from {zip_path} to {extract_to}.")

def extract_features(image_paths, model):
    features = []
    batch_size = 32
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_imgs = []
        for img_path in batch_paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load {img_path}")
                continue
            img = cv2.resize(img, (299, 299))
            batch_imgs.append(img)
        
        if not batch_imgs:
            continue
        
        batch_imgs = torch.tensor(np.stack(batch_imgs).transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
        if torch.cuda.is_available():
            batch_imgs = batch_imgs.cuda()
        
        with torch.no_grad():
            batch_feats = model(batch_imgs).cpu().numpy()
        features.extend(batch_feats)
        print(f"Processed batch of {len(batch_paths)} images.")
    
    return np.array(features)

# Main logic
input_path =r"C:\Users\tejan\Downloads\FaceForensics++_real_data_for_DF40.zip" # Change this
output_folder = r"C:\Users\tejan\OneDrive\Desktop\Deefake_Detection_Model\extracted_data" 

cnn = timm.create_model("xception", pretrained=True, num_classes=0)
cnn.eval()
if torch.cuda.is_available():
    cnn = cnn.cuda()
else:
    print("CUDA not available, using CPU")

if input_path.endswith(".zip"):
    extract_zip(input_path, output_folder)
elif input_path.endswith(('.mp4', '.avi', '.mov')):
    frame_folder = os.path.join(output_folder, "video_frames")
    frame_paths = extract_frames(input_path, frame_folder)
    features = extract_features(frame_paths, cnn)
    np.save(os.path.join(output_folder, "video_features.npy"), features)
    print(f"Saved features for {len(frame_paths)} extracted frames.")
else:
    print("Unsupported file type. Provide a .zip (dataset) or a video file.")
