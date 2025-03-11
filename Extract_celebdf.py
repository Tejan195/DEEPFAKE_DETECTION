import cv2
import os
import numpy as np
import torch
import timm
import zipfile

# Step 1: Extract Celeb-DF-v2_real_data_for_DF40.zip if dataset folder doesn't exist
downloads_folder = r"C:\Users\tejan\Downloads"
zip_filename = "Celeb-DF-v2_real_data_for_DF40.zip"
zip_path = os.path.join(downloads_folder, zip_filename)

base_folder = os.path.join("C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model", "dataset_celebdf")
if not os.path.isdir(base_folder):
    print(f"Extracting {zip_filename} to {base_folder}...")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"{zip_path} does not exist. Please ensure {zip_filename} is in Downloads.")
    os.makedirs(base_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_folder)
    print(f"Extracted {zip_filename} to {base_folder}")

# Load Xception model using timm
cnn = timm.create_model("xception", pretrained=True, num_classes=0)  # 2048 features
cnn.eval()
if torch.cuda.is_available():
    cnn = cnn.cuda()
else:
    print("CUDA not available, using CPU")

# Output files for features and labels
output_feature_file = os.path.join("C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model", "features_2048_timm.npy")
output_label_file = os.path.join("C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model", "labels.npy")

# Feature extraction function with batching
def extract_features(image_paths, model, batch_size=32):
    features = []
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
            batch_feats = model(batch_imgs).cpu().numpy()  # [batch_size, 2048]
        features.extend(batch_feats)
        print(f"Processed batch of {len(batch_paths)} images")
    return np.array(features)

# Frame extraction function (if videos are present)
def extract_frames(video_path, output_folder, seconds_per_frame=1):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
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
    print(f"Extracted {frame_count} frames from {video_path}")
    return frame_paths

# Step 2: Extract features from Celeb-DF-v2 real data
features = []
labels = []

# Search for frames or videos in Celeb-DF-v2
celebdf_base = os.path.join(base_folder, "Celeb-DF-v2")  # Adjust if folder name differs
if not os.path.exists(celebdf_base):
    celebdf_base = base_folder  # Fallback to root if no Celeb-DF-v2 folder

# Look for pre-extracted frames
frame_extensions = ('.png', '.jpg', '.jpeg')
image_files = []
for root, _, files in os.walk(celebdf_base):
    image_files.extend([os.path.join(root, f) for f in files if f.endswith(frame_extensions)])

if image_files:
    print(f"Found {len(image_files)} real frames in {celebdf_base}")
    real_features = extract_features(image_files, cnn)
    features.append(real_features)
    labels.extend([0] * len(real_features))  # Label 0 for real
else:
    # Look for videos if no frames found
    video_extensions = ('.mp4', '.avi', '.mov')
    video_files = []
    for root, _, files in os.walk(celebdf_base):
        video_files.extend([os.path.join(root, f) for f in files if f.endswith(video_extensions)])
    
    if video_files:
        print(f"Found {len(video_files)} real videos in {celebdf_base}")
        for video_path in video_files:
            frame_folder = os.path.join(base_folder, "celebdf_frames", os.path.basename(video_path).split('.')[0])
            frame_paths = extract_frames(video_path, frame_folder)
            if frame_paths:
                real_features = extract_features(frame_paths, cnn)
                features.append(real_features)
                labels.extend([0] * len(real_features))  # Label 0 for real
    else:
        print(f"No frames or videos found in {celebdf_base}")

# Step 3: Add fake data from existing features_2048_timm.npy
existing_fake_features_path = os.path.join("C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model", "features_2048_timm.npy")
if os.path.exists(existing_fake_features_path):
    fake_features = np.load(existing_fake_features_path)  # Your previous all-fake data
    features.append(fake_features)
    labels.extend([1] * len(fake_features))  # Label 1 for fake
    print(f"Loaded {len(fake_features)} fake features from {existing_fake_features_path}")
else:
    print(f"Warning: {existing_fake_features_path} not found. You need fake data for a balanced dataset.")

# Step 4: Combine and save features and labels
if features:
    features_combined = np.vstack(features)  # Stack all feature arrays
    labels_combined = np.array(labels, dtype=np.int32)
    
    np.save(output_feature_file, features_combined)
    np.save(output_label_file, labels_combined)
    
    print(f"Saved features to {output_feature_file}, shape: {features_combined.shape}, size: {os.path.getsize(output_feature_file) / 1024**2:.2f} MB")
    print(f"Saved labels to {output_label_file}, shape: {labels_combined.shape}")
    print(f"Label distribution: {np.bincount(labels_combined)}")
else:
    print("No features extracted. Check your data paths and contents.")