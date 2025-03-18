import cv2
import os
import numpy as np
import torch
import timm
import zipfile
import shutil

# Paths
base_path = r"C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model"
downloads_folder = r"C:\Users\tejan\Downloads"
ffpp_real_zip = os.path.join(downloads_folder, "FaceForensics++_real_data_for_DF40.zip")
danet_zip = os.path.join(downloads_folder, "danet_train.zip")
facedancer_zip = os.path.join(downloads_folder, "facedancer_train.zip")
e4s_zip = os.path.join(downloads_folder, "e4s_train.zip")
ffpp_real_extract = os.path.join(base_path, "dataset_ffpp")
danet_extract = os.path.join(base_path, "dataset_danet_train")
facedancer_extract = os.path.join(base_path, "dataset_facedancer_train")
e4s_extract = os.path.join(base_path, "dataset_e4s_train")
output_feature_file = os.path.join(base_path, "features_2048_timm.npy")
output_label_file = os.path.join(base_path, "labels.npy")

# Clean and extract zips
def extract_zip(zip_path, extract_to):
    if os.path.isdir(extract_to):
        print(f"Cleaning {extract_to}...")
        shutil.rmtree(extract_to)
    print(f"Extracting {zip_path} to {extract_to}...")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"{zip_path} not found.")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")
    print(f"Contents of {extract_to}: {os.listdir(extract_to)}")

print("Extracting all zips...")
extract_zip(ffpp_real_zip, ffpp_real_extract)
extract_zip(danet_zip, danet_extract)
extract_zip(facedancer_zip, facedancer_extract)
extract_zip(e4s_zip, e4s_extract)

# Load Xception model
cnn = timm.create_model("xception", pretrained=True, num_classes=0)
cnn.eval()
if torch.cuda.is_available():
    cnn = cnn.cuda()
    print("Using GPU")

# Feature extraction function
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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        print(f"Processed batch of {len(batch_paths)} images")
    return np.array(features)

# Frame extraction from videos
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

# Process all data
features = []
labels = []

# FF++ real data (label 0)
print("\nProcessing FF++ real data...")
ffpp_images = []
for root, _, files in os.walk(ffpp_real_extract):
    print(f"Checking FF++ folder: {root}")
    ffpp_images.extend([os.path.join(root, f) for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])
if ffpp_images:
    print(f"Found {len(ffpp_images)} FF++ real frames")
    ffpp_features = extract_features(ffpp_images, cnn)
    features.append(ffpp_features)
    labels.extend([0] * len(ffpp_features))
else:
    ffpp_videos = []
    for root, _, files in os.walk(ffpp_real_extract):
        print(f"Checking FF++ folder for videos: {root}")
        ffpp_videos.extend([os.path.join(root, f) for f in files if f.endswith(('.mp4', '.avi', '.mov'))])
    if ffpp_videos:
        print(f"Found {len(ffpp_videos)} FF++ real videos")
        for video in ffpp_videos:
            frame_folder = os.path.join(ffpp_real_extract, "frames", os.path.basename(video).split('.')[0])
            frame_paths = extract_frames(video, frame_folder)
            if frame_paths:
                ffpp_features = extract_features(frame_paths, cnn)
                features.append(ffpp_features)
                labels.extend([0] * len(ffpp_features))
    else:
        print(f"No FF++ real data found in {ffpp_real_extract}")

# Danet fake data (label 1)
print("\nProcessing Danet fake data...")
danet_images = []
for root, _, files in os.walk(danet_extract):
    print(f"Checking Danet folder: {root}")
    danet_images.extend([os.path.join(root, f) for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])
if danet_images:
    print(f"Found {len(danet_images)} Danet fake frames")
    danet_features = extract_features(danet_images, cnn)
    features.append(danet_features)
    labels.extend([1] * len(danet_features))
else:
    print(f"No Danet fake data found in {danet_extract}")

# Facedancer fake data (label 1) - frames folder
print("\nProcessing Facedancer fake data...")
facedancer_images = []
for root, _, files in os.walk(facedancer_extract):
    if "frames" in root.lower():  # Match facedancer/frames
        print(f"Checking Facedancer folder: {root}")
        facedancer_images.extend([os.path.join(root, f) for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])
if facedancer_images:
    print(f"Found {len(facedancer_images)} Facedancer fake frames")
    facedancer_features = extract_features(facedancer_images, cnn)
    features.append(facedancer_features)
    labels.extend([1] * len(facedancer_features))
else:
    facedancer_videos = []
    for root, _, files in os.walk(facedancer_extract):
        print(f"Checking Facedancer folder for videos: {root}")
        facedancer_videos.extend([os.path.join(root, f) for f in files if f.endswith(('.mp4', '.avi', '.mov'))])
    if facedancer_videos:
        print(f"Found {len(facedancer_videos)} Facedancer fake videos")
        for video in facedancer_videos:
            frame_folder = os.path.join(facedancer_extract, "frames", os.path.basename(video).split('.')[0])
            frame_paths = extract_frames(video, frame_folder)
            if frame_paths:
                facedancer_features = extract_features(frame_paths, cnn)
                features.append(facedancer_features)
                labels.extend([1] * len(facedancer_features))
    else:
        print(f"No Facedancer fake data found in {facedancer_extract}")

# E4S fake data (label 1) - frames folder
print("\nProcessing E4S fake data...")
e4s_images = []
for root, _, files in os.walk(e4s_extract):
    if "frames" in root.lower():  # Match e4s/frames
        print(f"Checking E4S folder: {root}")
        e4s_images.extend([os.path.join(root, f) for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])
if e4s_images:
    print(f"Found {len(e4s_images)} E4S fake frames")
    e4s_features = extract_features(e4s_images, cnn)
    features.append(e4s_features)
    labels.extend([1] * len(e4s_features))
else:
    e4s_videos = []
    for root, _, files in os.walk(e4s_extract):
        print(f"Checking E4S folder for videos: {root}")
        e4s_videos.extend([os.path.join(root, f) for f in files if f.endswith(('.mp4', '.avi', '.mov'))])
    if e4s_videos:
        print(f"Found {len(e4s_videos)} E4S fake videos")
        for video in e4s_videos:
            frame_folder = os.path.join(e4s_extract, "frames", os.path.basename(video).split('.')[0])
            frame_paths = extract_frames(video, frame_folder)
            if frame_paths:
                e4s_features = extract_features(frame_paths, cnn)
                features.append(e4s_features)
                labels.extend([1] * len(e4s_features))
    else:
        print(f"No E4S fake data found in {e4s_extract}")

# Combine and save
if features:
    features_combined = np.vstack(features)
    labels_combined = np.array(labels, dtype=np.int32)
    np.save(output_feature_file, features_combined)
    np.save(output_label_file, labels_combined)
    print(f"\nSaved features: {features_combined.shape}")
    print(f"Saved labels: {labels_combined.shape}")
    print(f"Label distribution: {np.bincount(labels_combined)}")
else:
    print("No features extracted. Check data paths.")