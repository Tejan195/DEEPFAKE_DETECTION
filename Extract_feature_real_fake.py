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
celebdf_zip = os.path.join(downloads_folder, "Celeb-DF-v2_real_data_for_DF40.zip")
danet_zip = os.path.join(downloads_folder, "danet.zip")
facedancer_zip = os.path.join(downloads_folder, "facedancer.zip")
e4s_zip = os.path.join(downloads_folder, "e4s.zip")
ffpp_real_extract = os.path.join(base_path, "dataset_ffpp")
celebdf_extract = os.path.join(base_path, "dataset_celebdf")
danet_extract = os.path.join(base_path, "dataset_danet")
facedancer_extract = os.path.join(base_path, "dataset_facedancer")
e4s_extract = os.path.join(base_path, "dataset_e4s")
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
extract_zip(celebdf_zip, celebdf_extract)
extract_zip(danet_zip, danet_extract)
extract_zip(facedancer_zip, facedancer_extract)
extract_zip(e4s_zip, e4s_extract)

# Load Xception model
cnn = timm.create_model("xception", pretrained=True, num_classes=0)
cnn.eval()
if torch.cuda.is_available():
    cnn = cnn.cuda()
    print("Using GPU")

# Feature extraction function with BGR to RGB fix
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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
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

# Frame extraction from videos (fallback)
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

# Celeb-DF-v2 real data (label 0)
print("\nProcessing Celeb-DF-v2 real data...")
celebdf_images = []
for root, _, files in os.walk(celebdf_extract):
    print(f"Checking Celeb-DF folder: {root}")
    celebdf_images.extend([os.path.join(root, f) for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])
if celebdf_images:
    print(f"Found {len(celebdf_images)} Celeb-DF-v2 real frames")
    celebdf_features = extract_features(celebdf_images, cnn)
    features.append(celebdf_features)
    labels.extend([0] * len(celebdf_features))
else:
    celebdf_videos = []
    for root, _, files in os.walk(celebdf_extract):
        print(f"Checking Celeb-DF folder for videos: {root}")
        celebdf_videos.extend([os.path.join(root, f) for f in files if f.endswith(('.mp4', '.avi', '.mov'))])
    if celebdf_videos:
        print(f"Found {len(celebdf_videos)} Celeb-DF-v2 real videos")
        for video in celebdf_videos:
            frame_folder = os.path.join(celebdf_extract, "frames", os.path.basename(video).split('.')[0])
            frame_paths = extract_frames(video, frame_folder)
            if frame_paths:
                celebdf_features = extract_features(frame_paths, cnn)
                features.append(celebdf_features)
                labels.extend([0] * len(celebdf_features))
    else:
        print(f"No Celeb-DF-v2 real data found in {celebdf_extract}")

# Danet fake data (label 1)
print("\nProcessing danet fake data...")
danet_images = []
for root, _, files in os.walk(danet_extract):
    print(f"Checking danet folder: {root}")
    danet_images.extend([os.path.join(root, f) for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])
if danet_images:
    print(f"Found {len(danet_images)} danet fake frames")
    danet_features = extract_features(danet_images, cnn)
    features.append(danet_features)
    labels.extend([1] * len(danet_features))
else:
    print(f"No danet fake data found in {danet_extract}")

# Facedancer fake data (label 1) - cdf/frames
print("\nProcessing facedancer fake data (cdf/frames)...")
facedancer_images = []
for root, _, files in os.walk(facedancer_extract):
    if "cdf" in root.lower() and "frames" in root.lower():  # Target cdf/frames
        print(f"Checking facedancer folder: {root}")
        facedancer_images.extend([os.path.join(root, f) for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])
if facedancer_images:
    print(f"Found {len(facedancer_images)} facedancer fake frames")
    facedancer_features = extract_features(facedancer_images, cnn)
    features.append(facedancer_features)
    labels.extend([1] * len(facedancer_features))
else:
    facedancer_videos = []
    for root, _, files in os.walk(facedancer_extract):
        if "cdf" in root.lower():  # Check cdf for videos
            print(f"Checking facedancer folder for videos: {root}")
            facedancer_videos.extend([os.path.join(root, f) for f in files if f.endswith(('.mp4', '.avi', '.mov'))])
    if facedancer_videos:
        print(f"Found {len(facedancer_videos)} facedancer fake videos")
        for video in facedancer_videos:
            frame_folder = os.path.join(facedancer_extract, "frames", os.path.basename(video).split('.')[0])
            frame_paths = extract_frames(video, frame_folder)
            if frame_paths:
                facedancer_features = extract_features(frame_paths, cnn)
                features.append(facedancer_features)
                labels.extend([1] * len(facedancer_features))
    else:
        print(f"No facedancer fake data found in {facedancer_extract}/cdf/frames")

# E4S fake data (label 1) - cdf/frames and ff/frames
print("\nProcessing e4s fake data (cdf/frames and ff/frames)...")
e4s_images = []
for root, _, files in os.walk(e4s_extract):
    if ("cdf" in root.lower() or "ff" in root.lower()) and "frames" in root.lower():  # Target cdf/frames or ff/frames
        print(f"Checking e4s folder: {root}")
        e4s_images.extend([os.path.join(root, f) for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])
if e4s_images:
    print(f"Found {len(e4s_images)} e4s fake frames")
    e4s_features = extract_features(e4s_images, cnn)
    features.append(e4s_features)
    labels.extend([1] * len(e4s_features))
else:
    e4s_videos = []
    for root, _, files in os.walk(e4s_extract):
        if "cdf" in root.lower() or "ff" in root.lower():  # Check cdf or ff for videos
            print(f"Checking e4s folder for videos: {root}")
            e4s_videos.extend([os.path.join(root, f) for f in files if f.endswith(('.mp4', '.avi', '.mov'))])
    if e4s_videos:
        print(f"Found {len(e4s_videos)} e4s fake videos")
        for video in e4s_videos:
            frame_folder = os.path.join(e4s_extract, "frames", os.path.basename(video).split('.')[0])
            frame_paths = extract_frames(video, frame_folder)
            if frame_paths:
                e4s_features = extract_features(frame_paths, cnn)
                features.append(e4s_features)
                labels.extend([1] * len(e4s_features))
    else:
        print(f"No e4s fake data found in {e4s_extract}/cdf/frames or ff/frames")

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