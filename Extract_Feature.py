import cv2
import os
import numpy as np
import torch
import timm
import zipfile

# Step 1: Extract faceapp.zip if dataset folder doesn't exist
downloads_folder = os.path.expanduser("~/Downloads")
zip_filename = "faceapp.zip"
zip_path = os.path.join(downloads_folder, zip_filename)

base_folder = os.path.join("C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model", "dataset")
if not os.path.isdir(base_folder):
    print(f"Extracting {zip_filename} to {base_folder}...")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"{zip_path} does not exist. Please ensure faceapp.zip is in Downloads.")
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

# Output file for features
output_feature_file = os.path.join("C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model", "features_2048_timm.npy")

# Extract features from all sets with batching
features = []
labels = []
# Adjust subfolders to match the actual structure
subfolders = ['faceapp/train', 'faceapp/validation', 'faceapp/test']
batch_size = 32  # Adjust for 4GB VRAM

for subfolder in subfolders:
    image_folder = os.path.join(base_folder, subfolder)
    if os.path.isdir(image_folder):
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
        for i in range(0, len(image_files), batch_size):
            batch_paths = image_files[i:i + batch_size]
            batch_imgs = []
            batch_labels = []
            for p in batch_paths:
                img_path = os.path.join(image_folder, p)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load {p}")
                    continue
                img = cv2.resize(img, (299, 299))
                batch_imgs.append(img)
                batch_labels.append(0 if 'real' in p.lower() else 1)  # Placeholder label
            if not batch_imgs:
                continue
            batch_imgs = torch.tensor(np.stack(batch_imgs).transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
            if torch.cuda.is_available():
                batch_imgs = batch_imgs.cuda()
            with torch.no_grad():
                batch_feats = cnn(batch_imgs).cpu().numpy()  # [batch_size, 2048]
            features.extend(batch_feats)
            labels.extend(batch_labels)
            print(f"Processed batch from {subfolder}, {len(batch_paths)} images")
    else:
        print(f"Subfolder {subfolder} not found in {base_folder}")

# Fallback: Search for .png files in the entire dataset directory if subfolders aren't found
if not features:
    print("Falling back to search for .png files in the entire dataset directory...")
    for root, _, files in os.walk(base_folder):
        image_files = [f for f in files if f.endswith('.png')]
        for i in range(0, len(image_files), batch_size):
            batch_paths = image_files[i:i + batch_size]
            batch_imgs = []
            batch_labels = []
            for p in batch_paths:
                img_path = os.path.join(root, p)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load {p}")
                    continue
                img = cv2.resize(img, (299, 299))
                batch_imgs.append(img)
                batch_labels.append(0 if 'real' in p.lower() else 1)  # Placeholder label
            if not batch_imgs:
                continue
            batch_imgs = torch.tensor(np.stack(batch_imgs).transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
            if torch.cuda.is_available():
                batch_imgs = batch_imgs.cuda()
            with torch.no_grad():
                batch_feats = cnn(batch_imgs).cpu().numpy()  # [batch_size, 2048]
            features.extend(batch_feats)
            labels.extend(batch_labels)
            print(f"Processed batch from {root}, {len(batch_paths)} images")

# Save features and labels
if features:
    features = np.array(features)  # [11089, 2048]
    np.save(output_feature_file, features)
    if labels:
        np.save(os.path.join("C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model", "labels.npy"), np.array(labels))
    print(f"Saved features to {output_feature_file}, shape: {features.shape}, size: {os.path.getsize(output_feature_file) / 1024**2:.2f} MB")
else:
    print("No features extracted. No .png files found in dataset or its subfolders.")