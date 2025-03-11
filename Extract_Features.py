import cv2
import os
import numpy as np
import torch
import timm
import zipfile

# Paths
downloads_folder = os.path.expanduser("~/Downloads")
ffpp_zip_path = os.path.join(downloads_folder, "FaceForensics++_real_data_for_DF40.zip")
celebdf_zip_path = os.path.join(downloads_folder, "Celeb-DF-v2_real_data_for_DF40.zip")
danet_zip_path = os.path.join(downloads_folder, "danet.zip")
base_folder = os.path.join("C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model", "dataset")
output_path = "C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model"

# Step 1: Extract zips if dataset folder doesn't exist
if not os.path.isdir(base_folder):
    os.makedirs(base_folder, exist_ok=True)
    for zip_path, name in [(ffpp_zip_path, "FF++ real"), (celebdf_zip_path, "Celeb-DF"), (danet_zip_path, "danet")]:
        if not os.path.exists(zip_path):
            print(f"Warning: {zip_path} not found. Ensure {name} zip is in Downloads.")
            continue
        print(f"Extracting {name} to {base_folder}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_folder)
        print(f"Extracted {name} to {base_folder}")

# Load Xception model
cnn = timm.create_model("xception", pretrained=True, num_classes=0)
cnn.eval()
if torch.cuda.is_available():
    cnn = cnn.cuda()
else:
    print("CUDA not available, using CPU")

# Output files
output_feature_file = os.path.join(output_path, "features_2048_timm.npy")
output_label_file = os.path.join(output_path, "labels.npy")

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

# Step 2: Process real and fake data
features = []
labels = []

# FF++ real frames
ffpp_real_dir = os.path.join(base_folder, "FaceForensics++", "original_sequences", "youtube", "c23", "frames")
if os.path.isdir(ffpp_real_dir):
    ffpp_real_images = [f for f in os.listdir(ffpp_real_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if ffpp_real_images:
        ffpp_real_paths = [os.path.join(ffpp_real_dir, f) for f in ffpp_real_images]
        print(f"Found {len(ffpp_real_paths)} real frames in {ffpp_real_dir}")
        ffpp_real_features = extract_features(ffpp_real_paths, cnn)
        features.append(ffpp_real_features)
        labels.extend([0] * len(ffpp_real_features))
    else:
        print(f"No images found in {ffpp_real_dir}")
else:
    print(f"FF++ real directory {ffpp_real_dir} not found")

# Celeb-DF real images (placeholder path, adjust after extraction)
celebdf_real_dir = os.path.join(base_folder, "Celeb-DF-v2_real_data_for_DF40", "Celeb-real")  # Adjust based on actual structure
if os.path.isdir(celebdf_real_dir):
    celebdf_real_images = [f for f in os.listdir(celebdf_real_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if celebdf_real_images:
        celebdf_real_paths = [os.path.join(celebdf_real_dir, f) for f in celebdf_real_images]
        print(f"Found {len(celebdf_real_paths)} real images in {celebdf_real_dir}")
        celebdf_real_features = extract_features(celebdf_real_paths, cnn)
        features.append(celebdf_real_features)
        labels.extend([0] * len(celebdf_real_features))
    else:
        print(f"No images found in {celebdf_real_dir}")
else:
    print(f"Celeb-DF real directory {celebdf_real_dir} not found. Adjust path after extraction.")

# danet fake images
danet_dir = os.path.join(base_folder, "danet")  # Adjust based on actual structure
if os.path.isdir(danet_dir):
    danet_image_paths = []
    for root, _, files in os.walk(danet_dir):
        danet_image_paths.extend(os.path.join(root, f) for f in files if f.endswith(('.png', '.jpg', '.jpeg')))
    if danet_image_paths:
        print(f"Found {len(danet_image_paths)} fake images in {danet_dir}")
        danet_features = extract_features(danet_image_paths, cnn)
        features.append(danet_features)
        labels.extend([1] * len(danet_features))
    else:
        print(f"No images found in {danet_dir}")
else:
    print(f"danet directory {danet_dir} not found. Adjust path after extraction.")

# Step 3: Combine and save
if features:
    features_combined = np.vstack(features)
    labels_combined = np.array(labels, dtype=np.int32)
    
    np.save(output_feature_file, features_combined)
    np.save(output_label_file, labels_combined)
    
    print(f"Saved features to {output_feature_file}, shape: {features_combined.shape}, size: {os.path.getsize(output_feature_file) / 1024**2:.2f} MB")
    print(f"Saved labels to {output_label_file}, shape: {labels_combined.shape}")
    print(f"Label distribution: {np.bincount(labels_combined)}")
else:
    print("No features extracted. Check your data paths and contents.")