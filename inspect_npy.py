import numpy as np
import os
# Define paths to your .npy files
base_path = "C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model"
files_to_check = {
    "features_2048_timm.npy": os.path.join(base_path, "features_2048_timm.npy"),
    "labels.npy": os.path.join(base_path, "labels.npy"),
    "X_train.npy": os.path.join(base_path, "X_train.npy"),
    "y_train.npy": os.path.join(base_path, "y_train.npy"),
    "X_val.npy": os.path.join(base_path, "X_val.npy"),
    "y_val.npy": os.path.join(base_path, "y_val.npy"),
    "X_test.npy": os.path.join(base_path, "X_test.npy"),
    "y_test.npy": os.path.join(base_path, "y_test.npy"),
    "video_features.npy": os.path.join(base_path, "extracted_data", "video_features.npy"),
    "X_test_video.npy": os.path.join(base_path, "X_test_video.npy")
}

# Function to summarize .npy file contents
def summarize_npy(file_path, file_name):
    try:
        data = np.load(file_path)
        print(f"\n=== {file_name} ===")
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        
        # For labels, show distribution
        if "label" in file_name.lower() or "y_" in file_name:
            unique, counts = np.unique(data, return_counts=True)
            print(f"Label distribution: {dict(zip(unique, counts))}")
        # For features, show a sample (first few values)
        else:
            print(f"Sample (first 5 values, flattened): {data.flatten()[:5]}")
            print(f"Min value: {data.min()}, Max value: {data.max()}")
    except Exception as e:
        print(f"Error loading {file_name}: {e}")

# Loop through and summarize each file
for name, path in files_to_check.items():
    summarize_npy(path, name)
