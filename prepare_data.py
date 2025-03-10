import numpy as np
from sklearn.model_selection import train_test_split
import os

# Paths
dataset_path = "C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model"
video_feature_path = os.path.join(dataset_path, "extracted_data", "video_features.npy")

# Load dataset features and labels
features = np.load(os.path.join(dataset_path, "features_2048_timm.npy"))
labels = np.load(os.path.join(dataset_path, "labels.npy"))

# Load video features
video_features = np.load(video_feature_path)

# Assign placeholder labels for video data (-1 for inference)
video_labels = np.full((video_features.shape[0],), -1)  

# Merge dataset and video features
X_combined = np.vstack((features, video_features))
y_combined = np.concatenate((labels, video_labels))

# Sequence settings
seq_length = 10  
num_sequences = len(X_combined) // seq_length  

# Trim to multiple of seq_length
X_combined = X_combined[:num_sequences * seq_length]
y_combined = y_combined[:num_sequences * seq_length]

# Reshape for LSTM [num_sequences, seq_length, 2048]
X_reshaped = X_combined.reshape(num_sequences, seq_length, 2048)
y_reshaped = y_combined[::seq_length]  # Take 1 label per sequence

print(f"Reshaped features shape: {X_reshaped.shape}")  
print(f"Reshaped labels shape: {y_reshaped.shape}")  

# Separate video frames (test set where labels == -1)
X_test_video = X_reshaped[y_reshaped == -1]  # For inference only
X_train_val = X_reshaped[y_reshaped != -1]
y_train_val = y_reshaped[y_reshaped != -1]

# Train-validation-test split (excluding video features)
X_train, X_test, y_train, y_test = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1765, random_state=42)  

print(f"Train shapes: {X_train.shape}, {y_train.shape}")  
print(f"Val shapes: {X_val.shape}, {y_val.shape}")  
print(f"Test shapes: {X_test.shape}, {y_test.shape}")  
print(f"Video test shape: {X_test_video.shape}")  

# Save splits
np.save(os.path.join(dataset_path, "X_train.npy"), X_train)
np.save(os.path.join(dataset_path, "y_train.npy"), y_train)
np.save(os.path.join(dataset_path, "X_val.npy"), X_val)
np.save(os.path.join(dataset_path, "y_val.npy"), y_val)
np.save(os.path.join(dataset_path, "X_test.npy"), X_test)
np.save(os.path.join(dataset_path, "y_test.npy"), y_test)

# Save video feature set for inference
np.save(os.path.join(dataset_path, "X_test_video.npy"), X_test_video)

print("Data preparation complete. Files saved.")
