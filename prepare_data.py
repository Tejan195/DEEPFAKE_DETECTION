import numpy as np
from sklearn.model_selection import train_test_split

# Load features and labels
features = np.load("C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model/features_2048_timm.npy")
labels = np.load("C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model/labels.npy")

# Trim to multiple of 10
seq_length = 10
num_sequences = len(features) // seq_length
features = features[:num_sequences * seq_length]
labels = labels[:num_sequences * seq_length]

# Reshape to [num_sequences, seq_length, features]
features_reshaped = features.reshape(num_sequences, seq_length, 2048)
labels_reshaped = labels[::seq_length]  # Take the last label of each sequence

print(f"Reshaped features shape: {features_reshaped.shape}")  # [1108, 10, 2048]
print(f"Reshaped labels shape: {labels_reshaped.shape}")     # [1108]

# Split into train, validation, test (70%, 15%, 15%)
X_train_val, X_test, y_train_val, y_test = train_test_split(features_reshaped, labels_reshaped, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=42)  # 0.1765 of 85% ≈ 15%

print(f"Train shapes: {X_train.shape}, {y_train.shape}")  # [775, 10, 2048], [775]
print(f"Val shapes: {X_val.shape}, {y_val.shape}")        # [166, 10, 2048], [166]
print(f"Test shapes: {X_test.shape}, {y_test.shape}")     # [167, 10, 2048], [167]

# Save splits
np.save("C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model/X_train.npy", X_train)
np.save("C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model/y_train.npy", y_train)
np.save("C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model/X_val.npy", X_val)
np.save("C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model/y_val.npy", y_val)
np.save("C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model/X_test.npy", X_test)
np.save("C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model/y_test.npy", y_test)
print("Data preparation complete. Files saved.")
