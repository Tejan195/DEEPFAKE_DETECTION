import numpy as np

# Paths to label file
label_file = "C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model/labels.npy"

# Load labels
labels = np.load(label_file)

# Count real (0) and fake (1) samples
real_count = np.sum(labels == 0)
fake_count = np.sum(labels == 1)

# Print results
print(f"Number of real photos: {real_count}")
print(f"Number of fake photos: {fake_count}")
print(f"Total photos: {real_count + fake_count}")
print(f"Fake to real ratio: {fake_count / real_count:.2f}:1")