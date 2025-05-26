import os
import numpy as np
from collections import Counter
base_path = r"C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model"
y_test_path = os.path.join(base_path, "y_test.npy")
y_test = np.load(y_test_path)
print(f"Test set size: {len(y_test)}")
print(f"Test labels distribution: {Counter(y_test)}")