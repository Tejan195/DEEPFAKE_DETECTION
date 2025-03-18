import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Load test ground truth and predictions
y_test = np.load("C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model/y_test.npy")
test_preds = np.load("C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model/test_preds.npy")  # Fixed!

# Debug lengths
print(f"y_test length: {len(y_test)}, test_preds length: {len(test_preds)}")

# Check if lengths match
if len(y_test) != len(test_preds):
    raise ValueError(f"Length mismatch: y_test ({len(y_test)}) vs test_preds ({len(test_preds)})")

# Compute and print metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds))
print("Classification Report:\n", classification_report(y_test, test_preds))