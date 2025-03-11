import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

y_test = np.load("C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model/y_test.npy")
test_preds = np.load("C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model/test_preds.npy")  # From train_lstm.py
print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds))
print("Classification Report:\n", classification_report(y_test, test_preds))