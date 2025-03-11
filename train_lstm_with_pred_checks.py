import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# Paths
base_path = r"C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model"
X_train_path = os.path.join(base_path, "X_train.npy")
y_train_path = os.path.join(base_path, "y_train.npy")
X_val_path = os.path.join(base_path, "X_val.npy")
y_val_path = os.path.join(base_path, "y_val.npy")
X_test_path = os.path.join(base_path, "X_test.npy")
y_test_path = os.path.join(base_path, "y_test.npy")

# Load data
X_train = np.load(X_train_path)
y_train = np.load(y_train_path)
X_val = np.load(X_val_path)
y_val = np.load(y_val_path)
X_test = np.load(X_test_path)
y_test = np.load(y_test_path)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Last time step
        return out

# Model params
input_size = 2048  # Xception features
hidden_size = 128
num_layers = 2
num_classes = 2  # Real (0) vs Fake (1)

model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
if torch.cuda.is_available():
    model = model.cuda()
    print("Using GPU")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_correct = 0
    train_total = 0
    for inputs, labels in train_loader:
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_acc = 100 * train_correct / train_total
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = 100 * val_correct / val_total
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

# Get predictions
def get_predictions(model, loader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
    return np.array(all_preds)

train_preds = get_predictions(model, train_loader)
val_preds = get_predictions(model, val_loader)
test_preds = get_predictions(model, test_loader)

# Save predictions
np.save(os.path.join(base_path, "train_preds.npy"), train_preds)
np.save(os.path.join(base_path, "val_preds.npy"), val_preds)
np.save(os.path.join(base_path, "test_preds.npy"), test_preds)

# Test accuracy
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
test_acc = 100 * test_correct / test_total
print(f"Test Accuracy: {test_acc:.2f}%")