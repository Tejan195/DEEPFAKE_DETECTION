import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score

# Paths
base_path = r"C:/Users/tejan/OneDrive/Desktop/Deefake_Detection_Model"
X_train_path = os.path.join(base_path, "X_train.npy")
y_train_path = os.path.join(base_path, "y_train.npy")
X_val_path = os.path.join(base_path, "X_val.npy")
y_val_path = os.path.join(base_path, "y_val.npy")
X_test_path = os.path.join(base_path, "X_test.npy")
y_test_path = os.path.join(base_path, "y_test.npy")

# Load data
X_train = np.load(X_train_path)  # (9585, 10, 2048)
y_train = np.load(y_train_path)  # (9585,)
X_val = np.load(X_val_path)      # (2054, 10, 2048)
y_val = np.load(y_val_path)      # (2054,)
X_test = np.load(X_test_path)    # (2054, 10, 2048)
y_test = np.load(y_test_path)    # (2054,)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model
class LSTMGNNClassifier(nn.Module):
    def __init__(self, input_size=2048, hidden_size=256, num_layers=3, gnn_hidden=128, num_classes=2):
        super(LSTMGNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.gnn_conv1 = GCNConv(hidden_size, gnn_hidden)
        self.gnn_conv2 = GCNConv(gnn_hidden, gnn_hidden)
        self.fc = nn.Linear(gnn_hidden, num_classes)

    def forward(self, x, edge_index):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))  # (batch, seq_len, hidden_size)
        batch_size, seq_len, _ = lstm_out.size()
        gnn_in = lstm_out.contiguous().view(-1, self.hidden_size)  # (batch * seq_len, hidden_size)
        gnn_out = torch.relu(self.gnn_conv1(gnn_in, edge_index))
        gnn_out = self.dropout(gnn_out)
        gnn_out = self.gnn_conv2(gnn_out, edge_index)  # (batch * seq_len, gnn_hidden)
        gnn_out = gnn_out.view(batch_size, seq_len, -1).mean(dim=1)  # (batch, gnn_hidden)
        return self.fc(gnn_out)  # (batch, num_classes)

# Fixed edge index function
def create_edge_index(seq_len, batch_size):
    edges = [[i, i+1] for i in range(seq_len-1)]  # [0-1, 1-2, ..., 8-9]
    edge_index = torch.tensor(edges, dtype=torch.long).T  # (2, 9)
    offset = torch.arange(batch_size, dtype=torch.long) * seq_len  # (batch_size,)
    
    if torch.cuda.is_available():
        edge_index = edge_index.cuda()
        offset = offset.cuda()
    
    edge_index = edge_index.repeat(1, batch_size)  # (2, 9 * batch_size)
    offset = offset.repeat_interleave(seq_len - 1).repeat(2, 1)  # (2, 9 * batch_size)
    edge_index = edge_index + offset  # (2, 9 * batch_size)
    
    return edge_index

# Model setup
model = LSTMGNNClassifier()
if torch.cuda.is_available():
    model = model.cuda()
    print("Using GPU")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training loop
num_epochs = 17
patience = 5
best_val_acc = 0
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    train_correct = 0
    train_total = 0
    for inputs, labels in train_loader:
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        edge_index = create_edge_index(seq_len=10, batch_size=inputs.size(0))
        optimizer.zero_grad()
        outputs = model(inputs, edge_index)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    train_acc = 100 * train_correct / train_total
    
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            edge_index = create_edge_index(seq_len=10, batch_size=inputs.size(0))
            outputs = model(inputs, edge_index)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = 100 * val_correct / val_total
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(base_path, "best_model_gnn.pt"))
        print(f"Saved best model with Val Acc: {best_val_acc:.2f}%")
    else:
        patience_counter += 1
        print(f"Patience counter: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("Early stopping.")
            break

# Test
model.load_state_dict(torch.load(os.path.join(base_path, "best_model_gnn.pt")))
model.eval()
test_correct = 0
test_total = 0
test_preds, test_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        edge_index = create_edge_index(seq_len=10, batch_size=inputs.size(0))
        outputs = model(inputs, edge_index)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
test_acc = 100 * test_correct / test_total
test_auc = roc_auc_score(test_labels, test_preds)
print(f"Test Accuracy: {test_acc:.2f}%, Test AUC: {test_auc:.4f}")

# Save predictions
np.save(os.path.join(base_path, "test_preds.npy"), np.array(test_preds))