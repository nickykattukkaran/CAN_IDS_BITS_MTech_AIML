import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load and preprocess datasets
def load_and_preprocess_data(filenames, labels):
    data_frames = []
    for i, file in enumerate(filenames):
        df = pd.read_csv(file)
        df['Label'] = labels[i]
        data_frames.append(df)
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Convert binary strings to lists of integers
    combined_df['Payload'] = combined_df['Payload'].apply(lambda x: [int(b) for b in x])
    combined_df['TimeInterval'] = combined_df['TimeInterval'].apply(lambda x: [int(b) for b in x])
    
    return combined_df

# File paths and labels
filenames = [
    'Attack_free_dataset.csv',
    'DoS_attack_dataset.csv',
    'Fuzzy_attack_dataset.csv',
    'Impersonate_attack_dataset.csv'
]
labels = ['Attack-Free', 'DoS', 'Fuzzy', 'Impersonate']

# Load data
df = load_and_preprocess_data(filenames, labels)

# Prepare features and labels
X = np.array(df['Payload'].tolist())
y = df['Label']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)  # Add channel dimension
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create a custom dataset
class CANData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CANData(X_train, y_train)
test_dataset = CANData(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the deep 1D CNN model
class Deep1DCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Deep1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, activation='relu')
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, activation='relu')
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, activation='relu')
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(256 * ((input_size - 6) // 8), 128)  # Adjust for input size after convolution
        self.dropout4 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = nn.ReLU()(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

# Model parameters
input_size = X_train.shape[1]
num_classes = len(label_encoder.classes_)
model = Deep1DCNN(input_size=input_size, num_classes=num_classes)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))
    
    # Evaluate on the test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    test_losses.append(test_loss / len(test_loader))
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Test Accuracy: {100 * correct / total:.2f}%")

# Plot training and test losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.show()

# Save the model
torch.save(model.state_dict(), 'deep_1dcnn_can_intrusion_model.pth')
