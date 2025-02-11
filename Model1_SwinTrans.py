'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
 
# Define Swin Transformer Block
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4., dropout=0., attn_dropout=0.,
                 drop_path=0.):
        super(SwinTransformerBlock, self).__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_dropout)
        self.drop_path = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
 
    def forward(self, x):
        B, N, C = x.shape
 
        # Multi-Head Self-Attention
        shortcut = x
        x = self.norm1(x)
        x = x.permute(1, 0, 2)  # Required for MultiheadAttention
        x, _ = self.attn(x, x, x)
        x = x.permute(1, 0, 2)  # Revert to original shape
        x = shortcut + self.drop_path(x)
 
        # Feedforward Neural Network
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(x)
        return x
 
# Define the Swin Transformer
class SwinTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embed_dim, num_heads, window_size, mlp_ratio=4.,
                 depths=[2, 2, 6, 2]):
        super(SwinTransformer, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
 
        # Patch embedding
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
 
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
 
        # Stages of Swin Transformer blocks
        self.blocks = nn.ModuleList()
        for depth in depths:
            block = nn.ModuleList([
                SwinTransformerBlock(
                    dim=embed_dim, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio
                ) for _ in range(depth)
            ])
            self.blocks.append(block)
 
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
 
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # Shape: (B, C, H/P, W/P)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # Shape: (B, N, C), where N = H*W
 
        # Add positional encoding
        x = x + self.pos_embed
 
        # Pass through Swin Transformer blocks
        for stage in self.blocks:
            for block in stage:
                x = block(x)
 
        # Classification head
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)
        return x
 
# Custom dataset class (same as before)
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
 
    def __len__(self):
        return len(self.image_paths)
 
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
 
# Function to load images and labels from a folder (same as before)
def load_images_from_folder(folder_path, label):
    image_paths = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            image_paths.append(os.path.join(folder_path, filename))
            labels.append(label)
    return image_paths, labels
 
# Data preparation (same as before)
train_image_paths, train_labels = [], []
attack_free_train_images, attack_free_train_labels = load_images_from_folder('Genimage/Attack_free/train', 0)
dos_attack_train_images, dos_attack_train_labels = load_images_from_folder('Genimage/Dos_Attack/train', 1)
fuzzy_attack_train_images, fuzzy_attack_train_labels = load_images_from_folder('Genimage/Fuzzy_Attack/train', 2)
impersonate_attack_train_images, impersonate_attack_train_labels = load_images_from_folder('Genimage/Impersonate_Attack/train', 3)
 
train_image_paths.extend(attack_free_train_images[:5037])
train_labels.extend(attack_free_train_labels[:5037])
train_image_paths.extend(dos_attack_train_images[:5037])
train_labels.extend(dos_attack_train_labels[:5037])
train_image_paths.extend(fuzzy_attack_train_images[:5037])
train_labels.extend(fuzzy_attack_train_labels[:5037])
train_image_paths.extend(impersonate_attack_train_images[:5037])
train_labels.extend(impersonate_attack_train_labels[:5037])
 
test_image_paths, test_labels = [], []
attack_free_test_images, attack_free_test_labels = load_images_from_folder('Genimage/Attack_free/test', 0)
dos_attack_test_images, dos_attack_test_labels = load_images_from_folder('Genimage/Dos_Attack/test', 1)
fuzzy_attack_test_images, fuzzy_attack_test_labels = load_images_from_folder('Genimage/Fuzzy_Attack/test', 2)
impersonate_attack_test_images, impersonate_attack_test_labels = load_images_from_folder('Genimage/Impersonate_Attack/test', 3)
 
test_image_paths.extend(attack_free_test_images[:1260])
test_labels.extend(attack_free_test_labels[:1260])
test_image_paths.extend(dos_attack_test_images[:1260])
test_labels.extend(dos_attack_test_labels[:1260])
test_image_paths.extend(fuzzy_attack_test_images[:1260])
test_labels.extend(fuzzy_attack_test_labels[:1260])
test_image_paths.extend(impersonate_attack_test_images[:1260])
test_labels.extend(impersonate_attack_test_labels[:1260])
 
# Define transformations
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])
 
# Create datasets and dataloaders
train_dataset = CustomDataset(train_image_paths, train_labels, transform=transform)
test_dataset = CustomDataset(test_image_paths, test_labels, transform=transform)
 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
 
# Initialize the Swin Transformer model
model = SwinTransformer(
    image_size=96, patch_size=4, num_classes=4, embed_dim=96, num_heads=8, window_size=7
)
 
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# Training loop
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
 
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
 
# Save the model
torch.save(model.state_dict(), 'swin_transformer_model.pth')
 
# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
 
print(f"Accuracy: {100 * correct / total}%")
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 
# Define Swin Transformer Block
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4., dropout=0., attn_dropout=0.,
                 drop_path=0.):
        super(SwinTransformerBlock, self).__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_dropout)
        self.drop_path = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
 
    def forward(self, x):
        B, N, C = x.shape
 
        # Multi-Head Self-Attention
        shortcut = x
        x = self.norm1(x)
        x = x.permute(1, 0, 2)  # Required for MultiheadAttention
        x, _ = self.attn(x, x, x)
        x = x.permute(1, 0, 2)  # Revert to original shape
        x = shortcut + self.drop_path(x)
 
        # Feedforward Neural Network
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(x)
        return x
 
# Define the Swin Transformer
class SwinTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embed_dim, num_heads, window_size, mlp_ratio=4.,
                 depths=[2, 2, 6, 2]):
        super(SwinTransformer, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
 
        # Patch embedding
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
 
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
 
        # Stages of Swin Transformer blocks
        self.blocks = nn.ModuleList()
        for depth in depths:
            block = nn.ModuleList([
                SwinTransformerBlock(
                    dim=embed_dim, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio
                ) for _ in range(depth)
            ])
            self.blocks.append(block)
 
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
 
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # Shape: (B, C, H/P, W/P)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # Shape: (B, N, C), where N = H*W
 
        # Add positional encoding
        x = x + self.pos_embed
 
        # Pass through Swin Transformer blocks
        for stage in self.blocks:
            for block in stage:
                x = block(x)
 
        # Classification head
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)
        return x
 
# Custom dataset class (same as before)
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
 
    def __len__(self):
        return len(self.image_paths)
 
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
 
# Function to load images and labels from a folder (same as before)
def load_images_from_folder(folder_path, label):
    image_paths = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            image_paths.append(os.path.join(folder_path, filename))
            labels.append(label)
    return image_paths, labels
 
# Data preparation (same as before)
train_image_paths, train_labels = [], []
attack_free_train_images, attack_free_train_labels = load_images_from_folder('Genimage/Attack_free/train', 0)
dos_attack_train_images, dos_attack_train_labels = load_images_from_folder('Genimage/Dos_Attack/train', 1)
fuzzy_attack_train_images, fuzzy_attack_train_labels = load_images_from_folder('Genimage/Fuzzy_Attack/train', 2)
impersonate_attack_train_images, impersonate_attack_train_labels = load_images_from_folder('Genimage/Impersonate_Attack/train', 3)
 
train_image_paths.extend(attack_free_train_images[:5037])
train_labels.extend(attack_free_train_labels[:5037])
train_image_paths.extend(dos_attack_train_images[:5037])
train_labels.extend(dos_attack_train_labels[:5037])
train_image_paths.extend(fuzzy_attack_train_images[:5037])
train_labels.extend(fuzzy_attack_train_labels[:5037])
train_image_paths.extend(impersonate_attack_train_images[:5037])
train_labels.extend(impersonate_attack_train_labels[:5037])
 
test_image_paths, test_labels = [], []
attack_free_test_images, attack_free_test_labels = load_images_from_folder('Genimage/Attack_free/test', 0)
dos_attack_test_images, dos_attack_test_labels = load_images_from_folder('Genimage/Dos_Attack/test', 1)
fuzzy_attack_test_images, fuzzy_attack_test_labels = load_images_from_folder('Genimage/Fuzzy_Attack/test', 2)
impersonate_attack_test_images, impersonate_attack_test_labels = load_images_from_folder('Genimage/Impersonate_Attack/test', 3)
 
test_image_paths.extend(attack_free_test_images[:1260])
test_labels.extend(attack_free_test_labels[:1260])
test_image_paths.extend(dos_attack_test_images[:1260])
test_labels.extend(dos_attack_test_labels[:1260])
test_image_paths.extend(fuzzy_attack_test_images[:1260])
test_labels.extend(fuzzy_attack_test_labels[:1260])
test_image_paths.extend(impersonate_attack_test_images[:1260])
test_labels.extend(impersonate_attack_test_labels[:1260])
 
# Define transformations
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])
 
# Create datasets and dataloaders
train_dataset = CustomDataset(train_image_paths, train_labels, transform=transform)
test_dataset = CustomDataset(test_image_paths, test_labels, transform=transform)
 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
 
# Initialize the Swin Transformer model
model = SwinTransformer(
    image_size=96, patch_size=4, num_classes=4, embed_dim=96, num_heads=8, window_size=7
)
 
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
 
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
 
# Save the model
torch.save(model.state_dict(), 'swin_transformer_model.pth')
 
# Testing loop with Classification Report and Confusion Matrix
model.eval()
all_preds = []
all_labels = []
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
 
accuracy = 100 * correct / total
print(f"Accuracy: {accuracy}%")
 
# Generate Classification Report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['Attack_Free', 'Dos_Attack', 'Fuzzy_Attack', 'Impersonate_Attack']))
 
# Generate Confusion Matrix
print("Confusion Matrix:")
conf_matrix = confusion_matrix(all_labels, all_preds)
print(conf_matrix)
 
# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Attack_Free', 'Dos_Attack', 'Fuzzy_Attack', 'Impersonate_Attack'],
            yticklabels=['Attack_Free', 'Dos_Attack', 'Fuzzy_Attack', 'Impersonate_Attack'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()