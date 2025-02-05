import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import random
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 11 * 11, 256)  # Adjusted input size
        self.fc2 = nn.Linear(256, 4)  # 4 classes: Attack_free, Dos_Attack, Fuzzy_Attack, Impersonate_Attack

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 128 * 11 * 11)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Function to load images and labels from a folder
def load_images_from_folder(folder_path, label):
    image_paths = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            image_paths.append(os.path.join(folder_path, filename))
            labels.append(label)
    return image_paths, labels

# Data augmentation function
def augment_data(image_paths, labels, target_count):
    augmented_images = []
    augmented_labels = []
    transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(94, scale=(0.8, 1.0)),
        #transforms.ToTensor()
    ])
    
    while len(augmented_images) < target_count:
        idx = random.randint(0, len(image_paths) - 1)
        image_path = image_paths[idx]
        label = labels[idx]
        image = Image.open(image_path).convert('L')
        image = transform(image)
        augmented_images.append(image)
        augmented_labels.append(label)
    
    return augmented_images, augmented_labels
    
# Data without augmentation function
def augment_No_data(image_paths, labels, target_count):
    augmented_images = []
    augmented_labels = []
    # Define transformations
    # transform = transforms.Compose([
    #     transforms.Resize((94, 94)),
    #     transforms.ToTensor()
    # ])
    
    for i in range(target_count):
        image_path = image_paths[i]
        label = labels[i]
        image = Image.open(image_path).convert('L')
        #image = transform(image)
        augmented_images.append(image)
        augmented_labels.append(label)
    
    return augmented_images, augmented_labels
    	
# Load training data
train_images = []
train_labels = []

# Load entire training datasets for attacks and augment data
attack_free_train_images, attack_free_train_labels = load_images_from_folder('Genimage/Attack_free/train', 0)
dos_attack_train_images, dos_attack_train_labels = load_images_from_folder('Genimage/Dos_Attack/train', 1)
fuzzy_attack_train_images, fuzzy_attack_train_labels = load_images_from_folder('Genimage/Fuzzy_Attack/train', 2)
impersonate_attack_train_images, impersonate_attack_train_labels = load_images_from_folder('Genimage/Impersonate_Attack/train', 3)

# Perform data augmentation
attack_free_train_images, attack_free_train_labels = augment_data(attack_free_train_images, attack_free_train_labels, 40000)
dos_attack_train_images, dos_attack_train_labels = augment_data(dos_attack_train_images, dos_attack_train_labels, 40000)
fuzzy_attack_train_images, fuzzy_attack_train_labels = augment_data(fuzzy_attack_train_images, fuzzy_attack_train_labels, 40000)
impersonate_attack_train_images, impersonate_attack_train_labels = augment_data(impersonate_attack_train_images, impersonate_attack_train_labels, 40000)

train_images.extend(attack_free_train_images)
train_labels.extend(attack_free_train_labels)
train_images.extend(dos_attack_train_images)
train_labels.extend(dos_attack_train_labels)
train_images.extend(fuzzy_attack_train_images)
train_labels.extend(fuzzy_attack_train_labels)
train_images.extend(impersonate_attack_train_images)
train_labels.extend(impersonate_attack_train_labels)

# Load testing data
test_images = []
test_labels = []

# Load entire testing datasets for attacks and augment data
attack_free_test_images, attack_free_test_labels = load_images_from_folder('Genimage/Attack_free/test', 0)
dos_attack_test_images, dos_attack_test_labels = load_images_from_folder('Genimage/Dos_Attack/test', 1)
fuzzy_attack_test_images, fuzzy_attack_test_labels = load_images_from_folder('Genimage/Fuzzy_Attack/test', 2)
impersonate_attack_test_images, impersonate_attack_test_labels = load_images_from_folder('Genimage/Impersonate_Attack/test', 3)

# Perform data augmentation
attack_free_test_images, attack_free_test_labels = augment_data(attack_free_test_images, attack_free_test_labels, 15000)
dos_attack_test_images, dos_attack_test_labels = augment_data(dos_attack_test_images, dos_attack_test_labels, 15000)
fuzzy_attack_test_images, fuzzy_attack_test_labels = augment_data(fuzzy_attack_test_images, fuzzy_attack_test_labels, 15000)
impersonate_attack_test_images, impersonate_attack_test_labels = augment_data(impersonate_attack_test_images, impersonate_attack_test_labels, 15000)

test_images.extend(attack_free_test_images)
test_labels.extend(attack_free_test_labels)
test_images.extend(dos_attack_test_images)
test_labels.extend(dos_attack_test_labels)
test_images.extend(fuzzy_attack_test_images)
test_labels.extend(fuzzy_attack_test_labels)
test_images.extend(impersonate_attack_test_images)
test_labels.extend(impersonate_attack_test_labels)

# Define transformations
main_transform = transforms.Compose([
    transforms.Resize((94, 94)),
    transforms.ToTensor()
])


# Create datasets and dataloaders
train_dataset = CustomDataset(train_images, train_labels, transform=main_transform)
test_dataset = CustomDataset(test_images, test_labels, transform=main_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model
model = CNN()

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
torch.save(model.state_dict(), 'model5_DeepCNN.pth')

# Testing loop
model.eval()
correct = 0
total = 0
all_labels = []
all_predictions = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

print(f"Accuracy: {100 * correct / total}%")

# Classification report
print("Classification Report:")
print(classification_report(all_labels, all_predictions, target_names=['Attack_free', 'Dos_Attack', 'Fuzzy_Attack', 'Impersonate_Attack']))

# Confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Attack_free', 'Dos_Attack', 'Fuzzy_Attack', 'Impersonate_Attack'], yticklabels=['Attack_free', 'Dos_Attack', 'Fuzzy_Attack', 'Impersonate_Attack'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
