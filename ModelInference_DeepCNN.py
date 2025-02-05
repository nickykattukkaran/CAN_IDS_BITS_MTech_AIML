import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 11 * 11, 256)
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

# Load the saved model
model = CNN()
model.load_state_dict(torch.load('model.pth', weights_only=True))
model.eval()

# Define transformation for the input image
transform = transforms.Compose([
    transforms.Resize((94, 94)),
    transforms.ToTensor()
])

# Function to classify a single binary image
def classify_image(image_path):
    image = Image.open(image_path).convert('L')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

# Classify the binary image in the test folder
image_path = 'test/image_0.jpg'  # Adjust the path as needed
attack_classes = ['Attack_free', 'Dos_Attack', 'Fuzzy_Attack', 'Impersonate_Attack']
predicted_class = classify_image(image_path)
print(f"The binary image Belongs to Attack Free and the model classified it as: {attack_classes[predicted_class]}")
