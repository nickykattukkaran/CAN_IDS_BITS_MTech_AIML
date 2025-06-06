import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time

# Define the Swin Transformer Block
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4., dropout=0., attn_dropout=0., drop_path=0.):
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
    def __init__(self, image_size, patch_size, embed_dim, num_heads, window_size, mlp_ratio=4., depths=[2, 2, 6, 2]):
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

        self.norm = nn.LayerNorm(embed_dim)

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

        x = self.norm(x)
        return x
    

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        #self.fc1 = nn.Linear(128 * 11 * 11, 256)
        self.fc1 = nn.Linear(128 * 12 * 12, 256)  # Adjust based on actual size

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        #x = x.view(-1, 128 * 11 * 11)  # Flatten the tensor
        #print(f"Shape before flattening: {x.shape}")  # Debugging line
        x = x.view(-1, 128 * 12 * 12)  # Automatically determine flatten size
        x = torch.relu(self.fc1(x))
        return x

# Define the hybrid model
class HybridModel(nn.Module):
    def __init__(self, swin_transformer, cnn, num_classes):
        super(HybridModel, self).__init__()
        self.swin_transformer = swin_transformer
        self.cnn = cnn
        self.mlp = nn.Sequential(
            nn.Linear(256 + 96, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        swin_features = self.swin_transformer(x)
        swin_features = swin_features.mean(dim=1)  # Global average pooling
        cnn_features = self.cnn(x)
        features = torch.cat((swin_features, cnn_features), dim=1)  # Concatenate features
        x = self.mlp(features)
        return x


# Define transformations
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

# Load the trained model
swin_transformer = SwinTransformer(
    image_size=96, patch_size=4, embed_dim=96, num_heads=8, window_size=7
)
cnn = CNN()
num_classes = 4
model = HybridModel(swin_transformer, cnn, num_classes)
model.load_state_dict(torch.load('hybrid_model1.pth', weights_only=True))
model.eval()

# Class labels
class_labels = ['Attack_free', 'Dos_Attack', 'Fuzzy_Attack', 'Impersonate_Attack']

def predict(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return class_labels[predicted.item()]

# Example usage
start_time_infer = time.time()
image_path = 'test/Dos_Attack_6982.jpg'  # Update with the actual image path
prediction = predict(image_path)
print(f'Predicted class: {prediction}')
end_time_infer = time.time()
time_infer = (end_time_infer - start_time_infer)
print(f"Time required for inference (Hybrid Model): {time_infer:.3f} s")
