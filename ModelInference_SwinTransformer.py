import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time

# Define the Swin Transformer Model (Same architecture as training script)
class SwinTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embed_dim, num_heads, window_size, mlp_ratio=4.,
                 depths=[2, 2, 6, 2]):
        super(SwinTransformer, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        self.blocks = nn.ModuleList()
        for depth in depths:
            block = nn.ModuleList([
                SwinTransformerBlock(
                    dim=embed_dim, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio
                ) for _ in range(depth)
            ])
            self.blocks.append(block)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        for stage in self.blocks:
            for block in stage:
                x = block(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

# Define the Swin Transformer Block
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
        shortcut = x
        x = self.norm1(x)
        x = x.permute(1, 0, 2)
        x, _ = self.attn(x, x, x)
        x = x.permute(1, 0, 2)
        x = shortcut + self.drop_path(x)

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(x)
        return x

# Load the trained model
model = SwinTransformer(image_size=96, patch_size=4, num_classes=4, embed_dim=96, num_heads=8, window_size=7)
model.load_state_dict(torch.load('swin_transformer_model1.pth',weights_only=True, map_location=torch.device('cpu')))
model.eval()

# Define transformation for inference
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

# Define function for inference
def predict(image_path):
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    class_labels = ['Attack_Free', 'Dos_Attack', 'Fuzzy_Attack', 'Impersonate_Attack']
    return class_labels[predicted_class.item()]

# Example usage
start_time_infer = time.time()
image_path = 'test/Dos_Attack_6982.jpg'  # Replace with actual image path
prediction = predict(image_path)
print(f'Predicted class: {prediction}')
end_time_infer = time.time()
time_infer = (end_time_infer - start_time_infer)
print(f"Time required for inference (Swin Transformer Model): {time_infer:.3f} s")