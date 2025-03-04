import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel, ViTConfig
import cv2
import imgaug.augmenters as iaa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Custom Dataset Class
class DeepfakeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)['image']
        
        img = Image.fromarray(img)
        img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img, label

# Augmentations
def get_augmentations():
    return iaa.Sequential([
        iaa.Sometimes(0.5, iaa.Affine(rotate=(-10, 10), translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, scale=(0.9, 1.1))),
        iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.1, 0.1))),  # Random cut-out
        # Face cut-out requires facial landmark detection; simplified here as random crop
        iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.2)))
    ])

# Hybrid Transformer Model
class HybridTransformer(nn.Module):
    def __init__(self):
        super(HybridTransformer, self).__init__()
        # Feature Extractors
        self.xception = models.xception(pretrained=True)
        self.efficientnet = models.efficientnet_b4(pretrained=True)
        
        # Modify the classifiers to output features
        self.xception.fc = nn.Identity()
        self.efficientnet.classifier = nn.Identity()
        
        # Vision Transformer
        self.vit_config = ViTConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12)
        self.vit = ViTModel(self.vit_config)
        
        # Final classifier
        self.fc = nn.Linear(768, 2)  # Binary classification: real or fake
        
    def forward(self, x):
        # Extract features
        xception_features = self.xception(x)  # Shape: [batch, 2048]
        effnet_features = self.efficientnet(x)  # Shape: [batch, 1280]
        
        # Concatenate features
        fused_features = torch.cat((xception_features, effnet_features), dim=1)  # Shape: [batch, 3328]
        
        # Reshape for ViT (ViT expects [batch, seq_len, hidden_size])
        fused_features = fused_features.unsqueeze(1)  # Shape: [batch, 1, 3328]
        fused_features = nn.Linear(3328, 768)(fused_features)  # Project to ViT's hidden size
        
        # Add [CLS] token
        cls_token = torch.zeros(x.shape[0], 1, 768).to(x.device)
        features_with_cls = torch.cat([cls_token, fused_features], dim=1)  # Shape: [batch, 2, 768]
        
        # Add positional embeddings
        pos_embedding = nn.Parameter(torch.randn(1, 2, 768)).to(x.device)
        features_with_pos = features_with_cls + pos_embedding
        
        # Pass through ViT
        vit_output = self.vit(inputs_embeds=features_with_pos).last_hidden_state[:, 0, :]  # Take [CLS] token output
        
        # Final classification
        output = self.fc(vit_output)
        return output

# Training Loop
def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-3, momentum=0.9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
        
        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss/len(val_loader)}")
        
        if val_loss > prev_val_loss for 3 epochs:  # Early stopping simplified
            break

# Main Execution
if __name__ == "__main__":
    # Placeholder for dataset paths (replace with actual paths)
    train_paths = ["path/to/train/images"]  # List of image paths
    train_labels = [0, 1]  # 0 for real, 1 for fake
    val_paths = ["path/to/val/images"]
    val_labels = [0, 1]
    
    train_dataset = DeepfakeDataset(train_paths, train_labels, transform=get_augmentations())
    val_dataset = DeepfakeDataset(val_paths, val_labels, transform=None)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = HybridTransformer()
    train_model(model, train_loader, val_loader)
