import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "images" / "segmentation"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
image_transform = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor()])
mask_transform = transforms.Compose([transforms.Resize((96, 96), interpolation=transforms.InterpolationMode.NEAREST), transforms.ToTensor()])

class SegmentationDataset(Dataset):
    def __init__(self, split):
        self.image_dir = DATA / split / "images"
        self.mask_dir = DATA / split / "masks"
        self.names = sorted(p.name for p in self.image_dir.glob("*.png"))
    def __len__(self): return len(self.names)
    def __getitem__(self, index):
        name = self.names[index]
        image = Image.open(self.image_dir / name).convert("RGB")
        mask = Image.open(self.mask_dir / name).convert("L")
        return image_transform(image), (mask_transform(mask) > 0.5).float()

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU())
    def forward(self, x): return self.layers(x)

class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(3, 16); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(16, 32); self.pool2 = nn.MaxPool2d(2)
        self.bridge = DoubleConv(32, 64)
        self.dec2 = DoubleConv(96, 32); self.dec1 = DoubleConv(48, 16)
        self.out = nn.Conv2d(16, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool1(e1)); b = self.bridge(self.pool2(e2))
        u2 = nn.functional.interpolate(b, size=e2.shape[2:], mode="bilinear", align_corners=False); d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = nn.functional.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False); d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.out(d1)

def metrics(pred, target):
    pred = pred.bool(); target = target.bool()
    intersection = (pred & target).sum().item()
    union = (pred | target).sum().item()
    total = pred.sum().item() + target.sum().item()
    return intersection / union if union else 1.0, 2 * intersection / total if total else 1.0

train_dataset = SegmentationDataset("train")
val_dataset = SegmentationDataset("val")
print("Train pairs:", len(train_dataset), "Validation pairs:", len(val_dataset))
first_image, first_mask = train_dataset[0]
print("Image shape:", first_image.shape, "Mask shape:", first_mask.shape, "Mask values:", torch.unique(first_mask))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetSmall().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
criterion = nn.BCEWithLogitsLoss()
for epoch in range(5):
    model.train(); loss_sum = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(); loss = criterion(model(images), masks); loss.backward(); optimizer.step(); loss_sum += loss.item() * images.size(0)
    print(f"Epoch {epoch+1}/5: loss={loss_sum / len(train_dataset):.4f}")

model.eval(); intersection = union = predicted_sum = target_sum = 0
with torch.no_grad():
    for images, masks in val_loader:
        logits = model(images.to(device)).cpu(); predictions = torch.sigmoid(logits) >= 0.5
        intersection += (predictions & masks.bool()).sum().item(); union += (predictions | masks.bool()).sum().item(); predicted_sum += predictions.sum().item(); target_sum += masks.sum().item()
iou = intersection / union if union else 1.0
dice = 2 * intersection / (predicted_sum + target_sum) if predicted_sum + target_sum else 1.0
print(f"Validation IoU: {iou:.4f}, Dice: {dice:.4f}")

image, mask = val_dataset[0]
with torch.no_grad(): probabilities = torch.sigmoid(model(image.unsqueeze(0).to(device))).cpu()[0, 0].numpy()
fig, axes = plt.subplots(1, 4, figsize=(12, 4))
axes[0].imshow(image.permute(1, 2, 0)); axes[0].set_title("Image")
axes[1].imshow(mask[0], cmap="gray"); axes[1].set_title("Ground truth")
axes[2].imshow(probabilities, cmap="viridis", vmin=0, vmax=1); axes[2].set_title("Probability")
axes[3].imshow(probabilities >= 0.5, cmap="gray"); axes[3].set_title("Prediction 0.5")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.savefig(OUT / "10_segmentation_prediction.png", dpi=150); plt.show()
for threshold in [0.3, 0.5, 0.7]:
    pred = torch.from_numpy(probabilities >= threshold); target = mask[0].bool()
    print(f"Threshold {threshold}: IoU={metrics(pred, target)[0]:.4f}, Dice={metrics(pred, target)[1]:.4f}")
