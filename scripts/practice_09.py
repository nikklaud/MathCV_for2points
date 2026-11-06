import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "images" / "classification"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
full_train = ImageFolder(DATA / "train", transform=transform)
test_dataset = ImageFolder(DATA / "test", transform=transform)
print("Classes:", full_train.classes)
for label, name in enumerate(full_train.classes): print(name, "train", sum(y == label for _, y in full_train.samples), "test", sum(y == label for _, y in test_dataset.samples))
train_size = int(0.8 * len(full_train)); val_size = len(full_train) - train_size
train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

class ShapeCNN(nn.Module):
    def __init__(self, pooling=True):
        super().__init__()
        self.pooling = pooling
        self.features = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2) if pooling else nn.Identity(), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2) if pooling else nn.Identity())
        feature_size = 32 * (16 if pooling else 64) ** 2
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(feature_size, 128), nn.ReLU(), nn.Linear(128, 3))
    def forward(self, x): return self.classifier(self.features(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ShapeCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
train_losses, val_accuracies = [], []
for epoch in range(5):
    model.train(); running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(); loss = criterion(model(images), labels); loss.backward(); optimizer.step(); running_loss += loss.item() * labels.size(0)
    train_losses.append(running_loss / len(train_dataset))
    model.eval(); correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            predicted = model(images.to(device)).argmax(1).cpu(); correct += (predicted == labels).sum().item(); total += labels.size(0)
    val_accuracies.append(correct / total)
    print(f"Epoch {epoch+1}/5: loss={train_losses[-1]:.4f}, val_accuracy={val_accuracies[-1]:.4f}")

model.eval(); targets, predictions = [], []
with torch.no_grad():
    for images, labels in test_loader:
        predictions.extend(model(images.to(device)).argmax(1).cpu().tolist()); targets.extend(labels.tolist())
print("Test accuracy:", accuracy_score(targets, predictions))
print("Confusion matrix:\n", confusion_matrix(targets, predictions))
fig, axes = plt.subplots(1, 2, figsize=(10, 4)); axes[0].plot(train_losses); axes[0].set_title("Train loss"); axes[1].plot(val_accuracies); axes[1].set_title("Validation accuracy")
plt.tight_layout(); plt.savefig(OUT / "09_training_curves.png", dpi=150); plt.show()
cm = confusion_matrix(targets, predictions)
plt.figure(figsize=(5, 4)); plt.imshow(cm, cmap="Blues"); plt.xticks(range(3), full_train.classes); plt.yticks(range(3), full_train.classes); plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(OUT / "09_confusion_matrix.png", dpi=150); plt.show()
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for ax, index in zip(axes.ravel(), range(min(9, len(test_dataset)))):
    image, label = test_dataset[index]; ax.imshow(image.permute(1, 2, 0)); ax.set_title(f"True: {full_train.classes[label]} / Pred: {full_train.classes[predictions[index]]}"); ax.axis("off")
plt.tight_layout(); plt.savefig(OUT / "09_test_examples.png", dpi=150); plt.show()
