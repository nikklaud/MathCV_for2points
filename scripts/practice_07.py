import cv2
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
IMG = BASE / "images" / "source"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
coins = cv2.imread(str(IMG / "coins.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
input_tensor = torch.from_numpy(coins).unsqueeze(0).unsqueeze(0)
sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
with torch.no_grad(): conv.weight.copy_(torch.from_numpy(sobel_x).reshape(1, 1, 3, 3))
torch_result = conv(input_tensor).detach().squeeze().numpy()
cv_result = cv2.filter2D(coins, cv2.CV_32F, sobel_x, borderType=cv2.BORDER_CONSTANT)
print("PyTorch/OpenCV max difference:", np.max(np.abs(torch_result - cv_result)))

kernels = np.stack([sobel_x, sobel_x.T, np.eye(3, dtype=np.float32), np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)])
bank = nn.Conv2d(1, 4, 3, padding=1, bias=False)
with torch.no_grad(): bank.weight.copy_(torch.from_numpy(kernels[:, None]))
features = bank(input_tensor).detach().numpy()[0]
fig, axes = plt.subplots(1, 4, figsize=(12, 4))
for ax, feature, title in zip(axes, features, ["Sobel X", "Sobel Y", "Identity", "Laplace"]): ax.imshow(feature, cmap="gray"); ax.set_title(title)
plt.tight_layout(); plt.savefig(OUT / "07_filter_bank.png", dpi=150); plt.show()

layer = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=True)
print("RGB to 32 parameters:", sum(p.numel() for p in layer.parameters()))
print("Output shape:", layer(torch.zeros(1, 3, 128, 128)).shape)
conv_params = 32 * (3 * 3 * 3 + 1)
dense_params = (128 * 128 * 3) * (32 * 128 * 128) + (32 * 128 * 128)
print("Conv parameters:", conv_params, "Dense parameters:", dense_params)
