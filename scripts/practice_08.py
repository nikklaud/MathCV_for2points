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
checker = cv2.imread(str(IMG / "checkerboard.png"), cv2.IMREAD_GRAYSCALE)
image = torch.from_numpy(checker.astype(np.float32) / 255).unsqueeze(0).unsqueeze(0)
max_pool = nn.MaxPool2d(2)(image)
avg_pool = nn.AvgPool2d(2)(image)
mean_kernel = torch.ones(1, 1, 3, 3) / 9
conv_stride1 = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
conv_stride2 = nn.Conv2d(1, 1, 3, stride=2, padding=1, bias=False)
with torch.no_grad(): conv_stride1.weight.copy_(mean_kernel); conv_stride2.weight.copy_(mean_kernel)
result_stride1 = conv_stride1(image); result_stride2 = conv_stride2(image)
conv_d1 = nn.Conv2d(1, 1, 3, padding=1, dilation=1, bias=False)
conv_d2 = nn.Conv2d(1, 1, 3, padding=2, dilation=2, bias=False)
with torch.no_grad(): conv_d1.weight.copy_(mean_kernel); conv_d2.weight.copy_(mean_kernel)
print("Dilation output shapes:", conv_d1(image).shape, conv_d2(image).shape)
height = int(np.floor((64 + 2 * 2 - 2 * (3 - 1) - 1) / 2 + 1))
width = height
check = nn.Conv2d(1, 1, 3, stride=2, padding=2, dilation=2)
print("Calculated output:", height, width, "Program output:", check(torch.zeros(1, 1, 64, 64)).shape)

first = nn.Conv2d(1, 1, 3, padding=1, bias=False)
second = nn.Conv2d(1, 1, 3, padding=1, bias=False)
with torch.no_grad(): first.weight.fill_(1); second.weight.fill_(1)
probe = torch.zeros(1, 1, 9, 9, requires_grad=True)
output = second(first(probe))
output[0, 0, 4, 4].backward()
gradient = probe.grad[0, 0].numpy()
fig, axes = plt.subplots(1, 4, figsize=(12, 4))
for ax, data, title in zip(axes, [max_pool[0,0].numpy(), avg_pool[0,0].numpy(), result_stride2[0,0].detach().numpy(), gradient], ["MaxPool 2", "AvgPool 2", "Conv stride 2", "Receptive field gradient"]): ax.imshow(data, cmap="gray"); ax.set_title(title)
plt.tight_layout(); plt.savefig(OUT / "08_pooling_stride.png", dpi=150); plt.show()
