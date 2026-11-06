import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
IMG = BASE / "images" / "source"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

camera = cv2.imread(str(IMG / "camera.png"), cv2.IMREAD_GRAYSCALE)
astronaut = cv2.imread(str(IMG / "astronaut.png"))
coffee = cv2.imread(str(IMG / "coffee.png"))

print("camera:", camera.dtype, camera.min(), camera.max(), camera.shape[1], camera.shape[0])
print("pixel (100, 70):", int(camera[70, 100]))

rgb = cv2.cvtColor(astronaut, cv2.COLOR_BGR2RGB)
channels = [rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]]
gray_cv = cv2.cvtColor(astronaut, cv2.COLOR_BGR2GRAY)
gray_manual = np.clip(0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2], 0, 255).astype(np.uint8)
print("Gray absolute difference:", np.abs(gray_cv.astype(np.int16) - gray_manual.astype(np.int16)).max(), np.mean(np.abs(gray_cv.astype(np.int16) - gray_manual.astype(np.int16))))
bright = np.clip(camera.astype(np.int16) + 35, 0, 255).astype(np.uint8)
coffee_gray = cv2.cvtColor(coffee, cv2.COLOR_BGR2GRAY)
print("camera mean/std:", camera.mean(), camera.std())
print("coffee mean/std:", coffee_gray.mean(), coffee_gray.std())

fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes[0, 0].imshow(rgb); axes[0, 0].set_title("Astronaut RGB")
for ax, channel, name in zip(axes[0, 1:], channels, ["R", "G", "B"]): ax.imshow(channel, cmap="gray"); ax.set_title(name)
axes[1, 0].imshow(gray_cv, cmap="gray"); axes[1, 0].set_title("Gray OpenCV")
axes[1, 1].imshow(gray_manual, cmap="gray"); axes[1, 1].set_title("Gray weighted")
axes[1, 2].hist(camera.ravel(), 256, range=(0, 256), alpha=.6, label="Original")
axes[1, 2].hist(bright.ravel(), 256, range=(0, 256), alpha=.6, label="Brightness +35"); axes[1, 2].legend(); axes[1, 2].set_title("Histograms")
axes[1, 3].imshow(coffee_gray, cmap="gray"); axes[1, 3].set_title("Coffee grayscale")
for ax in axes.ravel(): ax.axis("off")
plt.tight_layout(); plt.savefig(OUT / "01_channels.png", dpi=150); plt.show()
