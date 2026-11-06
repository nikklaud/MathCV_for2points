import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
IMG = BASE / "images" / "source"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

def spectrum(image):
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image))))

for filename in ["fine_stripes.png", "brick.png"]:
    image = cv2.imread(str(IMG / filename), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
    high_frequency = image - blurred
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, data, title in zip(axes.ravel(), [image, spectrum(image), blurred, spectrum(blurred), high_frequency + 128, spectrum(high_frequency)], ["Original", "Original spectrum", "Gaussian σ=2", "Blurred spectrum", "High-frequency residual +128", "Residual spectrum"]): ax.imshow(data, cmap="gray"); ax.set_title(title); ax.axis("off")
    plt.tight_layout(); plt.savefig(OUT / f"06_fft_{Path(filename).stem}.png", dpi=150); plt.show()
    if filename == "fine_stripes.png":
        simple = image[::4, ::4]
        area = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4), interpolation=cv2.INTER_AREA)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4)); axes[0].imshow(simple, cmap="gray"); axes[0].set_title("Stride slicing /4"); axes[1].imshow(area, cmap="gray"); axes[1].set_title("INTER_AREA /4")
        plt.tight_layout(); plt.savefig(OUT / "06_downsample_stripes.png", dpi=150); plt.show()

checker = cv2.imread(str(IMG / "checkerboard.png"), cv2.IMREAD_GRAYSCALE)
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
for ax, scale in zip(axes, [2, 4, 8]): ax.imshow(checker[::scale, ::scale], cmap="gray"); ax.set_title(f"Checkerboard stride {scale}")
plt.tight_layout(); plt.savefig(OUT / "06_checkerboard_aliasing.png", dpi=150); plt.show()
