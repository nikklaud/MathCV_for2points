import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
IMG = BASE / "images" / "source"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

def edge_maps(name):
    image = cv2.imread(str(IMG / name), cv2.IMREAD_GRAYSCALE)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    angle = np.mod(np.degrees(np.arctan2(gy, gx)), 180)
    thresholds = np.quantile(magnitude, [0.70, 0.85, 0.95])
    masks = [magnitude >= t for t in thresholds]
    laplacian = cv2.Laplacian(image, cv2.CV_32F)
    smoothed = [cv2.Sobel(cv2.GaussianBlur(image, (0, 0), sigma), cv2.CV_32F, 1, 0) for sigma in [1.0, 2.0]]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    data = [gx, gy, magnitude, angle, *masks, laplacian]
    titles = ["Gx signed", "Gy signed", "Magnitude", "Orientation 0-180", "q=0.70", "q=0.85", "q=0.95", "Laplacian"]
    for ax, value, title in zip(axes.ravel(), data, titles): ax.imshow(value, cmap="gray"); ax.set_title(title); ax.axis("off")
    plt.tight_layout(); plt.savefig(OUT / f"04_{Path(name).stem}.png", dpi=150); plt.show()
    plt.figure(figsize=(10, 4))
    for i, value in enumerate(smoothed): plt.subplot(1, 2, i + 1); plt.imshow(value, cmap="gray"); plt.title(f"Sobel after Gaussian σ={i+1}")
    plt.tight_layout(); plt.savefig(OUT / f"04_smoothing_{Path(name).stem}.png", dpi=150); plt.show()

for filename in ["coins.png", "brick.png", "page.png"]: edge_maps(filename)
