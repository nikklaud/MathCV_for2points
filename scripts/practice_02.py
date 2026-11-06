import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
IMG = BASE / "images" / "source"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

def correlate2d_manual(image, kernel):
    image = np.asarray(image, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    kh, kw = kernel.shape
    padded = np.pad(image, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="constant")
    result = np.zeros_like(image, dtype=np.float32)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            result[y, x] = np.sum(padded[y:y + kh, x:x + kw] * kernel)
    return result

small = np.array([[0,0,0,0,0],[0,1,2,3,0],[0,4,5,6,0],[0,7,8,9,0],[0,0,0,0,0]], dtype=np.float32)
kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
manual = correlate2d_manual(small, kernel)
cv_result = cv2.filter2D(small, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)
print("Manual result:\n", manual)
print("OpenCV max error:", np.max(np.abs(manual - cv_result)))

impulse = cv2.imread(str(IMG / "impulse.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
corr = cv2.filter2D(impulse, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)
conv = convolve2d(impulse, kernel, mode="same", boundary="fill", fillvalue=0)
camera = cv2.imread(str(IMG / "camera.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
avg = np.ones((3, 3), np.float32) / 9
borders = [cv2.filter2D(camera, cv2.CV_32F, avg, borderType=t) for t in [cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT, cv2.BORDER_REPLICATE]]
image_a = camera / 255
image_b = np.flip(camera, axis=1).copy() / 255
a, b = 1.7, -0.4
lhs = cv2.filter2D(a * image_a + b * image_b, cv2.CV_32F, avg)
rhs = a * cv2.filter2D(image_a, cv2.CV_32F, avg) + b * cv2.filter2D(image_b, cv2.CV_32F, avg)
print("Linearity error:", np.max(np.abs(lhs - rhs)))

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, data, title in zip(axes.ravel()[:3], [corr, conv, borders[0]], ["Correlation", "Convolution", "Constant border"]): ax.imshow(data, cmap="gray"); ax.set_title(title)
for ax, data, title in zip(axes.ravel()[3:], [borders[0][:30,:30], borders[1][:30,:30], borders[2][:30,:30]], ["Constant corner", "Reflect corner", "Replicate corner"]): ax.imshow(data, cmap="gray"); ax.set_title(title)
plt.tight_layout(); plt.savefig(OUT / "02_correlation_convolution.png", dpi=150); plt.show()
