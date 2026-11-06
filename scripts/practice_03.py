import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
IMG = BASE / "images"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
original = cv2.imread(str(IMG / "source" / "camera.png"), cv2.IMREAD_GRAYSCALE)
noisy_inputs = {"Gaussian noise": cv2.imread(str(IMG / "noisy" / "camera_gaussian_sigma22.png"), 0), "Salt pepper": cv2.imread(str(IMG / "noisy" / "camera_salt_pepper_7pct.png"), 0)}

def metrics(result):
    mse = np.mean((original.astype(np.float32) - result.astype(np.float32)) ** 2)
    psnr = float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
    return mse, psnr

rows = []
results = {}
for noise_name, noisy in noisy_inputs.items():
    results[noise_name] = {"Input": noisy, "Box 5x5": cv2.blur(noisy, (5, 5)), "Median 5x5": cv2.medianBlur(noisy, 5)}
    for sigma in [0.8, 1.5, 3.0]: results[noise_name][f"Gaussian {sigma}"] = cv2.GaussianBlur(noisy, (0, 0), sigma)
    for filter_name, filtered in results[noise_name].items():
        mse, psnr = metrics(filtered)
        rows.append({"Noise": noise_name, "Filter": filter_name, "MSE": mse, "PSNR": psnr})

report = pd.DataFrame(rows).sort_values("PSNR", ascending=False)
print(report.to_string(index=False))
report.to_csv(OUT / "03_metrics.csv", index=False)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes[0, 0].imshow(original, cmap="gray"); axes[0, 0].set_title("Original")
for ax, (name, img) in zip(axes[0, 1:], list(results["Gaussian noise"].items())[:3]): ax.imshow(img, cmap="gray"); ax.set_title(name)
axes[1, 0].imshow(noisy_inputs["Salt pepper"], cmap="gray"); axes[1, 0].set_title("Salt pepper input")
for ax, (name, img) in zip(axes[1, 1:], list(results["Salt pepper"].items())[1:4]): ax.imshow(img, cmap="gray"); ax.set_title(name)
plt.tight_layout(); plt.savefig(OUT / "03_noise_filters.png", dpi=150); plt.show()

sigmas = np.arange(0.5, 3.01, 0.5)
psnr_values = [metrics(cv2.GaussianBlur(noisy_inputs["Gaussian noise"], (0, 0), float(s)))[1] for s in sigmas]
plt.figure(figsize=(7, 4)); plt.plot(sigmas, psnr_values, marker="o"); plt.xlabel("Gaussian sigma"); plt.ylabel("PSNR (dB)"); plt.grid(); plt.tight_layout(); plt.savefig(OUT / "03_psnr_sigma.png", dpi=150); plt.show()
