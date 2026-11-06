import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
IMG = BASE / "images" / "templates"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
scene = cv2.imread(str(IMG / "search_scene.png"), cv2.IMREAD_GRAYSCALE)
template = cv2.imread(str(IMG / "template.png"), cv2.IMREAD_GRAYSCALE)
if scene.shape[0] < template.shape[0] or scene.shape[1] < template.shape[1]: raise ValueError("Scene must be larger than template")

def locate(image, templ, method=cv2.TM_CCOEFF_NORMED):
    response = cv2.matchTemplate(image, templ, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(response)
    if method == cv2.TM_SQDIFF_NORMED: return min_loc, min_val, response
    return max_loc, max_val, response

loc, score, response = locate(scene, template)
h, w = template.shape
marked = cv2.cvtColor(scene, cv2.COLOR_GRAY2RGB)
cv2.rectangle(marked, loc, (loc[0] + w, loc[1] + h), (255, 0, 0), 2)
print("Coordinates:", loc, "NCC:", score)
truth_path = IMG / "ground_truth.txt"
if truth_path.exists(): print("Ground truth:", truth_path.read_text(encoding="utf-8"))
bright_scene = np.clip(scene.astype(np.float32) + 25, 0, 255).astype(np.uint8)
bright_loc, bright_score, _ = locate(bright_scene, template)
print("Brightness altered:", bright_loc, bright_score)
for scale in [0.85, 1.15]:
    resized = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    print("Scale", scale, locate(scene, resized)[:2])
sq_loc, sq_score, sq_response = locate(scene, template, cv2.TM_SQDIFF_NORMED)
print("SQDIFF minimum:", sq_loc, sq_score)

fig, axes = plt.subplots(1, 3, figsize=(14, 5)); axes[0].imshow(marked); axes[0].set_title(f"NCC {score:.4f} at {loc}"); axes[1].imshow(response, cmap="viridis"); axes[1].set_title("NCC response map"); axes[2].imshow(sq_response, cmap="viridis"); axes[2].set_title("SQDIFF response")
plt.tight_layout(); plt.savefig(OUT / "05_template_matching.png", dpi=150); plt.show()
