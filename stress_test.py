import numpy as np
import matplotlib.pyplot as plt
from src.baseline import chambolle
from src.modified import adaptive_chambolle

from src.data import make_stress_image
from src.metrics import mse, psnr, gradient_mse

# generate stress image
clean, noisy = make_stress_image(seed=300, sigma=0.1)       #seed=300，避免与validation和test重叠

# baseline denoising
denoised_baseline = chambolle(noisy, lam=0.15, max_iter=200)

# adaptive denoising
denoised_adaptive, lam_adaptive = adaptive_chambolle(noisy, alpha=15, max_iter=200)

print("Adaptive lambda:", lam_adaptive)

# evaluate stress test results
for name, image in [
    ("Noisy", noisy),
    ("Baseline", denoised_baseline),
    ("Adaptive", denoised_adaptive)
]:
    print("--------------------")
    print("Method:", name)
    print("MSE:", mse(clean, image))
    print("PSNR:", psnr(clean, image))
    print("Gradient MSE:", gradient_mse(clean, image))

    # texture region evaluation
    print("Texture MSE:", mse(
        clean[32:64, 32:64],
        image[32:64, 32:64]
    ))

# display images
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(clean, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Clean Stress Image")

axes[1].imshow(noisy, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Noisy Stress Image")

axes[2].imshow(denoised_baseline, cmap="gray", vmin=0, vmax=1)
axes[2].set_title("Baseline Denoised Image")

axes[3].imshow(denoised_adaptive, cmap="gray", vmin=0, vmax=1)
axes[3].set_title("Adaptive Denoised Image")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()