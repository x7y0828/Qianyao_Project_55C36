import numpy as np
import matplotlib.pyplot as plt

from src.data import make_image
from src.operators import grad, div
from src.baseline import chambolle
from src.modified import adaptive_chambolle
from src.metrics import mse, psnr


def main():
    # display image
    clean, noisy = make_image(seed=42, sigma=0.1)

    # 261125xqy Chambolle Baseline
    denoised = chambolle(noisy, lam=0.1, max_iter=200)

    print("Image shape:", clean.shape)
    print("Clean range:", clean.min(), clean.max())
    print("Noisy range:", noisy.min(), noisy.max())

    # noise-adaptive TV denoising
    denoised_adaptive, lam_adaptive = adaptive_chambolle(
        noisy, alpha=10.0, max_iter=200
    )

    print("Adaptive lambda:", lam_adaptive)

    fig, axes = plt.subplots(1, 4, figsize=(12,4))

    axes[0].imshow(clean, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Clean Image')

    axes[1].imshow(noisy, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Noisy Image')

    axes[2].imshow(denoised, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Baseline Denoised Image')

    axes[3].imshow(denoised_adaptive, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title('Adaptive Denoised Image')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    # plt.show()
    
    # 261125xqy Step 3
    mse_noisy = mse(noisy, clean)
    mse_denoised = mse(denoised, clean)
    mse_adaptive = mse(denoised_adaptive, clean)

    print("Noisy MSE:", mse_noisy)
    print("Denoised MSE:", mse_denoised)
    print("Adaptive MSE:", mse_adaptive)
    # Step5, add PSNR output
    print("Noisy PSNR:", psnr(noisy, clean))
    print("Baseline PSNR:", psnr(denoised, clean))
    print("Adaptive PSNR:", psnr(denoised_adaptive, clean))

    plt.show()

if __name__ == "__main__":
    main()