import numpy as np
import matplotlib.pyplot as plt

from src.data import make_image
from src.operators import grad, div
from src.baseline import chambolle


def main():
    # display image
    clean, noisy = make_image(seed=42, sigma=0.1)

    # 261125xqy Chambolle Baseline
    denoised = chambolle(noisy, lam=0.1, max_iter=200)

    print("Image shape:", clean.shape)
    print("Clean range:", clean.min(), clean.max())
    print("Noisy range:", noisy.min(), noisy.max())

    fig, axes = plt.subplots(1, 3, figsize=(8,4))

    axes[0].imshow(clean, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Clean Image')

    axes[1].imshow(noisy, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Noisy Image')

    axes[2].imshow(denoised, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Denoised Image')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    # plt.show()
    
    # 261125xqy Step 3
    mse_noisy = np.mean((noisy - clean) ** 2)
    mse_denoised = np.mean((denoised - clean) ** 2)

    print("Noisy MSE:", mse_noisy)
    print("Denoised MSE:", mse_denoised)

    plt.show()

if __name__ == "__main__":
    main()