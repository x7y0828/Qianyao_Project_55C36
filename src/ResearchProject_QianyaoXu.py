import numpy as np
import matplotlib.pyplot as plt

from src.data import make_image
from src.operators import grad, div


def main():
    # display image
    clean, noisy = make_image(seed=42, sigma=0.1)

    print("Image shape:", clean.shape)
    print("Clean range:", clean.min(), clean.max())
    print("Noisy range:", noisy.min(), noisy.max())

    fig, axes = plt.subplots(1, 2, figsize=(8,4))

    axes[0].imshow(clean, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Clean Image')

    axes[1].imshow(noisy, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Noisy Image')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    # plt.show()


    ###Test for Step 2###
    # Test 1: Gradient of a constant image
    # constant = np.ones((5, 5))
    a, b = grad(clean)
    divergence = div(a, b)
    print("Gradient shape:", a.shape, b.shape)
    print("Divergence shape:", divergence.shape)
    # should be 0

plt.show()

if __name__ == "__main__":
    main()