import numpy as np
from src.operators import grad

###Step 5: Evaluation Metrics###
# mean Squared Error
def mse(u, clean):
    return np.mean((u - clean)**2)


# peak Signal-to-Noise Ratio
def psnr(u, clean):
    error = mse(u, clean)

    if error == 0:
        return np.inf

    return 10 * np.log10(1 / error)

# gradient Mean Squared Error
def gradient_mse(u, clean):
    u1, u2 = grad(u)
    c1, c2 = grad(clean)

    return np.mean((u1 - c1)**2 + (u2 - c2)**2)