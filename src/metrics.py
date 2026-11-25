import numpy as np

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