import numpy as np
from src.operators import div, grad
# 261125xqy
###Step 3: Chambolle Baseline###
def chambolle(f, lam=0.1, tau=1/8, max_iter=200):       #lam0.1暂时选定的调试参数
    # initialise dual variables
    p1 = np.zeros_like(f)
    p2 = np.zeros_like(f)

    # fixed-point ieration
    for k in range(max_iter):    
        g1, g2 = grad(div(p1, p2) - f / lam)    # calculate gradient
        g_norm = np.sqrt(g1**2 + g2**2)          # calculate gradient magnitude
        denominator = 1 + tau * g_norm          #update dual variables
        p1 = (p1 + tau * g1) / denominator
        p2 = (p2 + tau * g2) / denominator
    # recover denoised image
    u = f - lam * div(p1, p2)
    return u

# total variation
def tv(u):
    a, b =  grad(u)
    return np.sum(np.sqrt(a**2 + b**2))

# objective function
def objective(u, f, lam):
    fidelity = 0.5 * np.sum((u-f)**2)
    regularisation = lam * tv(u)
    return fidelity + regularisation