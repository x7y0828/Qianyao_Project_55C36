import numpy as np
from src.operators import grad, div
from src.baseline import chambolle, objective
from src.data import make_image
from src.modified import estimate_sigma, adaptive_lambda, adaptive_chambolle
from src.metrics import mse, psnr

##Test for Step 2###
# Test 1: Gradient of a constant image
constant = np.ones((5, 5))
a, b = grad(constant)
print("Constant gradient:", np.max(np.abs(a)), np.max(np.abs(b)))  # should be 0

assert np.allclose(a,0)
assert np.allclose(b,0)


# Test 2: Negative adjoint relationship
rng = np.random.default_rng(42)

u = rng.normal(size=(5, 5))
p1 = rng.normal(size=(5, 5))
p2 = rng.normal(size=(5, 5))

gu1, gu2 = grad(u)

lhs = np.sum(gu1*p1 + gu2*p2)
rhs = -np.sum(u * div(p1, p2))

print("Left side:", lhs)
print("Right side:", rhs)
print("Difference:", np.abs(lhs - rhs))

assert np.allclose(lhs, rhs)

print("All operator tests passed!")


# 261125xqy 
###Test for Step 3###
# Test 1: Chambolle on a constant image
#如果图像中所有像素亮度都是0.5，则不存在需要消除的亮度波动，算法不应该改变图像
constant = np.full((96,96), 0.5)
result = chambolle(constant, lam=0.1, max_iter=200)

print("Constant image error:", np.max(np.abs(result - constant)))  # should be close to 0
assert np.allclose(result, constant)

# Test 2: Mean preservation
#chambolle算法消除图像中的局部波动，但不应该改变图像的整体平均亮度。
clean, noisy = make_image(seed=42, sigma=0.1)
denoised = chambolle(noisy, lam=0.1, max_iter=200)

print("Noisy mean:", np.mean(noisy))
print("Denoised mean:", np.mean(denoised))
print("Mean difference:", np.abs(np.mean(noisy) - np.mean(denoised)))

assert np.allclose(np.mean(noisy), np.mean(denoised))

# Test 3: Objective function comparison
lam = 0.1

obj_noisy = objective(noisy, noisy, lam)
obj_denoised = objective(denoised, noisy, lam)

print("Noisy objective:", obj_noisy)
print("Denoised objective:", obj_denoised)

assert obj_denoised < obj_noisy

# Test 4: MSE comparison
mse_noisy = np.mean((noisy - clean)**2)
mse_denoised = np.mean((denoised - clean)**2)

print("Noisy MSE:", mse_noisy)
print("Denoised MSE:", mse_denoised)

assert mse_denoised < mse_noisy
print("All baseline tests passed!")

###Test for Step4###
### Test for Step 4 ###

# Test 1: Noise estimation accuracy
for sigma in [0.05, 0.1, 0.2]:
    clean, noisy = make_image(seed=42, sigma=sigma)

    sigma_est = estimate_sigma(noisy)
    error = abs(sigma_est - sigma)
    relative_error = error / sigma

    print("True sigma:", sigma)
    print("Estimated sigma:", sigma_est)
    print("Absolute error:", error)
    print("Relative error:", relative_error)
    print("--------------------")

# Test 2: Adaptive regularization parameter
lambda_values = []

for sigma in [0.05, 0.1, 0.2]:
    clean, noisy = make_image(seed=42, sigma=sigma)

    lam = adaptive_lambda(noisy, alpha=10.0)
    lambda_values.append(lam)

    print("True sigma:", sigma)
    print("Adaptive lambda:", lam)

    assert np.isfinite(lam)
    assert lam > 0
    assert np.isclose(
        lam, max(10.0 * estimate_sigma(noisy)**2, 1e-8)
    )

# Check increasing regularization strength
assert lambda_values[0] < lambda_values[1] < lambda_values[2]

# Check zero-noise limiting case
constant = np.full((96, 96), 0.5)
assert adaptive_lambda(constant) == 1e-8

print("Adaptive lambda tests passed!")

# Test 3: Constant image preservation
constant = np.full((96, 96), 0.5)

result, lam = adaptive_chambolle(constant)

print("Adaptive constant image error:",
      np.max(np.abs(result - constant)))

assert np.allclose(result, constant)

# Test 4: Mean preservation and MSE comparison
for sigma in [0.05, 0.1, 0.2]:
    clean, noisy = make_image(seed=42, sigma=sigma)

    denoised, lam = adaptive_chambolle(noisy)

    mean_error = abs(np.mean(noisy) - np.mean(denoised))

    mse_noisy = np.mean((noisy - clean)**2)
    mse_adaptive = np.mean((denoised - clean)**2)

    print("True sigma:", sigma)
    print("Adaptive lambda:", lam)
    print("Mean difference:", mean_error)
    print("Noisy MSE:", mse_noisy)
    print("Adaptive MSE:", mse_adaptive)
    print("--------------------")

    assert np.allclose(np.mean(noisy), np.mean(denoised))
    assert mse_adaptive < mse_noisy

print("All adaptive algorithm tests passed!")

###Test for Step 5###

# Test 1: Identical images
u = np.ones((5, 5))

assert mse(u, u) == 0
assert np.isinf(psnr(u, u))

# Test 2: Known reconstruction error
clean = np.ones((5, 5))
u = np.full((5, 5), 0.9)

print("Test MSE:", mse(u, clean))
print("Test PSNR:", psnr(u, clean))

assert np.isclose(mse(u, clean), 0.01)
assert np.isclose(psnr(u, clean), 20.0)

print("All metric tests passed!")