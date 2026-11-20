import numpy as np
import matplotlib.pyplot as plt

###Step1: Generate Noisy Images###
# make image
def make_image(seed, size=96, sigma=0.1):       #seed-固定随机生成结果，方便复现实验 sigma-高斯噪声标准差，控制噪声强度
    rng = np.random.default_rng(seed)
    clean = np.full((size, size), 0.15)
    yy, xx = np.mgrid[:size, :size]
    for _ in range(4):
        row, col = rng.integers(0, size-20, size=2)
        height, width = rng.integers(10, 22, size=2)
        clean[row:row+height, col:col+width] = (
            rng.uniform(0.3, 0.9))
    for _ in range(2):
        cy, cx = rng.integers(15, size-15, size=2)
        radius = rng.integers(6, 14)
        mask = (yy-cy)**2 + (xx-cx)**2 < radius**2
        clean[mask] = rng.uniform(0.3, 0.9)

    noisy = clean + sigma*rng.normal(size=clean.shape)
    return clean, noisy

# display image
clean, noisy = make_image(seed=42, sigma=0.1)

fig, axes = plt.subplots(1, 2, figsize=(8,4))

axes[0].imshow(clean, cmap='gray', vmin=0, vmax=1)
axes[0].set_title('Clean Image')

axes[1].imshow(noisy, cmap='gray', vmin=0, vmax=1)
axes[1].set_title('Noisy Image')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
# plt.show()

print("Image shape:", clean.shape)
print("Clean range:", clean.min(), clean.max())
print("Noisy range:", noisy.min(), noisy.max())

###Step2: Implement Gradient and Divergence###
# calculate the difference between adjacent pixels above and below,
# to the left and to the right of an image
# 计算图像上下左右相邻像素的差
def grad(u):
    a, b = np.zeros_like(u), np.zeros_like(u)
    a[:-1] = u[1:] - u[:-1]             #用于处理图像边界
    b[:, :-1] = u[:, 1:] - u[:, :-1]    #用于处理图像边界
    return a, b
# calculate the divergence based on vector fields in two directions 
# 根据两个方向的向量场计算散度
def div(a, b):
    out = np.zeros_like(a)
    out[:-1] += a[:-1]
    out[1:] -= a[:-1]
    out[:, :-1] += b[:, :-1]
    out[:, 1:] -= b[:, :-1]
    return out



###Test for Step 2###
# Test 1: Gradient of a constant image
constant = np.ones((5, 5))
a, b = grad(constant)
print("Constant gradient:", np.max(np.abs(a)), np.max(np.abs(b)))
  # should be 0

# Test 2: Negative adjoint relationship
rng = np.random.default_rng(42)

u =  rng.normal(size=(5, 5))
p1 = rng.normal(size=(5, 5))
p2 = rng.normal(size=(5, 5))

gu1, gu2 = grad(u)

lhs = np.sum(gu1*p1 + gu2*p2)
rhs = np.sum(u * div(p1, p2))

print("Left side:", lhs)
print("Right side:", rhs)
print("Difference:", abs(lhs - rhs))  # should be small

assert np.allclose(lhs, rhs)
