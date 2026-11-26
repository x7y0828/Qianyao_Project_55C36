import numpy as np

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

# generate textured stress image
def make_stress_image(seed, size=96, sigma=0.1):
    clean, noisy = make_image(seed, size, sigma)

    # preserve the original Gaussian noise
    noise = noisy - clean       #提取原来的高斯噪声

    # generate checkerboard texture
    yy, xx = np.mgrid[:32, :32]
    checkerboard = (yy + xx) % 2        #生成交替排列的0和1

    # add texture to a local region
    clean[32:64, 32:64] = 0.3 + 0.4 * checkerboard      #在图像中央加入32×32的纹理区域

    noisy = clean + noise       #重新叠加高斯噪声

    return clean, noisy