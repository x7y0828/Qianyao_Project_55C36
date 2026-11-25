import numpy as np
from src.baseline import chambolle

###Step 4: Noise-adaptive Regularisation###
# estimate noise standard deviation
# Median Absolute Deviation 中位数绝对偏差
def estimate_sigma(noisy):
    usable = 2 * (noisy.shape[1] // 2)      # ensure the number of columns included is even 确保参与计算的列数为偶数

    differences = (
        noisy[:, 1:usable:2] - noisy[:, :usable:2]
    ) / np.sqrt(2.0)        #calculate brightness difference between adjacent pixel and normalisation 计算相邻像素对的亮度差，并归一化

    centre = np.median(differences)         # calculate the median 计算中位数
    mad = np.median(np.abs(differences - centre))#计算差值相对于中位数偏差的中位数

    return mad / 0.67449        # convert MAD to an estimate of standard deviation of Gaussion noise 将MAD转换为高斯噪声标准差的估计值


