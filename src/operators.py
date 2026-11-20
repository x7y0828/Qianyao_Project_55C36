import numpy as np

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