from typing import Callable
import math

import numpy as np



class KernelFactory:
    # def __init__(self, kernel_size, weight_func):
    #     self.kernel_size = kernel_size
    #     self.weight_func = weight_func
    @staticmethod
    def create(kernel_size, dist_func: Callable) -> np.ndarray:
        kernel = np.zeros((kernel_size, kernel_size))  # 创建一个全零的卷积核
        center = (kernel_size - 1) / 2  # 卷积核的中心位置
        for i in range(kernel_size):
            for j in range(kernel_size):
                # dist = math.sqrt(
                #     (i - center) ** 2 + (j - center) ** 2
                # )  # 计算当前位置到中心位置的距离
                # dist is normized to 1, aka, the maxium distance is 1, center is 0
                dist = math.sqrt(
                    (i - center) ** 2 + (j - center) ** 2
                ) / center
                dist-min(1,dist)
                kernel[i, j] = dist_func(dist)  # 使用影响力函数计算影响力值
        return kernel