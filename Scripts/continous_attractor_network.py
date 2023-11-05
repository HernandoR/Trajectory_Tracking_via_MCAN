
from typing import Callable
import numpy as np
import scipy as sp

class AttractorNetwork:
    def __init__(
        self,
        N: int,
        # excite_rad: int,
        # inhibit_rad: int,
        excite_func: Callable,
        inhibit_func: Callable,
        # exite_ker: np.ndarray,
        # inhibit_ker: np.ndarray,
        iteration: int = 3,
        forget_ratio: float = 0.4,
        # scale=1 # distance per cube
    ) -> None:
        self.N = N
        self.shape = [N, N]
        # self.excite_ker=exite_ker
        # self.inhibit_ker=inhibit_ker
        # self.excite_rad = excite_rad # kernel size
        # self.inhibit_rad = inhibit_rad # kernel size

        # this 2 func should not modify the network itself
        self.excite_func = excite_func
        self.inhibit_func = inhibit_func

        self.iteration = iteration
        self.forget_ratio = forget_ratio
        # self.scale=scale

        self.activity = np.zeros(self.shape)
        self.activity_mag = 1
        self.activity_history = []
        # self.activity_history.append(self.activity.copy())

        self.weights = np.zeros(self.shape)
        self.weights_mag = 1
        self.weights_history = []
        # self.weights_history.append(self.weights.copy())
        self.init_activity()

    def spike(self):
        # return self.activity.argmax()
        return np.array(np.unravel_index(self.activity.argmax(), self.shape))

    def excite(self):
        self.activity = self.excite_func(self)
        # self.activity_history.append(self.activity.copy())

    def inhibit(self):
        self.activity = self.inhibit_func(self)
        # minimum is 0
        self.activity[self.activity<0]=0
        # self.activity_history.append(self.activity.copy())
        
    def scale_range (self, input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    def normize(self):
        # self.activity = self.scale_range(self.activity, 0, 1)
        # self.activity = np.normize(self.activity)
        
        self.activity_mag = np.linalg.norm(self.activity)
        self.activity /= self.activity_mag
        # self.activity_history.append(self.activity.copy())

    def copy_shift(self, delta_row, delta_col, positive_only=False):
        preActivity = self.activity.copy()
        if positive_only:
            preActivity[preActivity<0]=0
        self.activity = sp.ndimage.shift(
            preActivity, (delta_row, delta_col), mode="wrap"
        ) + self.activity * self.forget_ratio

    def shift(self, delta_row, delta_col):
        # self.activity *= self.forget_ratio
        self.activity = sp.ndimage.shift(
            self.activity, (delta_row, delta_col), mode="wrap"
        )
        # self.activity_history.append(self.activity.copy())

    def iterate(self):
        # for _ in range(self.iteration):
        self.update()
        assert not np.any(np.isnan(self.activity))
        self.activity_history.append(self.activity.copy())

    def update(self):
        self.excite()
        self.inhibit()
        self.normize()
        # self.activity_history.append(self.activity.copy())

    def init_activity(self):
        self.activity[0, 0] = 1
        self.iterate()
        # for _ in range(self.iteration):
        #     self.update()
        # self.activity_history.append(self.activity.copy())
