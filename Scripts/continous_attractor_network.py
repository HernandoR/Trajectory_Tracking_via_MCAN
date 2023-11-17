
from typing import Callable,List
from loguru import logger
import numpy as np
import scipy as sp

class AttractorNetwork:
    def __init__(
        self,
        N: int=None,
        excite_func: Callable = None,
        inhibit_func: Callable = None,
        iteration: int = 3,
        forget_ratio: float = 0.4,
        # scale=1 # distance per cube
    ) -> None:

        if any([arg is None for arg in [N, excite_func, inhibit_func]]):
            raise ValueError("At least one of N, excite_func, inhibit_func should be provided")

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
        self.warp=np.zeros(2)

    def spike(self):
        # return self.activity.argmax()
        # should be negative if in left or bottom part
        # return np.array(np.unravel_index(self.activity.argmax(), self.shape))
        raw_spike = np.array(np.unravel_index(self.activity.argmax(), self.shape))
        return raw_spike - np.array(self.shape) // 2

    def excite(self,activity):
        activity = self.excite_func(activity)
        return activity
        # self.activity_history.append(self.activity.copy())

    def inhibit(self,activity):
        activity = self.inhibit_func(activity)
        # minimum is 0
        activity[activity<0]=0
        return activity
        # self.activity_history.append(self.activity.copy())
    def log_activity(self):
        self.activity_history.append(self.activity.copy())

    def scale_range (self, input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    def normize(self,activity):
        # self.activity = self.scale_range(self.activity, 0, 1)
        # self.activity = np.normize(self.activity)

        activity_mag = np.linalg.norm(self.activity)
        activity /= activity_mag
        return activity

    # def copy_shift(self, delta_row, delta_col, positive_only=False):
    #     preActivity = self.activity.copy()
    #     if positive_only:
    #         preActivity[preActivity<0]=0
    #     self.activity = sp.ndimage.shift(
    #         preActivity, (delta_row, delta_col), mode="wrap"
    #     ) + self.activity * (1-self.forget_ratio)

    def shift(self, delta_row, delta_col, activity = None ):
        if activity is None:
            activity = self.activity.copy()
        activity = sp.ndimage.shift(
            activity, (delta_row, delta_col), mode="wrap"
        )
        return activity
        # self.activity_history.append(self.activity.copy())

    def iterate(self,delta=[0,0]):
        # for _ in range(self.iteration):
        self.update(delta)
        assert not np.any(np.isnan(self.activity))
        self.log_activity()

    def update(self,delta=[0,0]):
        X=self.activity.copy()
        S=self.shift(delta[0], delta[1])
        E=self.excite(S)
        I=self.inhibit(S+E)
        self.activity = self.normize(X*(1-self.forget_ratio)+I*self.forget_ratio)
        pass

    def init_activity(self):
        # self.activity[0, 0] = 1
        if self.N is None:
            raise ValueError("N should be provided")

        if self.N % 2 == 0:
            # even, activare the center 4
            self.activity[self.N // 2: self.N // 2 + 2, self.N // 2: self.N // 2 + 2] = 0.25
        else:
            # odd, activare the center 1
            self.activity[self.N // 2, self.N // 2] = 1

        self.iterate()
        # for _ in range(self.iteration):
        #     self.update()
        # self.activity_history.append(self.activity.copy())

    # report if boundry is crossed
    # This function should be called by a higher level object
    # as the network itself keeps a warp manifold
    def detect_boundry_across(self, injected_dir: np.asarray, post_spike: np.asarray, recall_fun: Callable=None) -> np.asarray:
        injected_dir = np.where(injected_dir > 0, 1, np.where(injected_dir < 0, -1, 0)) # 1 for positive, -1 for negative, movement on each dimention
        Spike_Movement = self.spike() - post_spike

        boundary_across=[0 for _ in range(len(injected_dir))]

        for idx, (injected_dir, Spike_Movement) in enumerate(zip(injected_dir, Spike_Movement)):

            if Spike_Movement == 0:
                # no boundry cross if spike is not moving
                boundary_across[idx] = 0
            elif Spike_Movement > 0 and injected_dir < 0:
                    # self.warp -= self.networks.shape * self.scale
                boundary_across[idx] = -1 # negative value represents cross to the left
            elif Spike_Movement < 0 and injected_dir > 0:
                # spike move to the left, while vel is to the right
                # therefore, the spike cross the right boundry
                    # self.warp += self.networks.shape * self.scale
                boundary_across[idx] = 1 # positive value represents cross to the right
            else:
                continue
                # raise ValueError(
                #     "Spike_Movement is not equal to 0, nor positive or negative"
                # )

        if recall_fun is not None:
            recall_fun(boundary_across)
        return boundary_across

    def handle_lower_boundry_across(self, boundary_across: np.ndarray) -> None:
        pass

    def respond_to_boundary_cross(self, boundary_across: np.ndarray) -> None:
        self.warp += self.networks.shape * boundary_across
        logger.warning(f'cross boundry {boundary_across} been handled by warp')
        pass