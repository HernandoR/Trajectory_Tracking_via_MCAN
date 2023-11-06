
from typing import Callable
import numpy as np
import scipy as sp
from loguru import logger
from continous_attractor_network import AttractorNetwork


class An_Slam:
    """
    An_Slam is a continous attractor network based slam
    """
    def __init__(self, network: AttractorNetwork, scale=1) -> None:
        self.networks = network
        self.scale = scale  # distance per cube
        self.scales_inv = 1/scale
        self.warp = np.zeros(2)
        self.last_spike = np.zeros(2)

        # self.tragety_histroy = [[0, 0]]
        self.tragety_histroy = [[0,0]]

    def inject(self, vels):
        """inject information to the net
        it is assumed that shifting will not ecceeding the network
        however, it is posible that it crosses boundry
        """
        post_spike = self.networks.spike()

        delta = vels * self.scales_inv
        
        if np.any(abs(delta) >= self.networks.shape):
            logger.warning(
                f"the vel of {vels} exceeding the limit {self.networks.shape[0] * self.scale} of current network"
            )
        preActivity = self.networks.activity.copy()
        # self.network.shift(delta[0], delta[1])
        self.networks.copy_shift(delta[0], delta[1],positive_only=True)
        # self.network.activity+=preActivity*self.network.forget_ratio
        self.networks.iterate()
        self.detect_boundry_across(vels, post_spike)
        self.tragety_histroy.append(list(self.decode()))
        pass

    def detect_boundry_across(self, vels, post_spike):
        vel_direction = np.where(vels > 0, 1, -1)
        Spike_Movement = self.networks.spike() - post_spike

        for dimention in range(2):
            if Spike_Movement[dimention] == 0:
                continue
            elif Spike_Movement[dimention] > 0:
                # spike move to the right, while vel is to the left
                # therefore, the spike cross the left boundry
                if vels[dimention] < 0:
                    self.warp[dimention] -= self.networks.shape[dimention] * self.scale
            elif Spike_Movement[dimention] < 0:
                # spike move to the left, while vel is to the right
                # therefore, the spike cross the right boundry
                if vels[dimention] > 0:
                    self.warp[dimention] += self.networks.shape[dimention] * self.scale
            else:
                raise ValueError(
                    "Spike_Movement is not equal to 0, nor positive or negative"
                )

    def decode(self):
        max_position = self.networks.spike()
        return max_position * self.scale + self.warp
    
    def __str__(self) -> str:
        profiles={
            "scale":self.scale,
            "network":self.networks,
        }
