
from collections import deque
from typing import Callable, List
import numpy as np
import scipy as sp
from loguru import logger
from continous_attractor_network import AttractorNetwork

def decide_deltas(vels: np.ndarray, scales: list):
    """decide the delta of each layer
    it is assumed that shifting will not ecceeding the network
    however, it is posible that it crosses boundry
    """
    # scales for now is a list of distance per cube, from large to small
    deltas = [0]*len(scales)

    for idx, scale in enumerate(scales):
        # deltas[idx] += vels // scale
        deltas[idx] = vels / scale
        # vel = vels % scale
    # fractional part will remains only goes to the last layer
    # deltas[-1] += vel/scales[-1]
    return deltas
            

class MAn_Slam:
    """
    MdAn_Slam is a continous attractor network based slam with multiple AttractorNetwork
    it is assumed that each layer would use the same dynamic of network
    AKA, it should have ideal N, excite_func, inhibit_func, iteration, forget_ratio
    """
    def __init__(self, network_args: dict, scales: list=[1]) -> None:
        self.networks = [AttractorNetwork(**network_args) for _ in scales]
        self.scales = sorted(scales,reverse=True) # distance per cube, from large to small
        # self.warp = np.zeros(2)
        # self.last_spike = np.zeros(2)
        self._tragety_histroy = [[0,0]]
        
    def get_activity_history(self):
        return [network.activity_history for network in self.networks]
    
    def get_tragety_histroy(self):
        return self._tragety_histroy

    def inject(self, vels:np.ndarray):
        """inject information to the net
        it is assumed that shifting will not ecceeding the network
        however, it is posible that it crosses boundry
        """
        Maxium_vels = self.scales[0] * self.networks[0].shape[0]
        if np.any(vels >= Maxium_vels):
            logger.warning(
                f"the vel of {vels} exceeding the limit {Maxium_vels} of Maximum network"
            )
        deltas=decide_deltas(vels, self.scales)
        for idx, delta in enumerate(deltas):
            # post_spike = self.networks[idx].spike()
            self.networks[idx].iterate(delta)
            # self.networks[idx].activity = self.networks[idx].shift(*delta)
            # self.networks[idx].iterate()
            
            # self.detect_boundry_across(vels, post_spike)
        # todo: detect boundry, or cross layer activity
        
        self._tragety_histroy.append(list(self.decode()))
        pass
    
    def decode(self):
        # Here, we should allow it to be some kind of ngeative
        # AKA, go left at the begining of the network
        posi=[0.0,0.0]
        for idx, network in enumerate(self.networks):
            posi+=network.spike()*self.scales[idx]
        return posi
        
        


class An_Slam:
    """
    An_Slam is a continous attractor network based slam
    """
    def __init__(self, network_args: dict, scale=1) -> None:
        self.networks = AttractorNetwork(**network_args)
        self.scale = scale  # distance per cube
        self.full_scale=self.networks.shape[0]*self.scale[0]
        self.scales_inv = 1/scale
        self.warp = np.zeros(2)
        self.last_spike = np.zeros(2)
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
