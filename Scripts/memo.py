# %% [markdown]
# ## Compare between different dts when estimating the position through kinematics
#
# Apparently, smaller dt i.e. higher freq will result in more accuracy position estimation.
#
# Latter we shall see what enhencement can MCAn bring to the control
#


# %%
import math
from typing import Callable
from loguru import logger

import numpy as np
import scipy as sp


# def F(dist):
#     return math.exp(-dist)


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
                dist = math.sqrt(
                    (i - center) ** 2 + (j - center) ** 2
                )  # 计算当前位置到中心位置的距离
                kernel[i, j] = dist_func(dist)  # 使用影响力函数计算影响力值
        return kernel


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
        # self.inhibit()
        self.normize()
        # self.activity_history.append(self.activity.copy())

    def init_activity(self):
        self.activity[0, 0] = 1
        self.iterate()
        # for _ in range(self.iteration):
        #     self.update()
        # self.activity_history.append(self.activity.copy())


# %%


class AN_Slam:
    def __init__(self, network: AttractorNetwork, scale=1) -> None:
        self.network = network
        self.scale = scale  # distance per cube
        self.warp = np.zeros(2)
        self.last_spike = np.zeros(2)

        # self.tragety_histroy = [[0, 0]]
        self.tragety_histroy = []

    def inject(self, vels):
        """inject information to the net
        it is assumed that shifting will not ecceeding the network
        however, it is posible that it crosses boundry
        """
        post_spike = self.network.spike()

        delta = vels / self.scale
        if np.any(abs(delta) >= self.network.shape):
            logger.warning(
                f"the vel of {vels} exceeding the limit {self.network.shape[0] * self.scale} of current network"
            )
        preActivity = self.network.activity.copy()
        # self.network.shift(delta[0], delta[1])
        self.network.copy_shift(delta[0], delta[1],positive_only=True)
        self.network.activity+=preActivity*self.network.forget_ratio
        self.network.iterate()
        self.detect_boundry_across(vels, post_spike)
        self.tragety_histroy.append(list(self.decode()))
        pass

    def detect_boundry_across(self, vels, post_spike):
        vel_direction = np.where(vels > 0, 1, -1)
        Spike_Movement = self.network.spike() - post_spike

        for dimention in range(2):
            if Spike_Movement[dimention] == 0:
                continue
            elif Spike_Movement[dimention] > 0:
                # spike move to the right, while vel is to the left
                # therefore, the spike cross the left boundry
                if vels[dimention] < 0:
                    self.warp[dimention] -= self.network.shape[dimention] * self.scale
            elif Spike_Movement[dimention] < 0:
                # spike move to the left, while vel is to the right
                # therefore, the spike cross the right boundry
                if vels[dimention] > 0:
                    self.warp[dimention] += self.network.shape[dimention] * self.scale
            else:
                raise ValueError(
                    "Spike_Movement is not equal to 0, nor positive or negative"
                )

        # # Used to be a more complicated way to detect the boundry
        # # but bug at Spike_Movement=0
        #
        # Spike_Movement_direction=np.where(Spike_Movement > 0, 1, -1)
        # # vels_cross_boundry=vel_direction-Spike_Movement_direction

        # # the compareson result would be 0 if the two direction are the same
        # # and 2 if R-L therefore vel twoards right and spike move left, aka, cross the right boundry
        # # and -2 if L-R
        # vels_cross_boundry=vel_direction-Spike_Movement_direction
        # self.warp+=(vels_cross_boundry/2)*self.network.shape*self.scale

    def decode(self):
        max_position = self.network.spike()
        return max_position * self.scale + self.warp


# %%

from matplotlib import pyplot as plt
from math import ceil, floor, sqrt


def construct_SLAM(exite_kernel_size=7, inhibt_kernel_size=5, net_size=200):
    dist_func = lambda dist: math.exp(-dist)
    # use gause kernel
    # dist_func = lambda dist: math.exp(-(dist**2) / 2)

    # excite_func = lambda self: sp.signal.convolve2d(
    #     self.activity,
    #     KernelFactory.create(exite_kernel_size, dist_func),
    #     mode="same",
    #     boundary="wrap",
    # )
    def excite_func(Network: AttractorNetwork):
        local_excite = sp.signal.convolve2d(
            Network.activity,
            KernelFactory.create(exite_kernel_size, dist_func),
            mode="same",
            boundary="wrap",
        )
        return Network.activity + local_excite

    def inhibit_func(Network: AttractorNetwork):
        local_inhibit = 0
        local_inhibit = sp.signal.convolve2d(
            Network.activity,
            KernelFactory.create(inhibt_kernel_size, dist_func),
            mode="same",
            boundary="wrap",
        )
        global_inhibit=Network.activity.sum()*6.51431074e-04
        Network.activity-=(global_inhibit+local_inhibit)
    
    # Kernel = KernelFactory.create(kernel_size, dist_func)

    an = AttractorNetwork(
        net_size, excite_func, inhibit_func, iteration=3, forget_ratio=0
    )

    test_ob = AN_Slam(an, 1)
    return test_ob


# %%
import yaml
import os

from tqdm import tqdm
import numpy as np
import scipy as sp

# np.random.seed(3407)

# TotalTime=1e+2 #s
# dt=1 #s
# TimeStepNum=int(TotalTime//dt+1)
# TimeSteps=np.arange(0,TotalTime+dt,dt)


# # vels=np.random.randn(2,TimeStepNum)
# # vel min=scale,max is N*scale
# vels=np.random.rand(2,TimeStepNum)*(net_size-1)*test_ob.scale+test_ob.scale
# vels = np.insert(vels, 0, 0, axis=1)
# vels.round(2)


# for t in range(TimeStepNum):
#     # network_weight+=np.outer(test_ob.activity,vels[:,t])
#     # test_ob.weights_history.append(network_weight.copy())
#     test_ob.inject(vels[:,t])
def parseVelType(vel_profile):
    # either a list or a string
    # if list, then it is [vel_min,vel_max,vel_var]
    # if string, then it mostly "GTpose" for kitty
    if type(vel_profile) == list:
        vel_min, vel_max, vel_var = vel_profile
        veltype = f"Speeds{vel_min}to{vel_max}var{vel_var}"
    else:
        veltype = vel_profile
    return veltype


def load_traverse_info(traverseInfo_filePart, index):
    if "kitti" in traverseInfo_filePart:
        traverseInfo_file = f"{traverseInfo_filePart}{index:02d}.npy"
        vel, ang_vel = np.load(traverseInfo_file)
    else:
        traverseInfo_file = f"{traverseInfo_filePart}{index}.npz"
        traverse_info = np.load(traverseInfo_file, allow_pickle=True)
        vel, ang_vel = traverse_info["speeds"], traverse_info["angVel"]
    return vel, ang_vel


def project_vel(vels, angVels):
    """project the velocity to the x,y axis
    vels, angVels both 1*N array
    return 2*N array
    """
    angs = np.comsum(angVels)
    return np.array([np.cos(angVels), np.sin(angVels)]) * vels
    # return np.vstack([vels*np.cos(angVels),vels*np.sin(angVels)])


def load_dataset(entry, configs_file="Datasets/profile.yml"):
    configs = yaml.load(open(configs_file, "r"), Loader=yaml.FullLoader)
    return configs[entry]


# %%
def runCanAtPath(
    City, index, scaleType, traverseInfo_filePart, vel_profile, pathfile, an_slam=None
):
    """running a single path in a city
    City = Berlin or Japan or Brisbane or Newyork or Kitti;
    scaleType = Single or Multi;
    traverseInfo_filePart = the file path to the traverse info, will append the index to the end;
    vel_profile = either a list or a string
    pathfile = the file path to save the output
    index = the index of the path
    """
    if an_slam is None:
        an_slam = construct_SLAM()

    vel, angVel = load_traverse_info(traverseInfo_filePart, index)

    if scaleType == "Multi":
        scales = [0.25, 1, 4, 16]
        numNeurons = 100
    elif scaleType == "Single":
        scales = [1]
        numNeurons = 200

    if City == "Kitti":
        test_length = len(vel)
    else:
        test_length = min(len(vel), 500)  # supress for testing

    if type(vel_profile) == list:
        # use a random seed to generate the velocities for reproducibility
        vel_min, vel_max, vel_var = vel_profile
        np.random.seed(index * vel_var)
        vel = np.random.uniform(vel_min, vel_max, test_length)

    TIMESTEP_LEN = len(vel)

    """__________________________Storage and initilisation parameters______________________________"""
    theta_weights = np.zeros(360)
    theata_called_iters = 0
    # wrap_counter=[0,0,0,0,0,0]
    q = np.zeros(3)
    q_err = np.zeros(3)
    # grid_expect=np.zeros(2)
    x_grid_expect, y_grid_expect = 0, 0

    # posi_integ_log=np.zeros([TIMESTEP_LEN,2])
    # grid_log=np.zeros([TIMESTEP_LEN,2])
    # integ_err_log=np.zeros([TIMESTEP_LEN,2])

    posi_integ_log = [[0, 0] for _ in range(TIMESTEP_LEN)]

    tbar = tqdm(range(1, TIMESTEP_LEN), disable="GITHUB_ACTIONS" in os.environ)
    for i in tbar:
        q[2] += angVel[i]
        q[0], q[1] = q[0] + vel[i] * np.cos(q[2]), q[1] + vel[i] * np.sin(q[2])
        posi_integ_log[i] = list(q[0:2])

        decare_vel = np.array([vel[i] * np.cos(q[2]), vel[i] * np.sin(q[2])])

        an_slam.inject(decare_vel)

    return posi_integ_log
    # print(f"finished {City}, id {index}")


configs = load_dataset("SelectiveMultiScale")
City = "Newyork"
index = 0
scaleType = "Single"
traverseInfo_filePart = configs[City]["traverseInfo_file"]
vel_profile = configs[City]["vel_profile"]
veltype = parseVelType(vel_profile)
pathfile = f"./Results/{City}/CAN_Experiment_Output_{scaleType}/TestingTrackswith{veltype}_Path"

# higher net_size, higher accuracy
# which is odd, since the scale is defined at the cell level
# however, it is possible that the higher net_size, the more accurate the excite and inhibit kernel
# or, the wrap around is introduced other noises
an_slam = construct_SLAM(exite_kernel_size=7, inhibt_kernel_size=5, net_size=50)

posi_integ_log = runCanAtPath(
    City, index, scaleType, traverseInfo_filePart, vel_profile, pathfile, an_slam
)



CAN_SLAM_Track = an_slam.tragety_histroy
SLAM_Spike_histroy = an_slam.network.activity_history









# %%

# test_ob.activity
data = CAN_SLAM_Track
data2 = posi_integ_log
# data=test_ob.tragety_histroy
N = len(data)
n_col = 10
n_row = ceil(N / n_col)
size_mag = 0.5
plt.close("all")
# data
plt.plot(*zip(*data))
plt.plot(*zip(*data2))
plt.gca().invert_yaxis()
fig.show()

# %%
sampled_SLAM_Spike_history = SLAM_Spike_histroy[::1]
# plt.close("all")

data = sampled_SLAM_Spike_history

N = len(data)
# n_col = 10
n_row = 10
n_col = ceil(N / n_row)

n_col, n_row = n_row, n_col

fig, ax = plt.subplots(
    n_row, n_col, figsize=(n_col * size_mag, n_row * size_mag), dpi=300
)

axs = ax.ravel()

for index in range(N):
    activity = data[index]

    """plot"""

    # cax=axs[index]
    axs[index].imshow(activity, cmap="viridis")
    axs[index].invert_yaxis()
    axs[index].axis("off")
    # axs[index].invert_yaxis()
    # axs[index].set_title(f'id={index}')


fig.suptitle(f"title")
# add cbar

plt.subplots_adjust(top=0.93)
# Adjust spacing between subplots
# plt.tight_layout()
fig.show()


# # %%
# import numpy as np
# import matplotlib.pyplot as plt

# # Create a 5x5 matrix filled with random values
# matrix = np.random.rand(5, 5)

# # Create subplots
# fig, axs = plt.subplots(1, 2)

# # Plot the matrix in the first subplot
# axs[0].imshow(matrix, cmap="viridis")
# axs[0].set_title("Matrix")

# # Plot the matrix transposed in the second subplot
# axs[1].imshow(matrix.T, cmap="viridis")
# axs[1].set_title("Transposed Matrix")

# # Adjust spacing between subplots
# plt.tight_layout()

# # Display the plot
# lt.show()

# # %%

# # class Object(object):
# #     def __init__(self,a,b,func) -> None:
# #         self.a=a
# #         self.b=b
# #         self.func=func
# #     def dosth(self):
# #         return self.func(self)

# # def func(Object):
# #     return Object.a+Object.b

# # test_ob=Object(1,2,func)
# # test_ob.dosth()

# # %%

# N = 4  # 卷积核的大小

# kernel = np.zeros((N, N))  # 创建一个全零的卷积核

# center = (N - 1) / 2  # 卷积核的中心位置

# for i in range(N):
#     for j in range(N):
#         dist = math.sqrt((i - center) ** 2 + (j - center) ** 2)  # 计算当前位置到中心位置的距离
#         kernel[i, j] = F(dist)  # 使用影响力函数计算影响力值

# weights = np.random.randn(N, N)
# # print(kernel)
# conv = sp.signal.convolve2d(weights, kernel, mode="same", boundary="wrap")
# circ = np.zeros(N, N)
# # for i in range(N):
# #     for j in range(N):
# #         circ[i,j]=np.sum

# # %%
# conv

# # %%

# %%
