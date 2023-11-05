# %% [markdown]
# ## Compare between different dts when estimating the position through kinematics
#
# Apparently, smaller dt i.e. higher freq will result in more accuracy position estimation.
#
# Latter we shall see what enhencement can MCAn bring to the control
#


# %%
import math
from re import S
from typing import Callable
from loguru import logger

import numpy as np
import scipy as sp

import pykitti
import yaml
import os

from tqdm import tqdm
import numpy as np
import scipy as sp


from matplotlib import pyplot as plt
from math import ceil, floor, sqrt

from Scripts.continous_attractor_network import AttractorNetwork
from can_slam import An_Slam


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



# %%



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

def load_kitti_odometry(index):
    base_dir="Datasets/kitti_dataset"
    kitti_odometry=pykitti.odometry(base_dir,sequence=index)
    gt_poses_y = -np.array(kitti_odometry.poses)[:, :, 3][:, 0]
    gt_poses_x = np.array(kitti_odometry.poses)[:, :, 3][:, 2]
    gt_poses = np.vstack([gt_poses_x, gt_poses_y]).T
    return gt_poses

# %%
def runCanAtPath(
    City, index, scaleType, traverseInfo_filePart, vel_profile,dt, pathfile, an_slam=None
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
        # vel/=20
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

        descartes_vel = np.array([vel[i] * np.cos(q[2]), vel[i] * np.sin(q[2])])

        an_slam.inject(descartes_vel)

    return posi_integ_log
    # print(f"finished {City}, id {index}")


def construct_SLAM(exite_kernel_size=7, inhibt_kernel_size=5, net_size=15):
    dist_func = lambda dist: math.exp(-dist)

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
        local_inhibit = 0.16*sp.signal.convolve2d(
            Network.activity,
            KernelFactory.create(inhibt_kernel_size, dist_func),
            mode="same",
            boundary="wrap",
        )
        global_inhibit=Network.activity.sum()*6.51431074e-04
        
        return Network.activity-(global_inhibit+local_inhibit)
    
    # Kernel = KernelFactory.create(kernel_size, dist_func)

    an = AttractorNetwork(
        net_size, excite_func, inhibit_func, iteration=3, forget_ratio=0
    )

    test_ob = An_Slam(an, 1)
    return test_ob


configs = load_dataset("SelectiveMultiScale")
City = "Kitti"
index = 0
dt=1
scaleType = "Single"
traverseInfo_filePart = configs[City]["traverseInfo_file"]
vel_profile = configs[City]["vel_profile"]
veltype = parseVelType(vel_profile)
pathfile = f"./Results/{City}/CAN_Experiment_Output_{scaleType}/TestingTrackswith{veltype}_Path"

# higher net_size, higher accuracy
# which is odd, since the scale is defined at the cell level
# however, it is possible that the higher net_size, the more accurate the excite and inhibit kernel
# or, the wrap around is introduced other noises
an_slam = construct_SLAM(exite_kernel_size=7, inhibt_kernel_size=5, net_size=30)

posi_integ_log = runCanAtPath(
    City, index, scaleType, traverseInfo_filePart, vel_profile ,dt, pathfile, an_slam
)



CAN_SLAM_Track = an_slam.tragety_histroy
SLAM_Spike_histroy = an_slam.network.activity_history








# %%

# test_ob.activity
data = CAN_SLAM_Track
data2 = posi_integ_log

ATE = np.linalg.norm(np.array(data)-np.array(data2))/len(data)

# data=test_ob.tragety_histroy
N = len(data)
n_col = 10
n_row = ceil(N / n_col)
size_mag = 0.5
plt.close("all")
# data
plt.plot(*zip(*data),label="CAN")
plt.plot(*zip(*data2),label="path integration")
# data3 = load_kitti_odometry('00')
# plt.plot(*zip(*data3),label="ground truth")
# plt.gca().invert_yaxis()
plt.gca().set_title(f"ATE={ATE}")
# fig.show()

sampled_SLAM_Spike_history = SLAM_Spike_histroy[::80]
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
