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

from continous_attractor_network import AttractorNetwork
from can_slam import An_Slam

from copy import deepcopy


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
    base_dir = "Datasets/kitti_dataset"
    kitti_odometry = pykitti.odometry(base_dir, sequence=index)
    gt_poses_y = -np.array(kitti_odometry.poses)[:, :, 3][:, 0]
    gt_poses_x = np.array(kitti_odometry.poses)[:, :, 3][:, 2]
    gt_poses = np.vstack([gt_poses_x, gt_poses_y]).T
    return gt_poses


# %%
def get_scales_and_neurons(scale_type):
    if scale_type == "Multi":
        scales = [0.25, 1, 4, 16]
        num_neurons = 100
    elif scale_type == "Single":
        scales = [1]
        num_neurons = 200
    return scales, num_neurons


def get_test_length(city, vel):
    if city == "Kitti":
        test_length = len(vel)
    else:
        test_length = min(len(vel), 500)  # suppress for testing
    return test_length


def generate_random_velocities(vel_profile, index, test_length):
    vel_min, vel_max, vel_var = vel_profile
    np.random.seed(index * vel_var)
    vel = np.random.uniform(vel_min, vel_max, test_length)
    return vel


def integrate_position(city, vel, ang_vel, q, test_length):
    posi_integ_log = [[0, 0] for _ in range(test_length)]
    tbar = tqdm(range(1, test_length), disable="GITHUB_ACTIONS" in os.environ)
    for i in tbar:
        q[2] += ang_vel[i]
        q[0] += vel[i] * np.cos(q[2])
        q[1] += vel[i] * np.sin(q[2])
        posi_integ_log[i] = list(q[0:2])
        descartes_vel = np.array([vel[i] * np.cos(q[2]), vel[i] * np.sin(q[2])])
        an_slam.inject(descartes_vel)
    return posi_integ_log


def construct_slam(
    excite_kernel_size=7,
    inhibit_kernel_size=5,
    net_size=80,
    local_inhibit_factor=0.16,
    global_inhibit_factor=6.51431074e-04,
    iteration=3,
    forget_ratio=0.95,
    scale=1,
    influence_func: Callable[[float], float] = None,
    excite_func: Callable[["AttractorNetwork"], np.ndarray] = None,
    inhibit_func: Callable[["AttractorNetwork"], np.ndarray] = None,
):
    def default_excite_func(Network: "AttractorNetwork") -> np.ndarray:
        local_excite = sp.signal.convolve2d(
            Network.activity,
            KernelFactory.create(excite_kernel_size, influence_func),
            mode="same",
            boundary="wrap",
        )
        return Network.activity + local_excite

    def default_inhibit_func(Network: "AttractorNetwork") -> np.ndarray:
        local_inhibit = local_inhibit_factor * sp.signal.convolve2d(
            Network.activity,
            KernelFactory.create(inhibit_kernel_size, influence_func),
            mode="same",
            boundary="wrap",
        )
        global_inhibit = Network.activity.sum() * global_inhibit_factor
        return Network.activity - (global_inhibit + local_inhibit)

    if influence_func is None:
        influence_func = lambda dist: math.exp(-dist)

    if excite_func is None:
        excite_func = default_excite_func

    if inhibit_func is None:
        inhibit_func = default_inhibit_func

    an = AttractorNetwork(
        net_size,
        excite_func,
        inhibit_func,
        iteration=iteration,
        forget_ratio=forget_ratio,
    )

    test_ob = An_Slam(an, scale=scale)
    return test_ob


def run_can_at_path(
    city,
    index,
    scale_type,
    traverse_info_file_part,
    vel_profile,
    dt,
    pathfile,
    an_slam=None,
    SNR=1e8,  # 80db
):
    if an_slam is None:
        an_slam = construct_slam()

    vel, ang_vel = load_traverse_info(traverse_info_file_part, index)

    scales, num_neurons = get_scales_and_neurons(scale_type)

    test_length = get_test_length(city, vel)

    if isinstance(vel_profile, list):
        vel = generate_random_velocities(vel_profile, index, test_length)

    np.random.seed(index)
    noise = np.random.normal(0, vel.max() / SNR, len(vel))
    vel += noise

    q = np.zeros(3)
    q_err = np.zeros(3)
    x_grid_expect, y_grid_expect = 0, 0

    posi_integ_log = integrate_position(city, vel, ang_vel, q, test_length)

    return posi_integ_log, an_slam


def singlePlot(data_list, label_list):
    fig = plt.figure()
    for data, label in zip(data_list, label_list):
        plt.plot(*zip(*data), label=label)
    # how legend
    plt.legend()
    return fig


def multi_plot(data):
    size_mag = 0.5

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

    plt.subplots_adjust(top=0.93)
    return fig


# %%

configs = load_dataset("SelectiveMultiScale")
City = "Kitti"
index = 0
dt = 1
scaleType = "Single"
traverseInfo_filePart = configs[City]["traverseInfo_file"]
vel_profile = configs[City]["vel_profile"]
veltype = parseVelType(vel_profile)
pathfile = f"./Results/{City}/CAN_Experiment_Output_{scaleType}/TestingTrackswith{veltype}_Path"

# higher net_size, higher accuracy
# which is odd, since the scale is defined at the cell level
# however, it is possible that the higher net_size, the more accurate the excite and inhibit kernel
# or, the wrap around is introduced other noises

SLAM_configs = {
    "excite_kernel_size": 9,
    "inhibit_kernel_size": 7,
    "net_size":200,
    "local_inhibit_factor": 0.16,
    "global_inhibit_factor": 6.51431074e-04,
    "iteration": 3,
    "forget_ratio": 0.95,
    "scale": 0.1,
    "influence_func": None,
    "excite_func": None,
    "inhibit_func": None,
}


an_slam = construct_slam(**SLAM_configs)

posi_integ_log, an_slam = run_can_at_path(
    City, index, scaleType, traverseInfo_filePart, vel_profile, dt, pathfile, an_slam
)

CAN_SLAM_Track_Baseline = an_slam.tragety_histroy
SLAM_Spike_histroy = an_slam.networks.activity_history

integrated_tracks = [posi_integ_log]
integrated_labels = ["posi_integ_log"]
Slam_tracks = [CAN_SLAM_Track_Baseline]
Slam_labels = ["CAN_SLAM_Track"]
SNRs=np.logspace(3, 0, 4)
for SNR in SNRs:
    an_slam = construct_slam(**SLAM_configs)
    integrated_track, an_slam = run_can_at_path(
        City,
        index,
        scaleType,
        traverseInfo_filePart,
        vel_profile,
        dt,
        pathfile,
        an_slam,
        SNR=SNR,
    )
    noisy_posi_integ_log = an_slam.tragety_histroy

    integrated_tracks.append(integrated_track)
    integrated_labels.append(f"posi_integ_log_{10*math.log10(SNR):.1f}db")
    Slam_tracks.append(noisy_posi_integ_log)
    Slam_labels.append(f"noisy_posi_integ_log_{10*math.log10(SNR):.1f}db")


# %%

# test_ob.activity
# data = CAN_SLAM_Track
# data2 = posi_integ_log

plt.close("all")
# ATE = np.linalg.norm(np.array(data) - np.array(data2)) / len(data)
fig = plt.figure()

for data, label in zip(integrated_tracks, integrated_labels):
    plt.plot(*zip(*data), '.-', label=label)
    
plt.legend()

# %%
# inte_ATEs_log=[ np.linalg.norm(data-posi_integ_log) for data in integrated_tracks]
inte_Errors_log = [
    np.linalg.norm(np.array(data) - np.array(posi_integ_log), axis=1)
    for data in integrated_tracks
]
SLAM_Errors_log = [
    np.linalg.norm(np.array(data) - np.array(posi_integ_log), axis=1)
    for data in Slam_tracks
]
# plt.plot(inte_ATEs_log[1], "o", label=integrated_labels[0])
# plt.plot(SLAM_ATEs_log[1], "-", label=Slam_labels[0])

for idx, (inte_ATE, SLAM_ATE) in enumerate(zip(inte_Errors_log, SLAM_Errors_log)):
    fig = plt.figure()
    plt.plot(inte_ATE,'o', label=integrated_labels[idx])
    plt.plot(SLAM_ATE,'-', label=Slam_labels[idx])
    plt.legend()

# %%

inte_ATEs=np.array(inte_Errors_log).mean(axis=1)
SLAM_ATEs=np.array(SLAM_Errors_log).mean(axis=1)

# plt.plot(inte_ATEs,'o', label=integrated_labels[0])
# plt.plot(SLAM_ATEs,'-', label=Slam_labels[0])

plt.loglog(SNRs,inte_ATEs[1:],'o', label=integrated_labels[0])
plt.loglog(SNRs,SLAM_ATEs[1:],'-', label=Slam_labels[0])
plt.legend()


configurations = {
    "City": City,
    "index": index,
    "dt": dt,
    "scaleType": scaleType,
    "vel_profile": vel_profile,
    "veltype": veltype,
}

# print(f"ATEs={list(ATEs)}")


# %%

# %%

# %%

plt.close("all")
# data=test_ob.tragety_histroy

# plt.gca().set_title(f"ATE={ATE}")
# fig.show()


sampled_SLAM_Spike_history = SLAM_Spike_histroy[::80]
# plt.close("all")


fig = multi_plot(sampled_SLAM_Spike_history)
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
