from pathlib import Path
from tqdm import tqdm

import yaml
import scienceplots
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np
import random
import math
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy import ndimage
import time
from os import listdir
import sys

import CAN

sys.path.append("./scripts")

plt.style.use(["science", "ieee"])
# plt.style.use(['science','no-latex'])




def load_traverse_info(traverseInfo_filePart, index):
    if "kitti" in traverseInfo_filePart:
        traverseInfo_file = f"{traverseInfo_filePart}{index:02d}.npy"
        vel, ang_vel = np.load(traverseInfo_file)
    else:
        traverseInfo_file = f"{traverseInfo_filePart}{index}.npz"
        traverse_info = np.load(traverseInfo_file, allow_pickle=True)
        vel, ang_vel = traverse_info['speeds'], traverse_info["angVel"]
    return vel, ang_vel

def parseVelType(vel_profile):
    # either a list or a string
    # if list, then it is [vel_min,vel_max,vel_var]
    # if string, then it mostly "GTpose" for kitty
    if type(vel_profile) == list:
        vel_min,vel_max,vel_var = vel_profile
        veltype = f"Speeds{vel_min}to{vel_max}var{vel_var}"
    else:
        veltype = vel_profile
    return veltype

    
"""Running All Paths in a City SingleScale and MultiScale"""


def runningAllPathsFromACity(City, scaleType, configs, run=False, plotting=False):
    # scaleType = Single or Multi;
    # City = Berlin or Japan or Brisbane or Newyork or Kitti;
    ATE = []

    def runCanAtPath(City, scaleType, traverseInfo_filePart, vel_profile, pathfile, index):
        vel, angVel = load_traverse_info(traverseInfo_filePart, index)

        if scaleType == "Multi":
            scales = [0.25, 1, 4, 16]
            numNeurons = 100
        elif scaleType == "Single":
            scales = [1]
            numNeurons = 200

        test_length = min(len(vel), 1000)  # supress for testing

        if type(vel_profile) == list:
                    # use a random seed to generate the velocities for reproducibility
            vel_min,vel_max,vel_var = vel_profile
            np.random.seed(index * vel_var)
            vel = np.random.uniform(vel_min, vel_max, test_length)
                    
        CAN.headDirectionAndPlaceNoWrapNet(
                    scales,
                    vel[:test_length],
                    angVel,
                    f"{pathfile}{index}.npy",
                    N=numNeurons,
                    printing=False,
                )
        print(f"finished {City}, id {index}")
        
    length = configs[City]["length"]
    traverseInfo_filePart = configs[City]["traverseInfo_file"]
    figrows, figcols = configs[City]["figshape"]
    vel_profile= configs[City]["vel_profile"]
    veltype = parseVelType(vel_profile)
    pathfile = f"./Results/{City}/CAN_Experiment_Output_{scaleType}/TestingTrackswith{veltype}_Path"
    savepath = f"./Results/{City}/TestingTrackswith{veltype}_{scaleType}scale.png"
    savepath2 = f"./Results/PaperFigures/5_{City}TestingTrackswith{veltype}_{scaleType}scale.pdf"

    if run == True:
        # for index in range(length):
        tbar = tqdm(range(length))
        tbar.set_description(f"Running {City} {scaleType}scale")
        for index in tbar:
            runCanAtPath(City, scaleType, traverseInfo_filePart, vel_profile, pathfile, index)
    if plotting == True:
        for index in range(length):
            # load ori planned vel and angVel
            vel, angVel = load_traverse_info(traverseInfo_filePart, index)
            if not Path(f"{pathfile}{index}.npy").exists():
                print(f"{City} {scaleType}scale havent run yet, running...")
                runCanAtPath(City, scaleType, traverseInfo_filePart, vel_profile, pathfile, index)
                
            x_grid, y_grid, x_integ, y_integ, x_integ_err, y_integ_err = np.load(
                f"{pathfile}{index}.npy"
            )
            # use fixed random vel instead of ori planned vel
            if type(vel_profile) == list:
                # use a random seed to generate the velocities for reproducibility
                vel_min,vel_max,vel_var = vel_profile
                np.random.seed(index * vel_var)
                vel = np.random.uniform(vel_min, vel_max, len(x_grid))
                
            dist = np.sum(vel)
            ATE.append(
                CAN.errorTwoCoordinateLists(x_integ, y_integ, x_grid, y_grid) / dist
            )

        orderedErrorIDs = np.argsort(ATE)
        print(f"scale type: {scaleType}")
        print(f"orderedErrorIDs: {orderedErrorIDs}, ATE: {ATE}")
        

        fig, ax = plt.subplots(figrows, figcols, figsize=(5, 5))
        fig.legend([f"{scaleType}scaleCAN", "Grid"])
        fig.tight_layout(pad=0.8)
        # fig.suptitle(f'{scaleType}scale Trajectory Tracking through {City} with CAN')
        axs = ax.ravel()
        for index in range(length):
            """load"""
            x_grid, y_grid, x_integ, y_integ, x_integ_err, y_integ_err = np.load(
                pathfile + f"{orderedErrorIDs[index]}.npy"
            )

            """color dictionary"""
            color = {"Multi": "m-", "Single": "b-"}

            """plot"""
            (l1,) = axs[index].plot(
                x_grid, y_grid, color[scaleType], label=f"{scaleType}scaleCAN"
            )
            (l2,) = axs[index].plot(x_integ, y_integ, "g--")
            axs[index].axis("equal")
            axs[index].tick_params(axis="x", labelsize=6)
            axs[index].tick_params(axis="y", labelsize=6)
            # axs[i].ticklabel_format(axis='both', style='sci', scilimits=(0,0))

        print(
            f"City: {City}, Scale Type: {scaleType}, mean = {np.average(ATE)}, std = {np.std(ATE)}"
        )
        # plt.subplots_adjust(bottom=0.1)
        plt.subplots_adjust(top=0.93)
        fig.legend(
            (l1, l2),
            (f"{scaleType}scale CAN", "Ground Truth"),
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.0),
            prop={"size": 10},
        )
        for p in savepath, savepath2:
            p = Path(p)
            if not p.parent.exists():
                p.parent.mkdir(parents=True)
        plt.savefig(savepath)
        plt.savefig(savepath2)


""" Multi versus Single over Large Velocity Range"""


def multiVsSingle(City, index, configs,desiredTestLength=500, run=False, plotting=False):
    traverseInfo_filePart=configs[City]["traverseInfo_file"]
    vel, angVel = load_traverse_info(traverseInfo_filePart, index)
    
    test_length = min(len(vel), desiredTestLength)  # supress for testing
    filepath = f"./Results/{City}/MultivsSingleErrors_Path{index}.npy"
    
    savepath= f"./Results/{City}/MultivsSingleErrors_Path{index}.png"
    savepath2=f"./Results/PaperFigures/3_MultivsSingleErrors_Path{index}.pdf"
    
    if run == True:
        errors = []
        # what the fuck and why ?
        for maxSpeed in range(1, 21):
            vel = np.random.uniform(0, maxSpeed, test_length)
            
            true_x, true_y = CAN.pathIntegration(vel, angVel)

            scales = [1]
            singleError = CAN.headDirectionAndPlaceNoWrapNet(
                scales, vel, angVel, None, N=200, returnTypes="Error", printing=False
            )

            scales = [0.25, 1, 4, 16]
            multipleError = CAN.headDirectionAndPlaceNoWrapNet(
                scales, vel, angVel, None, returnTypes="Error", printing=False
            )

            errors.append([singleError, multipleError])

        np.save(filepath, errors)
    if plotting == True:
        if not Path(filepath).exists():
            multiVsSingle(City, index, configs,desiredTestLength=desiredTestLength, run=True, plotting=False)
        # plt.figure(figsize=(2.7,2))
        fig, ax = plt.subplots(1, 1, figsize=(2.8, 2.2))
        # fig.tight_layout(pad=3)
        singleErrors, multipleErrors = zip(*np.load(filepath))
        ax.plot(np.arange(20), singleErrors, "b")
        ax.plot(np.arange(20), multipleErrors, "m")
        # plt.legend(['Single-scale', 'Multiscale'])
        ax.set_xlabel("\# Trajectory")
        ax.set_ylabel("ATE [m] ")
        # plt.title('Network Perfomance over Large Velocity Ranges')
        # plt.tight_layout()
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        # plt.subplots_adjust(left=0.1, right=0.75)
        fig.legend(
            ("Single scale", "Multiscale"),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=2,
            prop={"size": 8},
        )
        for p in savepath, savepath2:
            p = Path(p)
            if not p.parent.exists():
                p.parent.mkdir(parents=True)
        plt.savefig(savepath)
        plt.savefig(savepath2)



"""Cumalitive error Distribution Single vs Multi"""


# def CumalativeError_SinglevsMulti(singlePath, multiPath, run=False, plotting=False):
def CumalativeError_SinglevsMulti(City, index, configs, run=False, plotting=False):
    singlePath=f"./Results/{City}/CumalitiveError_Path{index}_SingleScale.npy"
    multiPath=f"./Results/{City}/CumalativeError_Path{index}_MultiScale.npy"
    
    traverseInfo_filePart=configs[City]["traverseInfo_file"]
    vel, angVel = load_traverse_info(traverseInfo_filePart, index)
    
    test_length = min(len(vel), 1000)  # supress for testing
    
    savepath= f"./Results/{City}/CumalitiveError_SinglevsMulti_Path{index}.png"
    savepath2=f"./Results/PaperFigures/6_CumalitiveError_SinglevsMulti_Path{index}.pdf"

    # TODO Compatative to kitti, the vel is not uniformly distributed
    vel = np.random.uniform(0, 20, test_length)

    if run == True:
        scales = [1]
        CAN.headDirectionAndPlaceNoWrapNet(
            scales, vel, angVel, singlePath, returnTypes="posInteg+CAN", printing=False
        )
        scales = [0.25, 1, 4, 16]
        CAN.headDirectionAndPlaceNoWrapNet(
            scales, vel, angVel, multiPath, returnTypes="posInteg+CAN", printing=False
        )
    if plotting == True:
        if not Path(singlePath).exists():
            CumalativeError_SinglevsMulti(City, index, configs, run=True, plotting=False)
        x_gridM, y_gridM, x_integM, y_integM, x_integ_err, y_integ_err = np.load(
            multiPath
        )
        x_gridS, y_gridS, x_integS, y_integS, x_integ_err, y_integ_err = np.load(
            singlePath
        )
        multipleError = CAN.errorTwoCoordinateLists(
            x_integM, y_integM, x_gridM, y_gridM, errDistri=True
        )
        singleError = CAN.errorTwoCoordinateLists(
            np.concatenate([x_integS, [x_integS[-1]]]), 
            np.concatenate([y_integS, [y_integS[-1]]]), 
            x_gridS, y_gridS, errDistri=True
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.6, 2.2))
        # fig.legend(['MultiscaleCAN', 'Grid'])
        fig.tight_layout()
        # fig.suptitle('ATE Error Over Time',y=1.07)
        ax1.bar(np.arange(len(vel)), singleError[:len(vel)], color="royalblue", width=1)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("ATE [m]")
        ax2.bar(np.arange(len(vel)), multipleError[:len(vel)], color="mediumorchid", width=1)
        ax2.set_xlabel("Time [s]")
        # plt.subplots_adjust(top=0.9)
        ax1.tick_params(axis="x", labelsize=6)
        ax1.tick_params(axis="y", labelsize=6)
        ax2.tick_params(axis="x", labelsize=6)
        ax2.tick_params(axis="y", labelsize=6)

        fig.legend(
            ("Single scale", "Multiscale"),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=2,
            columnspacing=7,
            prop={"size": 8},
        )
        for p in savepath, savepath2:
            p = Path(p)
            if not p.parent.exists():
                p.parent.mkdir(parents=True)
        plt.savefig(savepath)
        plt.savefig(savepath2)
    pass



"""Local Error segments Berlin"""


# def plotMultiplePathsErrorDistribution():
#     length = 18
#     pathfileSingle = f"./Results/Berlin/CAN_Experiment_Output_Single/TestingTrackswithSpeeds0to20_Path"
#     pathfileMulti = f"./Results/Berlin/CAN_Experiment_Output_Multi/TestingTrackswithSpeeds0to20_Path"
def plotMultiplePathsErrorDistribution(City, configs, run=False, plotting=False):
    length = configs[City]["length"]
    
    vel_profile= configs[City]["vel_profile"]
    veltype = parseVelType(vel_profile)
        
    pathfileSingle = f"./Results/{City}/CAN_Experiment_Output_Single/TestingTrackswith{veltype}_Path"
    pathfileMulti = f"./Results/{City}/CAN_Experiment_Output_Multi/TestingTrackswith{veltype}_Path"
    
    savepath = f"./Results/{City}/LocalSegmentError_AllPaths_SinglevsMulti.png"
    savepath2 = f"./Results/PaperFigures/7_LocalSegmentError_AllPaths_SinglevsMulti.pdf"

    fig, axs = plt.subplots(1, 1, figsize=(2.7, 1.9))
    errorSingle, erroMulti = [], []
    for i in range(length):
        x_grid, y_grid, x_integ, y_integ, x_integ_err, y_integ_err = np.load(
            pathfileSingle + f"{i}.npy"
        )
        errorSingle.append(
            CAN.errorTwoCoordinateLists(x_integ, y_integ, x_grid, y_grid)
        )

    for i in range(length):
        x_grid, y_grid, x_integ, y_integ, x_integ_err, y_integ_err = np.load(
            pathfileMulti + f"{i}.npy"
        )
        erroMulti.append(CAN.errorTwoCoordinateLists(x_integ, y_integ, x_grid, y_grid))

    # plt.subplots_adjust(bottom=0.2)
    axs.bar(np.arange(length), errorSingle, color="royalblue")
    axs.bar(np.arange(length), erroMulti, color="mediumorchid")
    # axs.legend(['Single-scale', 'Multiscale'],ncol=2,loc='best')
    axs.set_xlabel(f"\# {City} Trajectory", y=0)
    axs.set_ylabel("ATE [m]")
    # axs.set_ylim([0,14000])
    axs.tick_params(axis="x", labelsize=7)
    axs.tick_params(axis="y", labelsize=7)
    # axs.set_title('ATE within 18 Trajectories through Berlin')
    fig.legend(
        ("Single scale", "Multiscale"),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        prop={"size": 7},
    )
    
    for p in savepath, savepath2:
        p = Path(p)
        if not p.parent.exists():
            p.parent.mkdir(parents=True)
            
    plt.savefig(savepath)
    plt.savefig(savepath2)



"""Running All Paths in a City SingleScale and MultiScale"""
configs = yaml.load(open("Datasets/profile.yml", "r"), Loader=yaml.FullLoader)
configs = configs["SelectiveMultiScale"]



def exp_on_city(City='Newyork', index=0, configs=configs,run=False, plotting=False):
    scaleType = "Single"
    runningAllPathsFromACity(City, scaleType, configs, run=run, plotting=True)
    print("")
    scaleType = "Multi"
    runningAllPathsFromACity(City, scaleType, configs, run=run, plotting=True)

    """ Multi versus Single over Large Velocity Range"""
    multiVsSingle(City, index,configs, 500, run=run, plotting=True)
    CumalativeError_SinglevsMulti(City, index, configs, run=run, plotting=True)
    plotMultiplePathsErrorDistribution(City, configs, run=run, plotting=True)

for City in ['Newyork', 'Brisbane', 'Berlin', 'Japan', 'Kitti']:
    exp_on_city(City=City, index=0, configs=configs,run=True, plotting=True)
# exp_on_city(City='Kitti', index=0, configs=configs,run=True, plotting=False)


""" Kitti GT Poses"""


def data_processing(index):
    poses = pd.read_csv(
        f"./Datasets/kittiOdometryPoses/{index}.txt", delimiter=" ", header=None
    )
    gt = np.zeros((len(poses), 3, 4))
    for i in range(len(poses)):
        gt[i] = np.array(poses.iloc[i]).reshape((3, 4))

    # extracting velocities from poses
    sparse_gt = gt
    data_x = sparse_gt[:, :, 3][:, 0]  # [:200]
    data_y = sparse_gt[:, :, 3][:, 2]  # [:200]
    delta1, delta2 = [], []
    for i in range(1, len(data_x)):
        x0 = data_x[i - 2]
        x1 = data_x[i - 1]
        x2 = data_x[i]
        y0 = data_y[i - 2]
        y1 = data_y[i - 1]
        y2 = data_y[i]

        delta1.append(np.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2)))  # translation
        delta2.append((math.atan2(y2 - y1, x2 - x1)) - (math.atan2(y1 - y0, x1 - x0)))

    np.save(
        f"./Datasets/kittiVelocities/kittiVels_{index}.npy", np.array([delta1, delta2])
    )


# for i in range(10):
#     data_processing(f'0{i}')
# data_processing('10')

# TODO Merge with Cities
def plotKittiGT_singlevsMulti(index):
    multiPath = f"./Results/Kitti/CAN_Experiment_Output_Multi/TestingTrackswithGTpose_path{index}.npy"
    singlePath = f"./Results/Kitti/CAN_Experiment_Output_Single/TestingTrackswithGTpose_path{index}.npy"
    x_gridM, y_gridM, x_integM, y_integM, x_integ_err, y_integ_err = np.load(multiPath)
    x_gridS, y_gridS, x_integS, y_integS, x_integ_err, y_integ_err = np.load(singlePath)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.6, 2.2))
    # fig.suptitle('Multiscale vs. Single Scale Kitti Odometry Path')
    # plt.subplots_adjust(bottom=0.2)
    (l2,) = ax1.plot(x_gridM, y_gridM, "m-")
    (l1,) = ax1.plot(x_integM, y_integM, "g--")
    ax1.axis("equal")

    (l3,) = ax2.plot(x_gridS, y_gridS, "b-")
    (l4,) = ax2.plot(x_integS, y_integS, "g--")
    ax2.axis("equal")

    ax1.tick_params(axis="x", labelsize=6)
    ax1.tick_params(axis="y", labelsize=6)

    ax2.tick_params(axis="x", labelsize=6)
    ax2.tick_params(axis="y", labelsize=6)

    fig.legend(
        (l2, l3, l1),
        ("Multiscale", "Single scale", "Ground Truth"),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.03),
        prop={"size": 8},
    )
    plt.savefig(f"./Results/Kitti/KittiSinglevsMulti_{index}.png")
    plt.savefig(f"./Results/PaperFigures/2_KittiSinglevsMulti_{index}.pdf")


# plotKittiGT_singlevsMulti(0)
City = "Kitti"
scaleType = "Single"
# runningAllPathsFromACity(City, scaleType, configs, run=False, plotting=True)

# # runningAllPathsFromACity('Japan', scaleType, run=False, plotting=True)
# # runningAllPathsFromACity(City, scaleType, configs, run=False, plotting=True)
# # runningAllPathsFromACity('Brisbane', scaleType,run=False, plotting=True)
# # runningAllPathsFromACity('Berlin', scaleType, run=False, plotting=True)
# # runningAllPathsFromKittiGT(11, scaleType, run=False, plotting=True)
# print("")
# scaleType = "Multi"
# runningAllPathsFromACity(City, scaleType, configs, run=False, plotting=True)
# # runningAllPathsFromACity('Japan', scaleType, run=False, plotting=True)
# # runningAllPathsFromACity(City, scaleType, configs, run=True, plotting=True)
# # runningAllPathsFromACity('Brisbane', scaleType,run=False, plotting=True)
# # runningAllPathsFromACity('Berlin', scaleType, run=False, plotting=True)
# # runningAllPathsFromKittiGT(11, scaleType, run=False, plotting=True)

# """ Multi versus Single over Large Velocity Range"""

# # index = 0
# # filepath = f"./Results/Berlin/MultivsSingleErrors_Path{index}.npy"
# # mutliVs_single(filepath, index, 500, run=False, plotting=True)
# multiVsSingle(City, 0,configs, 500, run=False, plotting=True)

# # singlePath = "./Results/Berlin/CumalativeError_Path1_SingleScale.npy"
# # multiPath = "./Results/Berlin/CumalativeError_Path1_MultiScale.npy"
# # CumalativeError_SinglevsMulti(singlePath, multiPath, run=False, plotting=True)
# CumalativeError_SinglevsMulti(City, 0, configs, run=False, plotting=True)
# plotMultiplePathsErrorDistribution(City, configs, run=False, plotting=True)


"""Random Data for Ablation"""


def generatinScales(multiplier, length):
    middle = (length - 1) // 2
    start = 1 / (multiplier**middle)
    scales = np.zeros(length)
    scales[0] = start
    for i in range(1, length):
        scales[i] = scales[i - 1] * multiplier
    return scales


def scaleAblation(
    scaleRatios, numScales, randomSeedVariation=5, run=False, plotting=False
):
    # savePath1=f'./Results/RandomData/Ablation_Experiment_Output_LargeVelRange/AblationOfScalesErrors.npy'
    # savePath2=f'./Results/RandomData/Ablation_Experiment_Output_LargeVelRange/AblationOfScalesDurations.npy'
    # pathfile=f'./Results/RandomData/Ablation_Experiment_Output_LargeVelRange/TestingRandomInputs_'
    # plotPath=f'./Results/RandomData/AblationofScales_ErrorsandDurations_LargeVelRange.png'

    # savePath1=f'./Results/RandomData/Ablation_Experiment_Output_SmallerVelRange/AblationOfScalesErrors.npy'
    # savePath2=f'./Results/RandomData/Ablation_Experiment_Output_SmallerVelRange/AblationOfScalesDurations.npy'
    # pathfile=f'./Results/RandomData/Ablation_Experiment_Output_SmallerVelRange/TestingRandomInputs_'
    # plotPath=f'./Results/RandomData/AblationofScales_ErrorsandDurations_SmallerVelRange.png'

    savePath1 = f"./Results/RandomData/Ablation_Experiment_Output_FineGrain/AblationOfScalesErrors_smallerRange2.npy"
    savePath2 = f"./Results/RandomData/Ablation_Experiment_Output_FineGrain/AblationOfScalesDurations_smallerRange2.npy"
    pathfile = f"./Results/RandomData/Ablation_Experiment_Output_FineGrain/TestingRandomInputs2_"
    plotPath = f"./Results/RandomData/Ablation_Experiment_Output_FineGrain/AblationofScales_ErrorsandDurations_FineGrain_SmallerVelRange2.png"

    if run == True:
        errors = np.zeros((len(scaleRatios), len(numScales)))
        durations = np.zeros((len(scaleRatios), len(numScales)))
        for i, ratio in enumerate(scaleRatios):
            for j, length in enumerate(numScales):
                test_length = 50
                np.random.seed(randomSeedVariation)
                vel = np.random.uniform(0, 2, test_length)
                angVel = np.random.uniform(0, np.pi / 6, test_length)
                scales = generatinScales(ratio, length)

                # t=time.time()
                # errors[i,j]=CAN.headDirectionAndPlaceNoWrapNet(scales, vel, angVel,None, printing=False, returnTypes='Error')
                # durations[i,j]=(time.time()-t)

                x_integ, y_integ, x_grid, y_grid = CAN.headDirectionAndPlaceNoWrapNet(
                    scales, vel, angVel, None, returnTypes="posInteg+CAN"
                )
                vel_CANoutput, angVel_CANoutput = CAN.positionToVel2D(x_grid, y_grid)
                vel_GT, angVel_GT = CAN.positionToVel2D(x_integ, y_integ)
                errors[i, j] = np.sum(abs(vel_CANoutput - vel_GT))

                print(f"Finished ratio {ratio} and length {length}")

        np.save(savePath1, errors)
        np.save(savePath2, durations)

    if plotting == True:
        errors = np.load(savePath1)
        durations = np.load(savePath2)

        fig, ax0 = plt.subplots(figsize=(5, 4), ncols=1)

        pos = ax0.imshow(errors)
        plt.colorbar(pos, ax=ax0)
        ax0.set_xlabel("Number of Scales")
        ax0.set_ylabel("Scale Ratio")
        ax0.set_title("Errors [ATE]")
        ax0.set_yticks(np.arange(len(scaleRatios)), [a for a in scaleRatios])
        ax0.set_xticks(np.arange(len(numScales)), [a for a in numScales], rotation=90)

        # pos1= ax1.imshow(durations,cmap='jet')
        # plt.colorbar(pos1,ax=ax1)
        # ax1.set_xlabel('Number of Scales')
        # ax1.set_ylabel('Scales Ratio')
        # ax1.set_title('Duration [secs]')
        # ax1.set_yticks(np.arange(len(scaleRatios)),[a for a in scaleRatios])
        # ax1.set_xticks(np.arange(len(numScales)), [a for a in numScales],rotation=90)

        plt.savefig(plotPath)


# scaleRatios,numScales=[1, 1.5, 2, 2.5, 3, 3.5, 4],[1,2,3,4,5]
# scaleAblation(scaleRatios,numScales, run=True, plotting=True)


"""Response to Velocity Spikes"""


def resposneToVelSpikes(randomSeedVariation=5, run=False, plotting=False):
    savePath = f"./Results/RandomData/VelSpikes_Experiment_Output/Integrated+CAN_velsWithSpikes.npy"
    savePath2 = (
        f"./Results/RandomData/VelSpikes_Experiment_Output/CANoutput_velsWithSpikes.npy"
    )
    plotPath = f"./Results/RandomData/MCAN_path_kidnappedAgent_1uni_Tun0.png"
    if run == True:
        test_length = 1000
        np.random.seed(randomSeedVariation)
        vel = np.random.uniform(0, 1, test_length)
        # for i in range(20,test_length,test_length//3):
        #     vel[i-3]=5
        #     vel[i-2]=10
        #     vel[i-1]=15
        #     vel[i]=20
        vel[test_length - 10 :] = np.random.uniform(10, 12, 10)
        angVel = np.random.uniform(-np.pi / 6, np.pi / 6, test_length)
        scales = [0.25, 1, 4, 16]

        x_integ, y_integ, x_grid, y_grid = CAN.headDirectionAndPlaceNoWrapNet(
            scales, vel, angVel, savePath2, returnTypes="posInteg+CAN"
        )
        vel_CANoutput, angVel_CANoutput = CAN.positionToVel2D(x_grid, y_grid)

        # np.save(savePath,np.array([vel,vel_CANoutput]))

    if plotting == "Vel":
        vel, vel_CANoutput = np.load(savePath, allow_pickle=True)
        fig, ax0 = plt.subplots(figsize=(4, 4), ncols=1)
        l2 = ax0.plot(vel, "g.-")
        l3 = ax0.plot(vel_CANoutput, "m.-")
        ax0.legend(("Ground Truth", "Multiscale CAN"))
        ax0.set_title("MCAN integration vs. Ground Truth Velocity Profile ")
        ax0.set_ylabel("Velocity")
        ax0.set_xlabel("Time")
        plt.savefig(plotPath)
        # plt.show()

    if plotting == "Position":
        x_grid, y_grid, x_integ, y_integ, x_integ_err, y_integ_err = np.load(savePath2)
        fig, ax0 = plt.subplots(figsize=(4, 4), ncols=1)
        l2 = ax0.plot(x_integ, y_integ, "g.-")
        l3 = ax0.plot(x_grid, y_grid, "m.-")
        ax0.legend(("Ground Truth", "Multiscale CAN"))
        ax0.set_title("MCAN Path for Velocity with Spikes ")
        ax0.set_ylabel("y[m]")
        ax0.set_xlabel("x[m]")
        plt.savefig(plotPath)


resposneToVelSpikes(randomSeedVariation=7, run=True, plotting="Position")
