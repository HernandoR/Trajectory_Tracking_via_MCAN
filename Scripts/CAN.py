import math
import os
from pathlib import Path
import numpy as np 
import time
import random  
import pandas as pd
from scipy import signal
from scipy import ndimage
from tqdm import tqdm

import scienceplots
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from matplotlib.colors import ListedColormap

from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import Axes3D 

'''CAN networks'''
from AttractorNetwork import attractorNetwork1D, attractorNetwork2D,AttractorNetwork, activityDecoding, activityDecodingAngle

import numba

'''Multiscale CAN Helper Functions'''
def positionToVel2D(data_x,data_y):
    vel,angVel=[],[]
    for i in range(0,len(data_x)):
        x0=data_x[i-2]
        x1=data_x[i-1]
        x2=data_x[i]
        y0=data_y[i-2]
        y1=data_y[i-1]
        y2=data_y[i]

        vel.append(np.sqrt(((x2-x1)**2)+((y2-y1)**2))) #translation
        angVel.append((math.atan2(y2-y1,x2-x1)) - (math.atan2(y1-y0,x1-x0))) 
    
    return np.array(vel),np.array(angVel)


def pathIntegration(speed, angVel):
    q=[0,0,0]
    x_integ,y_integ=[],[]
    for i in range(len(speed)):
        q[0],q[1]=q[0]+speed[i]*np.cos(q[2]), q[1]+speed[i]*np.sin(q[2])
        q[2]+=angVel[i]
        x_integ.append(round(q[0],4))
        y_integ.append(round(q[1],4))

    return x_integ, y_integ


def errorTwoCoordinateLists(x_pi, y_pi, x2, y2, errDistri=False):
    '''RMSE error'''
    err=[]
    x_err_sum,y_err_sum=0,0
    for i in range(len(x2)):
        x_err_sum+=(x_pi[i]-x2[i])**2
        y_err_sum+=(y_pi[i]-y2[i])**2
        if errDistri== True:
            err.append(np.abs(x_pi[i]-x2[i])+ np.abs(y_pi[i]-y2[i]))

    x_error=np.sqrt(x_err_sum/len(x2))
    y_error=np.sqrt(y_err_sum/len(y2))

    if errDistri==True:
        return np.array(err)#np.cumsum(np.array(err))
    else:
        return (x_error+y_error)


def scale_selection(input,scales, swap_val=1):
    if len(scales)==1:
        scale_idx=0
    else: 

        if input<=scales[0]*swap_val:
            scale_idx=0

        for i in range(len(scales)-1):
            if input>scales[i]*swap_val and input<=scales[i+1]*swap_val:
                scale_idx=i+1
        
        # if input>scales[-2]*swap_val:
        #     scale_idx=len(scales)-1
        
        if input>scales[-1]*swap_val:
            scale_idx=len(scales)-1

    return scale_idx

# test if global variable is better, otherwise classify it.


def headDirection(theta_weights, angVel, init_angle,theata_called_iters):
    N=360
    # num_links,excite,activity_mag,inhibit_scale, iterations=16, 17, 2.16818183,  0.0281834545, 2
    # num_links,excite,activity_mag,inhibit_scale, iterations=16, 17, 2.16818183,  0.0381834545, 2
    num_links,excite,activity_mag,inhibit_scale, iterations=13,4,2.70983783e+00,4.84668851e-02,2
    net=attractorNetwork1D(N,num_links,excite, activity_mag,inhibit_scale)
    
    if theata_called_iters==0:
        theta_weights[net.activation(init_angle)]=net.full_weights(num_links)
        theata_called_iters+=1

    for j in range(iterations):
        theta_weights=net.update_weights_dynamics(theta_weights,angVel)
        theta_weights[theta_weights<0]=0
    
    
    return theta_weights,theata_called_iters

def hierarchicalNetwork2DGridNowrapNet(
    prev_weights, net:AttractorNetwork,N, vel, direction, iterations, wrap_iterations, x_grid_expect, y_grid_expect,scales):
    '''
    Update the selected scale of the network weights based on the given velocity and scales.
    Args:
        prev_weights (list): List of previous weights for each scale.
        net (object): Network object.
        N (int): Number of neurons in the network.
        vel (float): Velocity of the object.
        direction (float): Direction of the object.
        iterations (int): Number of iterations to update the weights.
        wrap_iterations (int): Number of iterations to wrap the weights.
        x_grid_expect (float): Expected x-coordinate of the grid.
        y_grid_expect (float): Expected y-coordinate of the grid.
        scales (list): List of scales for the network.z

    Returns:
        tuple: A tuple containing the updated previous weights, wrap, x_grid_expect, and y_grid_expect.
    '''
    # Select scale and initilise wrap storage
    delta = [(vel/scales[i]) for i in range(len(scales))]
    chosen_scale_idx=scale_selection(vel,scales)
    # print(vel, scales, cs_idx)
    wrap_rows=np.zeros((len(scales)))
    wrap_cols=np.zeros((len(scales)))
    prev_weight=prev_weights[chosen_scale_idx]

    #Update selected scale
    for i in range(iterations):
        prev_weights[chosen_scale_idx][:], wrap_rows_cs, wrap_cols_cs= net.update_weights_dynamics(
            prev_weights[chosen_scale_idx][:],direction, delta[chosen_scale_idx])
        prev_weights[chosen_scale_idx][prev_weights[chosen_scale_idx][:]<0]=0
        x_grid_expect+=wrap_cols_cs*net.N[0]*scales[chosen_scale_idx]
        y_grid_expect+=wrap_rows_cs*net.N[1]*scales[chosen_scale_idx]
    wrap=0   
    return prev_weights, wrap, x_grid_expect, y_grid_expect

def headDirectionAndPlaceNoWrapNet(scales, vel, angVel,savePath: str|None, printing=False, N=100, returnTypes=None, genome=None):
    if savePath != None:
        savePath=Path(savePath)
        if not savePath.parent.exists():
            savePath.parent.mkdir(parents=True)

    if genome is not None: 
        num_links=int(genome[0]) #int
        excite=int(genome[1]) #int
        activity_mag=genome[2] #uni
        inhibit_scale=genome[3] #uni
        iterations=int(genome[4])
        wrap_iterations=int(genome[5])
    
    else:
        num_links = 10
        excite = 2
        activity_mag = 1.10262708e-01
        inhibit_scale = 6.51431074e-04
        iterations = 3
        wrap_iterations = 2
        # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=11,8,5.09182735e-01,2.78709739e-04,5,2
    network=attractorNetwork2D([N,N],num_links,excite, activity_mag,inhibit_scale)
    
    TIMESTEP_LEN=len(vel)
    
    '''__________________________Storage and initilisation parameters______________________________'''
    theta_weights=np.zeros(360)
    theata_called_iters=0
    # wrap_counter=[0,0,0,0,0,0]
    q=np.zeros(3)
    q_err=np.zeros(3)
    # grid_expect=np.zeros(2)
    x_grid_expect, y_grid_expect = 0,0
    
    posi_integ_log=np.zeros([TIMESTEP_LEN,2])
    grid_log=np.zeros([TIMESTEP_LEN,2])
    integ_err_log=np.zeros([TIMESTEP_LEN,2])
    
    '''__________________________Initilising scales in the center and at the edge_____________________________'''
    init_weights=[np.zeros((N,N)) for _ in range(len(scales))]
    for n in range(len(scales)):
        for _ in range(iterations):
            init_weights[n]=network.excitations(0,0)
            init_weights[n]=network.update_weights_dynamics_row_col(init_weights[n][:], 0, 0)
            init_weights[n][init_weights[n][:]<0]=0
            
    '''_______________________________Iterating through simulation Timesteps_______________________________'''
    tbar = tqdm(range(1,TIMESTEP_LEN), disable='GITHUB_ACTIONS' in os.environ)
    for i in tbar:
    # for i in range(1,TIMESTEP_LEN):
        # q+=np.array([vel[i]*np.cos(angVel[i]), vel[i]*np.sin(angVel[i]), angVel[i]])
        q[2]+=angVel[i]
        q[0],q[1]=q[0]+vel[i]*np.cos(q[2]), q[1]+vel[i]*np.sin(q[2])
        
        posi_integ_log[i]=q[0:2]
        '''Mutliscale CAN update'''
        N_dir=360
        theta_weights,theata_called_iters=headDirection(theta_weights, np.rad2deg(angVel[i]), 0,theata_called_iters)
        
        direction=activityDecodingAngle(theta_weights,5,N_dir)
        init_weights, wrap, x_grid_expect, y_grid_expect= hierarchicalNetwork2DGridNowrapNet(init_weights, network, N, vel[i], direction, iterations,wrap_iterations, x_grid_expect, y_grid_expect, scales)
        
        
        '''1D method for decoding'''

        def activityDecode1D(scale,weights,radius=5,N=100):
            return activityDecoding(weights,radius,N)*scale
        
        # TODO, Bad Coding
        def activityDecode2D(scale,weights,radius=5,axis=0,N=100):
            maxSliceIdx=np.argmax(np.max(weights, axis=axis))
            
            # weightSlice=weights.take(maxSliceIdx, axis=axis)
            if axis==0:
                weightSlice=weights[:,maxSliceIdx]
            elif axis==1:
                weightSlice=weights[maxSliceIdx,:]
            return activityDecoding(weightSlice,radius,N)*scale
            
        x_multiscale_grid,y_multiscale_grid=0,0
        for idx, scale in enumerate(scales):
            x_multiscale_grid+=activityDecode2D(scale,init_weights[idx],radius=5,axis=1,N=N)
            y_multiscale_grid+=activityDecode2D(scale,init_weights[idx],radius=5,axis=0,N=N)
        grid_log[i]=np.array([x_multiscale_grid+x_grid_expect, y_multiscale_grid+y_grid_expect])
        
        '''Error integrated path'''
        q_err[2]+=angVel[i]
        q_err[0],q_err[1]=q_err[0]+vel[i]*np.cos(np.deg2rad(direction)), q_err[1]+vel[i]*np.sin(np.deg2rad(direction))
        integ_err_log[i]=q_err[0:2]
        
        if printing==True:
            print(f'dir: {np.rad2deg(q[2])}, {direction}')
            print(f'vel: {vel[i]}')
            print(f'expected: {x_grid_expect}, {y_grid_expect}')
            print(f'integ: x,y: {posi_integ_log[i]}')
            print(f'CAN: x,y: {grid_log[i]}')
            print('')
            
    x_grid, y_grid=grid_log[:,0], grid_log[:,1]
    x_integ, y_integ=posi_integ_log[:,0], posi_integ_log[:,1]
    x_integ_err, y_integ_err=integ_err_log[:,0], integ_err_log[:,1]
    if savePath != None:
        savePath=Path(savePath)
        np.save(savePath, np.array([x_grid, y_grid, x_integ, y_integ, x_integ_err, y_integ_err]))
        
    
    # print(f'CAN error: {errorTwoCoordinateLists(x_integ,y_integ, x_grid, y_grid)}')
    # if returnTypes=='Error':
    #     return errorTwoCoordinateLists(x_integ,y_integ, x_grid, y_grid)
    # elif returnTypes=='PlotShow':
    #     plt.plot(x_integ, y_integ, 'g.')
    #     # plt.plot(x_integ_err, y_integ_err, 'y.')····
    #     plt.plot(x_grid, y_grid, 'b.')
    #     plt.axis('equal')
    #     plt.title('Test Environment 2D space')
    #     plt.legend(('Path Integration without Error','Multiscale Grid Decoding'))
    #     plt.show()
    # elif returnTypes=='posInteg+CAN':
    return x_integ,y_integ, x_grid, y_grid
        


def headDirectionAndPlaceNoWrapNetAnimate(scales, test_length, vel, angVel,savePath, plot=False, printing=True, N=100):
    global theata_called_iters,theta_weights, init_weights, q, wrap_counter, current_i, x_grid_expect, y_grid_expect 

    num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=10,2,1.10262708e-01,6.51431074e-04,3,2 #with decimals 200 iters fitness -395 modified
    num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=10,10,1,0.0008,2,1
    network=attractorNetwork2D([N,N],num_links,excite, activity_mag,inhibit_scale)

    

    '''__________________________Storage and initilisation parameters______________________________'''
    # scales=[0.25,1,4,16]
    theta_weights=np.zeros(360)
    theata_called_iters=0
    # start_x, start_y=(50*scales[3])+(50*scales[4])+(50*scales[5]),(50*scales[3])+(50*scales[4])+(50*scales[5])
    wrap_counter=[0,0,0,0,0,0]
    x_grid, y_grid=[], []
    x_grid_expect, y_grid_expect =0,0
    x_integ, y_integ=[],[]
    q=[0,0,0]
    x_integ_err, y_integ_err=[],[]
    q_err=[0,0,0]

    '''__________________________Initilising scales in the center and at the edge_____________________________'''
    init_weights=[np.zeros((N,N)) for _ in range(len(scales))]
    for n in range(len(scales)):
        for m in range(iterations):
            init_weights[n]=network.excitations(0,0)
            init_weights[n]=network.update_weights_dynamics_row_col(init_weights[n][:], 0, 0)
            init_weights[n][init_weights[n][:]<0]=0
    

    '''_______________________________Iterating through simulation velocities_______________________________'''
    fig, axs = plt.subplots(1,4,figsize=(5, 3),dpi=300) 
    def animate(i):  
        global theata_called_iters,theta_weights, init_weights, q, wrap_counter, current_i, x_grid_expect, y_grid_expect 

        '''Path integration'''
        q[2]+=angVel[i]
        q[0],q[1]=q[0]+vel[i]*np.cos(q[2]), q[1]+vel[i]*np.sin(q[2])
        x_integ.append(q[0])
        y_integ.append(q[1])


        '''Mutliscale CAN update'''
        N_dir=360
        theta_weights,theata_called_iters=headDirection(theta_weights, np.rad2deg(angVel[i]), 0,theata_called_iters)
        direction=activityDecodingAngle(theta_weights,5,N_dir)
        init_weights, wrap, x_grid_expect, y_grid_expect= hierarchicalNetwork2DGridNowrapNet(init_weights, network, N, vel[i], direction, iterations,wrap_iterations, x_grid_expect, y_grid_expect, scales)

        '''1D method for decoding'''
        maxXPerScale, maxYPerScale = np.array([np.argmax(np.max(init_weights[m], axis=1)) for m in range(len(scales))]), np.array([np.argmax(np.max(init_weights[m], axis=0)) for m in range(len(scales))])
        decodedXPerScale=[activityDecoding(init_weights[m][maxXPerScale[m], :],5,N)*scales[m] for m in range(len(scales))]
        decodedYPerScale=[activityDecoding(init_weights[m][:,maxYPerScale[m]],5,N)*scales[m] for m in range(len(scales))]
        x_multiscale_grid, y_multiscale_grid=np.sum(decodedXPerScale), np.sum(decodedYPerScale)
        # x_multiscale_grid, y_multiscale_grid=np.sum(decodedXPerScale[0:3]+x_grid_expect[3:6]), np.sum(decodedYPerScale[0:3]+y_grid_expect[3:6])
        x_grid.append(x_multiscale_grid+x_grid_expect)
        y_grid.append(y_multiscale_grid+y_grid_expect)

        '''Error integrated path'''
        q_err[2]+=angVel[i]
        q_err[0],q_err[1]=q_err[0]+vel[i]*np.cos(np.deg2rad(direction)), q_err[1]+vel[i]*np.sin(np.deg2rad(direction))
        x_integ_err.append(q_err[0])
        y_integ_err.append(q_err[1])

        if printing==True:
            print(f'dir: {np.rad2deg(q[2])}, {direction}')
            print(f'vel: {vel[i]}')
            print(f'decoded: {decodedXPerScale}, {decodedYPerScale}')
            print(f'expected: {x_grid_expect}, {y_grid_expect}')
            print(f'integ: {x_integ[-1]}, {y_integ[-1]}')
            print(f'CAN: {x_grid[-1]}, {y_grid[-1]}')
            print('')

        for k in range(4):
            axs[k].clear()
            axs[k].imshow(init_weights[k][:][:], cmap='jet')#(np.arange(N),prev_weights[k][:],color=colors[k])
            axs[k].spines[['top', 'left', 'right']].set_visible(False)
            axs[k].invert_yaxis()

    ani = FuncAnimation(fig, animate, interval=1,frames=test_length,repeat=False)
    # plt.show()

    writergif = animation.PillowWriter(fps=30) 
    ani.save(savePath, writer=writergif)

    print(f'CAN error: {errorTwoCoordinateLists(x_integ,y_integ, x_grid, y_grid)}')

