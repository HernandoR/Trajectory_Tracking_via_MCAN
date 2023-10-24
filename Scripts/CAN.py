import math
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
from AttractorNetwork import attractorNetwork1D, attractorNetwork2D, activityDecoding, activityDecodingAngle

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
    prev_weights, net,N, vel, direction, iterations, wrap_iterations, x_grid_expect, y_grid_expect,scales):
    '''Select scale and initilise wrap storage'''
    delta = [(vel/scales[i]) for i in range(len(scales))]
    chosen_scale_idx=scale_selection(vel,scales)
    # print(vel, scales, cs_idx)
    wrap_rows=np.zeros((len(scales)))
    wrap_cols=np.zeros((len(scales)))

    '''Update selected scale'''

    for i in range(iterations):
        prev_weights[chosen_scale_idx][:], wrap_rows_cs, wrap_cols_cs= net.update_weights_dynamics(
            prev_weights[chosen_scale_idx][:],direction, delta[chosen_scale_idx])
        prev_weights[chosen_scale_idx][prev_weights[chosen_scale_idx][:]<0]=0
        x_grid_expect+=wrap_cols_cs*N*scales[chosen_scale_idx]
        y_grid_expect+=wrap_rows_cs*N*scales[chosen_scale_idx]
    


    
        # if np.any(wrap_cols_cs!=0):
        #     print(f"------------------------------------------------------------------------------------------------------wrap_cols {wrap_cols_cs}, {scales[cs_idx]}")
        # if np.any(wrap_rows_cs!=0):
        #     print(f"------------------------------------------------------------------------------------------------------wrap_rows {wrap_rows_cs}, {scales[cs_idx]}")

    wrap=0   
    return prev_weights, wrap, x_grid_expect, y_grid_expect


# @numba.jit(nopython=False)
def headDirectionAndPlaceNoWrapNet(scales, vel, angVel,savePath: str|None, printing=False, N=100, returnTypes=None, genome=None):
    # global theata_called_iters,theta_weights, prev_weights, q, wrap_counter, current_i, x_grid_expect, y_grid_expect 
    
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
    network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)

    

    '''__________________________Storage and initilisation parameters______________________________'''
    # scales=[0.25,1,4,16]
    theta_weights=np.zeros(360)
    theata_called_iters=0
    # start_x, start_y=(50*scales[3])+(50*scales[4])+(50*scales[5]),(50*scales[3])+(50*scales[4])+(50*scales[5])
    wrap_counter=[0,0,0,0,0,0]
    x_grid_expect, y_grid_expect =0,0
    q=[0,0,0]
    # x_integ_err, y_integ_err=[],[]
    x_grid, y_grid=[0]*len(vel),[0]*len(vel)
    x_integ, y_integ=[0]*len(vel),[0]*len(vel)
    x_integ_err, y_integ_err=[0]*len(vel),[0]*len(vel)
    q_err=[0,0,0]

    '''__________________________Initilising scales in the center and at the edge_____________________________'''
    prev_weights=[np.zeros((N,N)) for _ in range(len(scales))]
    for n in range(len(scales)):
        for m in range(iterations):
            prev_weights[n]=network.excitations(0,0)
            prev_weights[n]=network.update_weights_dynamics_row_col(prev_weights[n][:], 0, 0)
            prev_weights[n][prev_weights[n][:]<0]=0
    

    '''_______________________________Iterating through simulation velocities_______________________________'''
    # for i in range(1,len(vel)):   
    tbar=tqdm(range(1,len(vel)))
    tbar.set_description("CAN")
    for i in tbar:
        '''Path integration'''
        q[2]+=angVel[i]
        q[0],q[1]=q[0]+vel[i]*np.cos(q[2]), q[1]+vel[i]*np.sin(q[2])
        x_integ[i]=q[0]
        y_integ[i]=q[1]

        # go through next 15 lines find input
        # input is the speed with direction as the shifting in the grid
        '''Mutliscale CAN update'''
        N_dir=360
        theta_weights,theata_called_iters=headDirection(theta_weights, np.rad2deg(angVel[i]), 0,theata_called_iters)
        direction=activityDecodingAngle(theta_weights,5,N_dir)
        prev_weights, wrap, x_grid_expect, y_grid_expect= hierarchicalNetwork2DGridNowrapNet(
            prev_weights, network, N, vel[i], direction, iterations,wrap_iterations, x_grid_expect, y_grid_expect, scales)

        '''1D method for decoding'''
        maxXPerScale, maxYPerScale = (np.array([np.argmax(np.max(prev_weights[m], axis=1)) for m in range(len(scales))]),
                                    np.array([np.argmax(np.max(prev_weights[m], axis=0)) for m in range(len(scales))]))
        decodedXPerScale=[activityDecoding(prev_weights[m][maxXPerScale[m], :],5,N)*scales[m] for m in range(len(scales))]
        decodedYPerScale=[activityDecoding(prev_weights[m][:,maxYPerScale[m]],5,N)*scales[m] for m in range(len(scales))]
        x_multiscale_grid, y_multiscale_grid=np.sum(decodedXPerScale), np.sum(decodedYPerScale)
        # x_multiscale_grid, y_multiscale_grid=np.sum(decodedXPerScale[0:3]+x_grid_expect[3:6]), np.sum(decodedYPerScale[0:3]+y_grid_expect[3:6])
        x_grid[i]=x_multiscale_grid+x_grid_expect
        y_grid[i]=y_multiscale_grid+y_grid_expect

        '''Error integrated path'''
        q_err[2]+=angVel[i]
        q_err[0],q_err[1]=q_err[0]+vel[i]*np.cos(np.deg2rad(direction)), q_err[1]+vel[i]*np.sin(np.deg2rad(direction))
        x_integ_err[i]=q_err[0]
        y_integ_err[i]=q_err[1]

        if printing==True:
            print(f'dir: {np.rad2deg(q[2])}, {direction}')
            print(f'vel: {vel[i]}')
            print(f'decoded: {decodedXPerScale}, {decodedYPerScale}')
            print(f'expected: {x_grid_expect}, {y_grid_expect}')
            print(f'integ: {x_integ[-1]}, {y_integ[-1]}')
            print(f'CAN: {x_grid[-1]}, {y_grid[-1]}')
            print('')

    if savePath != None:
        savePath=Path(savePath)
        if not savePath.parent.exists():
            savePath.parent.mkdir(parents=True)
        np.save(savePath, np.array([x_grid, y_grid, x_integ, y_integ, x_integ_err, y_integ_err]))
    
    print(f'CAN error: {errorTwoCoordinateLists(x_integ,y_integ, x_grid, y_grid)}') 
        
    

    if returnTypes=='Error':
        return errorTwoCoordinateLists(x_integ,y_integ, x_grid, y_grid)
    elif returnTypes=='PlotShow':
        plt.plot(x_integ, y_integ, 'g.')
        # plt.plot(x_integ_err, y_integ_err, 'y.')
        plt.plot(x_grid, y_grid, 'b.')
        plt.axis('equal')
        plt.title('Test Environment 2D space')
        plt.legend(('Path Integration without Error','Multiscale Grid Decoding'))
        plt.show()
    elif returnTypes=='posInteg+CAN':
        return x_integ,y_integ, x_grid, y_grid


def headDirectionAndPlaceNoWrapNetAnimate(scales, test_length, vel, angVel,savePath, plot=False, printing=True, N=100):
    global theata_called_iters,theta_weights, prev_weights, q, wrap_counter, current_i, x_grid_expect, y_grid_expect 

    num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=10,2,1.10262708e-01,6.51431074e-04,3,2 #with decimals 200 iters fitness -395 modified
    num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=10,10,1,0.0008,2,1
    network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)

    

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
    prev_weights=[np.zeros((N,N)) for _ in range(len(scales))]
    for n in range(len(scales)):
        for m in range(iterations):
            prev_weights[n]=network.excitations(0,0)
            prev_weights[n]=network.update_weights_dynamics_row_col(prev_weights[n][:], 0, 0)
            prev_weights[n][prev_weights[n][:]<0]=0
    

    '''_______________________________Iterating through simulation velocities_______________________________'''
    fig, axs = plt.subplots(1,4,figsize=(5, 3)) 
    def animate(i):  
        global theata_called_iters,theta_weights, prev_weights, q, wrap_counter, current_i, x_grid_expect, y_grid_expect 

        '''Path integration'''
        q[2]+=angVel[i]
        q[0],q[1]=q[0]+vel[i]*np.cos(q[2]), q[1]+vel[i]*np.sin(q[2])
        x_integ.append(q[0])
        y_integ.append(q[1])


        '''Mutliscale CAN update'''
        N_dir=360
        theta_weights,theata_called_iters=headDirection(theta_weights, np.rad2deg(angVel[i]), 0,theata_called_iters)
        direction=activityDecodingAngle(theta_weights,5,N_dir)
        prev_weights, wrap, x_grid_expect, y_grid_expect= hierarchicalNetwork2DGridNowrapNet(prev_weights, network, N, vel[i], direction, iterations,wrap_iterations, x_grid_expect, y_grid_expect, scales)

        '''1D method for decoding'''
        maxXPerScale, maxYPerScale = np.array([np.argmax(np.max(prev_weights[m], axis=1)) for m in range(len(scales))]), np.array([np.argmax(np.max(prev_weights[m], axis=0)) for m in range(len(scales))])
        decodedXPerScale=[activityDecoding(prev_weights[m][maxXPerScale[m], :],5,N)*scales[m] for m in range(len(scales))]
        decodedYPerScale=[activityDecoding(prev_weights[m][:,maxYPerScale[m]],5,N)*scales[m] for m in range(len(scales))]
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
            axs[k].imshow(prev_weights[k][:][:], cmap='jet')#(np.arange(N),prev_weights[k][:],color=colors[k])
            axs[k].spines[['top', 'left', 'right']].set_visible(False)
            axs[k].invert_yaxis()

    ani = FuncAnimation(fig, animate, interval=1,frames=test_length,repeat=False)
    # plt.show()

    writergif = animation.PillowWriter(fps=30) 
    ani.save(savePath, writer=writergif)

    print(f'CAN error: {errorTwoCoordinateLists(x_integ,y_integ, x_grid, y_grid)}')

