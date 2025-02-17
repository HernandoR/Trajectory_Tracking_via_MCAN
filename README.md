# Trajectory_Tracking_via_MCAN

This repository is an implementation of the paper 'Trajectory Tracking via Multiscale Continuous Attractor Networks' published in IROS 2023. In this research, we present Multiscale Continuous Attractor Networks (MCAN), consisting of parallel neural networks at multiple spatial scales, to enable trajectory tracking over large velocity ranges. To overcome the limitations of the reliance of previous systems on hand-tuned parameters, we present a genetic algorithm-based approach for automated tuning of these networks, substantially improving their usability. To provide challenging navigational scale ranges, we open-source a flexible city-scale navigation simulator that adapts to any street network, enabling high throughput experimentation. 

![image info](./Results/PaperFigures/Architecture.png)

The simulator extracts road network data from Open Street Maps to generate an occupancy map consisting of traversable, realistic roads. The occupied cells in the map represent the drivable areas of the road network. A path-finding distance transform algorithm is then used to find the optimal route between two randomly generated points on the road map. Once the sample paths are generated, they can be traversed using the kinematics of a [bicycle motion model](https://github.com/winstxnhdw/KinematicBicycleModel), which is a common model used in the navigation of ground vehicles. During the traversal of the paths, motion information such as linear and angular velocities are recorded. This motion data is then used to evaluate the performance of the system, such as the accuracy of the estimated position and heading, the stability of the continuous attractor network, and the effectiveness of the buffer to prevent position resetting.
![image info](./Results/PaperFigures/BerlinPathFollowing.gif)

## Running Scripts 
1. The script GeospatialRoadMaps.py extracts road network data downloaded from open street maps and stores the map as an image with a specified resolution and radius 
2. The  script TestEnvironmentPathPlanning implements the city scale simulator described above for 4 cities using the maps stored in Datasets/CityScaleSimulatorMaps and stores path and velocity information in folders Datasets/CityScaleSimulatorPaths and Datasets/CityScaleSimulatorVelocities
3. The script CAN.py is a library of methods defining the dynamics of the attractor network (1D and 2D), the position decoding from the network state and the multiscale network implementation 
4. The script SimpleGA_multiscale_dynamics.py is an implementation of a genetic algorithm that determines the CAN network parameters for accurate trajectory tracking 
5. The script SelectiveMultiScalewithWraparound2D includes all the experiments described in the paper using the simulated and kitti dataset information


**Citation:** If you find this code useful in your research or project, please consider citing the following paper:

"Trajectory Tracking via Multiscale Continuous Attractor Networks"

*Author(s): Therese Joseph, Tobias Fischer, Michael Milford*
*Published in: IEEE/RSJ International Conference on Intelligent Robots and Systems 2023*
