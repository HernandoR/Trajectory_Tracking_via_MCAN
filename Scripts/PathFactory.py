import os
from pathlib import Path
from datetime import datetime
import math

import cv2
import random
import numpy as np

from tqdm import tqdm
import roboticstoolbox as rtb
from PIL import Image, ImageFilter
from roboticstoolbox import DistanceTransformPlanner
from roboticstoolbox import Bicycle, RandomPath

class PathFactory:
    def __init__(self,city='Newyork', scaletype="single", run=False, plotting=True) -> None:
        self.city = city
        self.scaletype = scaletype
        self.run = run
        self.plotting = plotting
        
    @classmethod
    def processMap(map_path, scale_percent):
        img = np.array(Image.open(map_path).convert("L"))
        imgColor = np.array(Image.open(map_path))

        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        imgColor = cv2.resize(imgColor, dim, interpolation=cv2.INTER_AREA)

        imgSharp = cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        imgdia = np.zeros((np.shape(img)))
        imgdia[img == 255] = 0
        imgdia[img < 255] = 1
        imgdia = cv2.dilate(imgdia, np.ones((1, 1), np.uint8))

        binMap = np.zeros((np.shape(img)))
        binMap[imgSharp < 255] = 0
        binMap[imgSharp == 255] = 1

        return img, imgColor, imgdia, binMap


    def findPathsThroughRandomPoints(img, num_locations, outfile):
        if Path(outfile).exists():
            print("random path {outfile} exists, using existing file ...")
            return 1

        free_spaces = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] == 0:
                    free_spaces.append((j, i))

        locations = random.choices(free_spaces, k=num_locations)
        print(locations)

        paths = []
            
        tbar = tqdm(range(len(locations) - 1), disable='GITHUB_ACTIONS' in os.environ)
        for i in tbar:
            try:
                dx = DistanceTransformPlanner(
                    img, goal=locations[i + 1], distance="euclidean"
                )
                dx.plan()
                path = dx.query(start=locations[i])
            except TimeoutError:
                # locations.append(free_spaces, k=1)
                print("removed a goal")
                continue
            paths.append(path)
            print(f"done {i+1} paths")

        # outfile='./results/testEnvMultiplePathsSeparate_5kmrad_100pdi_0.2line.npy'
        # if exist, rename existing by it's modiﬁed Ymd-HMS
        if Path.exists(outfile):
            create_time = datetime.fromtimestamp(Path.stat(outfile).st_mtime)
            Path.rename(outfile, outfile + create_time.strftime("%Y%m%d-%H%M%S"))
        np.save(outfile, np.array(paths))

        print(paths)


    def remove_consecutive_duplicates(coords):
        # Initialize a new list to store the filtered coordinates
        filtered = []
        # Add the first element to the filtered list
        filtered.append(coords[0])
        # Loop through the remaining elements
        for i in range(1, len(coords)):
            # Check if the current element is different from the previous element
            if coords[i] != coords[i - 1]:
                # If it is, add it to the filtered list
                filtered.append(coords[i])
            # if math.tan((coords[i][1]-coords[i-1][1])/((coords[i][0]-coords[i-1][0])))>= 2*np.pi :

        # Return the filtered list
        return filtered


    def rescalePath(paths, path_idx, img, scale, pxlPerMeter):
        # convert path to image
        path_x, path_y = zip(*paths[path_idx])
        pathImg = np.zeros((np.shape(img)))
        pathImg[(path_y, path_x)] = 1

        if scale != 1:
            # scale down path image by given percentage
            width = int(img.shape[1] * scale)
            height = int(img.shape[0] * scale)
            dim = (width, height)
            newImg = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            pathImgRescaled = cv2.resize(pathImg, dim, interpolation=cv2.INTER_AREA)

        else:
            # newImg = path_idx
            newImg = img
            pathImgRescaled = pathImg

        return (
            [np.round(x * scale) for x in path_x],
            [np.round(y * scale) for y in path_y],
            newImg,
            pathImgRescaled,
            pxlPerMeter * scale,
        )
