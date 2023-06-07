# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:24:05 2023

@author: Eric.Honert
"""

import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import cycpd # cython (c-compiler)
import addcopyfighandler
from sklearn.neighbors import NearestNeighbors

# Read in the previously converted .npy files from the .obj files that are saved from the Aetrex scanner
fPath = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Data\\Aetrex Object Files\\python_files\\'
fileExt = r".npy"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]


def viz3d(PC1,PC2):
    # Visualize the 3D point cloud. Needs to be updated.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(PC1[:,0],PC1[:,1],PC1[:,2])
    ax.scatter3D(PC2[:,0],PC2[:,1],PC2[:,2])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

# Example: Use the first right scan as the "reference" scan
v_f_n = np.load(fPath+entries[1],allow_pickle=True)
# obj = pywavefront.Wavefront(fPath+entries[ii],collect_faces=True)
ref_point_cloud = v_f_n[0]

# Downsample the reference point cloud: this will be the same number of points on each scan
ref_len = len(ref_point_cloud)
random.seed(3) # set the seed to make it repeatable
ran_idx = np.sort(random.choices(np.array(range(ref_len)),k=4000))
ref_point_cloud = ref_point_cloud[ran_idx,:]

v_f_n = np.load(fPath+entries[3],allow_pickle=True)
# obj = pywavefront.Wavefront(fPath+entries[ii],collect_faces=True)
child_point_cloud = v_f_n[0]

# source_points = np.random.randn(100, 3) # Replace with your source point cloud 
# target_points = np.random.randn(100, 3) # Replace with your target point cloud


# Perform rigid registration: to be done with ICP in the future
# reg = RigidRegistration(X = ref_point_cloud, Y = child_point_cloud, max_iterations=200)
# TY, (s_reg, R_reg, t_reg) = reg.register()




# Perform deformable registration 
deformable_registration = cycpd.deformable_registration(X = child_point_cloud, Y = ref_point_cloud, alpha = 0.01, beta = 1, tolerance = 0.0001, max_iterations = 700) 
dum = deformable_registration.register()
viz3d(dum[0],child_point_cloud)

neigh = NearestNeighbors(n_neighbors=1)
test = neigh.fit(dum[0],child_point_cloud)


