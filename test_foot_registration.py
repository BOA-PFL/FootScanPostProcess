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
from scipy.spatial.distance import cdist
import alphashape

# Read in the previously converted .npy files from the .obj files that are saved from the Aetrex scanner
fPath = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\BigData\\FootScan Data\\Aetrex Object Files\\python_files\\'
fSave = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\BigData\\FootScan Data\\Aetrex Object Files\\registered_files\\'
fileExt = r".npy"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

# Overall sets for the registration
npoints = 3000
zrot = np.array([[-1,0,0],[0,-1,0],[0,0,1]])

def viz3d(PC1,PC2):
    # Visualize the 3D point cloud. Needs to be updated.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(PC1[:,0],PC1[:,1],PC1[:,2],'r')
    ax.scatter3D(PC2[:,0],PC2[:,1],PC2[:,2],'b')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

def create_foot_outline(footPC,outline_height):
    idx = footPC[:,2] < outline_height
    foot_low = footPC[idx,0:2]
    
    test_shape = alphashape.alphashape(foot_low,100)
    
    foot_outline = np.asarray([x for x in test_shape.boundary.coords])
    
    return foot_outline

def AlignBraddock(footPC):
    foot_outline = create_foot_outline(footPC,0.02)
    
    footL = np.max(foot_outline[:,1])
    # Step 1: Find the most medial point less than 20% of the foot length
    idx = foot_outline[:,1] < 0.2*footL
    rear_foot = foot_outline[idx,:]
    min_idx = np.argmin(rear_foot[:,0])
    pt1 = rear_foot[min_idx,:]
        
    # Step 2: Find the Approximate MTP1 Head
    MTP1 = findMTP1(foot_outline)
        
    # Find the rotation angle
    rot_angle = np.arctan2(pt1[0]-MTP1[0],MTP1[1]-pt1[1])
    # Provide the 2D rotation matrix
    R2D = np.asarray([[np.cos(rot_angle),np.sin(rot_angle)],[-np.sin(rot_angle),np.cos(rot_angle)]])
        
    foot_outline = np.transpose(R2D @ np.transpose(foot_outline))
        
    MTP1 = findMTP1(foot_outline)
    R3D = np.asarray([[np.cos(rot_angle),np.sin(rot_angle),0],[-np.sin(rot_angle),np.cos(rot_angle),0],[0,0,1]])
    scan_rot = np.transpose(R3D @ np.transpose(footPC))
    scan_rot[:,0] = scan_rot[:,0] - MTP1[0]
    scan_rot[:,1] = scan_rot[:,1] - np.min(scan_rot[:,1])
    
    return scan_rot
    
    
def findMTP1(foot_outline):
    footL = np.max(foot_outline[:,1])
    idx = np.argmax(foot_outline[:,1])
    
    current_point = np.array([foot_outline[idx,0],0.77*footL+np.min(foot_outline[:,1])])
    
    der_angle = 1
        
    track_st_point = []
    track_angle = []
    
    while der_angle < 90 and current_point[1] > 0.6*footL:
        st_point = current_point
        track_st_point.append(st_point)
        
        # Find the closest point 1mm behind the start point
        idx = np.logical_and(foot_outline[:,1]<st_point[1]-0.001,foot_outline[:,0] < np.mean(foot_outline[:,0]))
        new_pts = foot_outline[idx,:]
        
        min_idx = np.argmin(np.linalg.norm(foot_outline[idx,:] - st_point,axis=1))
        
        current_point = new_pts[min_idx,:]
        # Angle w.r.t. th horizontal: call it the derivative angle
        der_angle = np.arctan2(st_point[1]-current_point[1],st_point[0]-current_point[0])*180/np.pi
        track_angle.append(der_angle)
        
    # If the Met head was not found, pick greatest angle
    if der_angle < 90:
        idx = np.argmax(np.asarray(track_angle))
        MTP1 = track_st_point[idx]
    else: 
        MTP1 = st_point
        
    return MTP1
  
# Example: Use the first right scan as the "reference" scan
v_f_n = np.load(fPath+'Amanda Basham Right_info.npy',allow_pickle=True)
# Note: reference foot scan is Amanda Basham Left.obj
ref_point_cloud = v_f_n[0]
ref_faces = v_f_n[1]
ref_point_cloud = ref_point_cloud @ zrot
# Perform a base alignment of the point cloud
ref_point_cloud[:,0] = ref_point_cloud[:,0]-np.mean(ref_point_cloud[:,0])
ref_point_cloud[:,1] = ref_point_cloud[:,1]-np.min(ref_point_cloud[:,1])
ref_point_cloud[:,2] = ref_point_cloud[:,2]-np.min(ref_point_cloud[:,2])
#______________________________________________________________________
# Align the point cloud in the same manor similar to using a braddock device
ref_point_cloud = AlignBraddock(ref_point_cloud)

# Pull already registered scans to make sure that they are not re-registered
reg_files = [fName for fName in os.listdir(fSave) if fName.endswith(r'.npz')]
reg_files = [fName[:-8] for fName in reg_files]

# Index through all of the scans
for entry in entries:
    # Only look un-registered scans
    if entry[:-9] not in reg_files:
        print(entry)
        # Open the .py file saved from the .obj file
        v_f_n = np.load(fPath+entry,allow_pickle=True)
        child_point_cloud = v_f_n[0]
        # Reflect around the M/L axis for left feet: enable the same registration 
        # for all of the scans
        if 'Left' in entry:
            child_point_cloud[:,0] = -child_point_cloud[:,0]
        # Perform alignment of the scans
        child_point_cloud = child_point_cloud @ zrot        
        # Perform a base alignment of the point cloud
        child_point_cloud[:,0] = child_point_cloud[:,0]-np.mean(child_point_cloud[:,0])
        child_point_cloud[:,1] = child_point_cloud[:,1]-np.min(child_point_cloud[:,1])
        child_point_cloud[:,2] = child_point_cloud[:,2]-np.min(child_point_cloud[:,2])
        # Align the point cloud in the same manor similar to using a braddock device
        child_point_cloud = AlignBraddock(child_point_cloud)
        # Downsample the child point cloud
        random.seed(2) # set the seed to make it repeatable
        ran_idx = np.sort(random.sample(list(range(len(child_point_cloud))),k=npoints))
        child_point_cloudDS = child_point_cloud[ran_idx,:]
        # Perform deformable registration
        deformable_registration = cycpd.deformable_registration(X = child_point_cloudDS, Y = ref_point_cloud, alpha = 0.0001, beta = 0.5, tolerance = 0.0001, max_iterations = 1000) 
        reg_ptcloud_tot = deformable_registration.register()
        reg_point_cloud = reg_ptcloud_tot[0]
        # viz3d(reg_point_cloud,child_point_cloud)    
        # Find the error in the estimation using euclidean distances
        dists = cdist(reg_point_cloud,child_point_cloud,'euclidean')
        idx = np.argmin(dists,axis = 1)
        reg_errors = np.min(dists,axis = 1)
        print('Average Error: ', np.mean(reg_errors)*1000,' mm')
        print('Max Error: ', np.max(reg_errors)*1000,' mm')
        
        
        # Save the relevant information for each registration:
        # Registered point cloud, normals, errors, array for redundant registration (good_idx)
        # Note: the faces of the reduced point cloud will be calcuated separately    
        np.savez(fSave + entry[:-9] + '_reg', **{'reg_point_cloud': reg_point_cloud,'reg_faces': ref_faces,
                                                 'reg_errors': reg_errors})
