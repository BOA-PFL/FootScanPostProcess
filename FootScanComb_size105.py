# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 15:38:49 2023

@author: Eric.Honert
"""


# Libraries
import numpy as np
import pandas as pd
from procrustes import rotational
import alphashape
import matplotlib.pyplot as plt
import os
from plotly.tools import FigureFactory as ff
import matplotlib.tri as mtri
from scipy.spatial.distance import cdist
from stl import mesh

fPath = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\BigData\\FootScan Data\\Aetrex Object Files\\'

# Read the DB for the ovezrall scan metrics that were computed directly from the 
# foot scan data
foot_metrics = pd.read_csv(fPath + 'python_files\\SummaryMetrics.csv',sep=',', header = 0)

maxL = 0.282
minL = 0.262

subset = foot_metrics[(foot_metrics['FootLength'] > minL) & (foot_metrics['FootLength'] < maxL)]
subname = []

###############################################################################
def viz3d(PC1):
    # Visualize the 3D point cloud. Needs to be updated.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(PC1[:,0],PC1[:,1],PC1[:,2],'r')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

def viz3d_2pc(PC1,PC2):
    # Visualize the 3D point cloud. Needs to be updated.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(PC1[:,0],PC1[:,1],PC1[:,2],'r')
    ax.scatter3D(PC2[:,0],PC2[:,1],PC2[:,2],'b')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    
def plt_trimesh(vert,face):
    import plotly.io as pio
    pio.renderers.default='browser'
    fig = ff.create_trisurf(vert[:,0],vert[:,1],vert[:,2],face)
    fig.update_layout(scene = dict(aspectmode = 'data'))
    fig.show()
    
def plt_normals(footPC,norms):
    ax = plt.figure().add_subplot(projection='3d')
    ax.quiver(footPC[:,0], footPC[:,1], footPC[:,2], norms[:,0], norms[:,1], norms[:,2], length=5, normalize=True) 

def FootDiscreteMetrics(footPC):
     foot_outline = create_foot_outline(footPC,0.025)
     
     # Foot Length
     footL = np.max(foot_outline[:,1])-min(foot_outline[:,1])
     
     # Heel Width
     idx = foot_outline[:,1]<(0.2*footL)
     rear_foot = foot_outline[idx,:]
     heel_width = np.max(rear_foot[:,0]) - np.min(rear_foot[:,0])
     # 1st Met Head Distance
     MTP1 = findMTP1(foot_outline)
     MTP1d = MTP1[1]-min(foot_outline[:,1])
     
     # Forefoot Width
     foot_outline = create_foot_outline(footPC,0.025)
     idx = (foot_outline[:,1] < MTP1[1]-0.015)*(foot_outline[:,1]>0.5*np.max(foot_outline[:,1]))
     mid_foot = foot_outline[idx,:]
     MTP5 = max(mid_foot[:,0])
     forefoot_width = MTP5-MTP1[0]
         
     # Instep height: Based on Jurca 2019
     idx = footPC[:,1]>(0.55*footL)
     inH = np.max(footPC[idx,2])
     
     return [footL,heel_width,forefoot_width,inH,MTP1d]

def AlignBraddock(footPC):
    foot_outline = create_foot_outline(footPC,0.025)
    

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

def create_foot_outline(footPC_f,outline_height):
    
    # footPC = dum
    # outline_height = 0.025
    
    idx = footPC_f[:,2] < outline_height
    foot_low = footPC_f[idx,0:2]
    
    test_shape = alphashape.alphashape(foot_low,100)
    
    foot_outline = np.asarray([x for x in test_shape.boundary.coords])
    
    return foot_outline

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
###############################################################################
# Preallocate variables
store_FL = []
store_FFW = []
store_HW = []
store_inH = []
store_MTP1 = []


for name in subset['Name_Side']:
    if 'Left' in name:
        subname.append(name[:-5])
    else:
        subname.append(name[:-6])

subname = np.unique(subname)

# Longer loop to only include the sub-set of the subjects with the correct foot length
reg_files = []
for fName in os.listdir(fPath + 'registered_files\\'):
    if fName.endswith('.npz'):
        if 'Left' in fName:
            if fName[:-13] in subname:
                reg_files.append(fName)
        elif 'Right' in fName:
            if fName[:-14] in subname:
                reg_files.append(fName)


count = 0
# Open the files and store the point clouds
# Note: *1000 for the point clouds to put them into mm
for fName in reg_files:
    reg_info = np.load(fPath + 'registered_files\\' + fName, allow_pickle=True)
    print(fName)
    print(max(reg_info['reg_errors'])*1000)
    if max(reg_info['reg_errors'])*1000 < 8:
        if count == 0:
            footPCs = np.expand_dims(reg_info['reg_point_cloud']*1000, axis = 2)
        else:
            footPCs = np.append(footPCs,np.expand_dims(reg_info['reg_point_cloud']*1000, axis = 2),axis = 2)
        count = 1
faces = reg_info['reg_faces']

#______________________________________________________________________________
# Align the scans with a general procrustes analysis
tol_gpa = 0.001
max_IT = 1000

x_bar = np.mean(footPCs,axis = 2)
P2d = 1 # intial error
IT = 1

Z = footPCs # Intialize storage array for the GPA

while P2d > tol_gpa and IT < max_IT:
    
    for ii in range(footPCs.shape[2]):
        result = rotational(Z[:,:,ii],Z[:,:,0], translate = True)
        Z[:,:,ii] = result.new_a
    
    p_bar = np.mean(Z,axis = 2)
    P2d = np.sum((p_bar-x_bar)**2)
    x_bar = p_bar
    
    IT = IT + 1

avg_shape = x_bar
# Estimate normals


PCin = avg_shape
# norms = np.zeros(PCin.shape)
# for ii in range(len(PCin)):
#     # first find the local faces for the selected point
#     loc_faces = faces[np.sum(faces == ii,axis = 1, dtype = bool),:]
#     loc_norms = np.zeros(loc_faces.shape)
#     for jj in range(len(loc_faces[:,0])):
#         A = PCin[loc_faces[jj,1],:] - PCin[loc_faces[jj,0],:]
#         B = PCin[loc_faces[jj,2],:] - PCin[loc_faces[jj,0],:]
#         loc_norms[jj,:] = np.cross(A,B)/np.linalg.norm(np.cross(A,B))
#         # Make sure that the local norms are pointing approx. in the same direction
#         # Test will be based on the first point, if the points are greater than 
#         # 90 degrees out of line from one another, flip the local normal
#         # 90 degrees: dot product > 0
#         if np.dot(loc_norms[0,:],loc_norms[jj,:]) < 0:
#             loc_norms[jj,:] = -loc_norms[jj,:]
    
#     norms[ii,:] = np.mean(loc_norms,axis = 0)/np.linalg.norm(np.mean(loc_norms,axis = 0))

# Look at the closest n points to estimate the normals
NN = 5 # number of neighbors to estimate the normals for each point
norms = np.zeros(PCin.shape)
for ii in range(len(PCin)):
    # Find the closest n points
    close_neigh = np.argsort(np.linalg.norm(PCin - PCin[ii,:],axis=1))
    
    for count, jj in enumerate(close_neigh[0:NN]):
        # first find the local faces for the selected point
        pnt_faces = faces[np.sum(faces == jj,axis = 1, dtype = bool),:]
        pnt_norms = np.zeros(pnt_faces.shape)
        for kk in range(len(pnt_faces[:,0])):
            A = PCin[pnt_faces[kk,1],:] - PCin[pnt_faces[kk,0],:]
            B = PCin[pnt_faces[kk,2],:] - PCin[pnt_faces[kk,0],:]
            pnt_norms[kk,:] = np.cross(A,B)/np.linalg.norm(np.cross(A,B))
            # Make sure that the local norms are pointing approx. in the same direction
            # Test will be based on the first point, if the points are greater than 
            # 90 degrees out of line from one another, flip the local normal
            # 90 degrees: dot product > 0
            if np.dot(pnt_norms[0,:],pnt_norms[kk,:]) < 0:
                pnt_norms[kk,:] = -pnt_norms[kk,:]
        
        if count == 0:
            loc_norms = pnt_norms
        else:
            loc_norms = np.concatenate((loc_norms,pnt_norms))
    
    loc_norms = np.unique(loc_norms, axis = 0)  # Use only unique norms
    norms[ii,:] = np.mean(loc_norms,axis = 0)/np.linalg.norm(np.mean(loc_norms,axis = 0))

# Compute the standard deviation at each point:
std_norm = np.sum(abs(norms)*np.std(Z,axis = 2),axis = 1)

SD_shape = np.array([ele*norms[count,:] for count, ele in enumerate(std_norm)])
avg_shape = avg_shape-np.min(avg_shape,axis=0)

m2s_shape = avg_shape-2*SD_shape
m1s_shape = avg_shape-1*SD_shape
#______________________________________________________________________________
# Quick visualization
# plt_trimesh(avg_shape,faces,std_norm)
plt_trimesh(avg_shape,faces)
plt_trimesh(m1s_shape,faces)

# Visualize on top of one another
viz3d_2pc(avg_shape, m1s_shape)

avg105_foot = mesh.Mesh(np.zeros(faces.shape[0],dtype = mesh.Mesh.dtype))

for ii, f in enumerate(faces):
    for jj in range(3):
        avg105_foot.vectors[ii][jj] = avg_shape[f[jj],:]
        
m2s_105_foot = mesh.Mesh(np.zeros(faces.shape[0],dtype = mesh.Mesh.dtype))

for ii, f in enumerate(faces):
    for jj in range(3):
        m2s_105_foot.vectors[ii][jj] = m2s_shape[f[jj],:]




# m2s_105_foot.save('m2s_105_foot.stl')
avg105_foot.save('avg105_foot.stl')

# # Look at the discrete metrics for both shapes
# dum = m2s_shape

# dum = dum/1000
# # Perform a base alignment of the point cloud
# dum[:,0] = dum[:,0]-np.mean(dum[:,0])
# dum[:,1] = dum[:,1]-np.min(dum[:,1])
# dum[:,2] = dum[:,2]-np.min(dum[:,2])
# #______________________________________________________________________
# # Align the point cloud in the same manor similar to using a braddock device
# dum = AlignBraddock(dum)

# # Extract the discrete metrics
# [footL,heel_width,forefoot_width,instepH,MTP1d] = FootDiscreteMetrics(dum)

# # Storing
# store_FL.append(footL)
# store_FFW.append(forefoot_width)
# store_HW.append(heel_width)
# store_inH.append(instepH)
# store_MTP1.append(MTP1d)

# # 
# # viz3d_2pc(dum1, dum2)

# # viz3d(dum)

# # plt_normals(avg_shape,norms)

# plt.figure
# plt.subplot(2,2,1)
# plt.hist(subset.HeelWidth)
# plt.xlabel('Heel Width')
# plt.subplot(2,2,2)
# plt.hist(subset.ForefootWidth)
# plt.xlabel('Forefoot Width')
# plt.subplot(2,2,3)
# plt.hist(subset.InstepHeight)
# plt.xlabel('Instep Height')
# plt.subplot(2,2,4)
# plt.hist(subset.MTP1)
# plt.xlabel('MTP1')

# # ax = plt.figure().add_subplot(projection='3d')
# # ax.plot_trisurf(avg_shape[:,0],avg_shape[:,1],avg_shape[:,2],triangles = faces)

# # ax = plt.figure().add_subplot(projection='3d')
# # ax.plot_trisurf(avg_shape[:,0],avg_shape[:,1],avg_shape[:,2],triangles = faces)