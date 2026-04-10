# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:40:37 2023

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

fPath = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\BigData\\FootScan Data\\Aetrex Object Files\\python_files\\'

#______________________________________________________________________________
# Functions
def viz3d(PC):
    # Visualize the 3D point cloud. Needs to be updated.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(PC[:,0],PC[:,1],PC[:,2])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

def plt_trimesh(vert,face):
    import plotly.io as pio
    pio.renderers.default='browser'
    fig = ff.create_trisurf(vert[:,0],vert[:,1],vert[:,2],face)
    fig.update_layout(scene = dict(aspectmode = 'data'))
    fig.show()

#______________________________________________________________________________



v_f_n = np.load(fPath+'Robin Fassett Carman Left_info.npy',allow_pickle=True)

point_cloud = v_f_n[0]*1000
faces = v_f_n[1]

# Rotate the point cloud and the correspondin unique normals
zrot = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
point_cloud = point_cloud @ zrot

# Perform a base alignment of the point cloud
point_cloud[:,0] = (point_cloud[:,0]-np.mean(point_cloud[:,0]))*-1
point_cloud[:,1] = point_cloud[:,1]-np.min(point_cloud[:,1])
point_cloud[:,2] = point_cloud[:,2]-np.min(point_cloud[:,2])
viz3d(point_cloud)
plt_trimesh(point_cloud, faces)

appm1s_105_foot = mesh.Mesh(np.zeros(faces.shape[0],dtype = mesh.Mesh.dtype))

for ii, f in enumerate(faces):
    for jj in range(3):
        appm1s_105_foot.vectors[ii][jj] = point_cloud[f[jj],:]

appm1s_105_foot.save('appm1s_105_foot.stl')
