# -*- coding: utf-8 -*-
"""
The purpose of this code is to save .obj files of foot scans in .npy files.
This will increase processing time.

Created on Fri Mar 18 11:16:09 2022

@author: Eric.Honert
"""
#______________________________________________________________________________
# Import libraries
import numpy as np
import pywavefront
import os
#______________________________________________________________________________
# Read in files
fPath = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Data\\Aetrex Object Files\\'
fileExt = r'.obj'
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]
#______________________________________________________________________________
# Make a list of already created .npy files: Don't need to re-create
npyfiles = [fName for fName in os.listdir(fPath + 'python_files\\') if fName.endswith(r'.npy')]
# Eliminate extension
npyfiles = [fName[:-9] for fName in npyfiles]

# Loop throught the files & process only the new files
for fName in entries:
    if fName[:-4] not in npyfiles:
        # Read in the object file
        obj = pywavefront.Wavefront(fPath+fName,collect_faces=True,create_materials=True)
        # Create a variable with both the vertices, faces, and normals. 
        n_f = np.reshape(np.array(obj.mesh_list[0].materials[0].vertices),(int(len(obj.mesh_list[0].materials[0].vertices)/6),6))
        v_f_n = np.array([np.array(obj.vertices),np.array(obj.mesh_list[0].faces),n_f])
        # Save the data in a numpy format. This format can only be read in python,
        # but is fast to save and load.
        np.save(fPath+'python_files\\'+fName[:-4]+'_info.npy',v_f_n,)
    