# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:18:56 2017

@author: Arthur Le Guennec
"""

###########################################################
###        STEP ONE                                     ###
###   Script to compute feature on raster               ###
###########################################################

## Goals:   - compute feature on raster
##          - use several scales (radiuses: 0.25 to 7 meters)

## Step:    1) load raster laz files
##          2) load laz files
##          3) compute descriptors on raster with the original file

from laspy.file import File
from math import atan2
from sklearn.neighbors import KDTree
from useful_functions.function_las import *
from useful_functions.function_canupo import *
#from useful_functions.function_figures import *
#from useful_functions.function_canupo import *
#from useful_functions.function_random_forest import *


import numpy as np
import scipy.linalg as la
import pandas as pd
import copy


import os
import sys


def main():
    dir_raster_in = "../../data/rasterized_tiles/"
    dir_in = "../../data/labeled_tiles/"

    scales = [0.25, 0.5, 1, 2, 2.5, 3, 4, 5, 6, 7]

    f = open('train_data.txt', 'r')
    filenames_train_C2 = f.readlines()
    filenames_train_C3 = copy.copy(filenames_train_C2)
    filenames_train_raster_C2 = copy.copy(filenames_train_C2)
    for i in range(len(filenames_train_C2)):
        if (i != len(filenames_train_C2) - 1): 
            filenames_train_C2[i] = str(filenames_train_C2[i][0:-1]) #to delete '\n'
        else:
            filenames_train_C2[i] = str(filenames_train_C2[i])
        foo = str(filenames_train_C2[i])
        filenames_train_C3[i] = foo.replace('C2', 'C3')
        filenames_train_raster_C2[i] = foo.replace('.laz', 
                                                   '_raster_2_5_low.laz')
    
    for i in range(len(filenames_train_C2)):
        inFile_C2 = load_las_file(dir_in + filenames_train_C2[i])
        inFile_C3 = load_las_file(dir_in + filenames_train_C3[i])
        inFile_C2_raster = load_las_file(dir_raster_in 
                                         + filenames_train_raster_C2[i])
        
        
        cloud_C2 = np.vstack([inFile_C2.x,
                              inFile_C2.y,
                              inFile_C2.z,
                              inFile_C2.classification,
                              inFile_C2.intensity,
                              inFile_C2.num_returns,
                              inFile_C2.return_num]).transpose()

        global cloud_C3
        cloud_C3 = np.vstack([inFile_C3.x,
                              inFile_C3.y,
                              inFile_C3.z,
                              inFile_C3.classification,
                              inFile_C3.intensity,
                              inFile_C3.num_returns,
                              inFile_C3.return_num]).transpose()
        
        
        print("Calcul KDTree C2")
        kdtree_C2 = KDTree(coord_C2, leaf_size=40)
        global kdtree_C3
        print("Calcul KDTree C3")
        kdtree_C3 = KDTree(coord_C3, leaf_size=40)
#        if (filename == filenames_train_C2[0]):
#            nam_feat = name_features.copy()
#
#            [foo, labels_train, column_names] = extract_features(inFile_C2, nam_feat, scales)
#            data_train = foo
#        else:
#            nam_feat = name_features.copy()
#            [foo, lab_tmp, column_names] = extract_features(inFile_C2, nam_feat, scales)
#            data_train = np.vstack([data_train, foo])
#            #foo = inFile_C2.classification
#            labels_train = np.concatenate([labels_train, lab_tmp])
    
    return
    
    

if __name__ == "__main__":
    main()