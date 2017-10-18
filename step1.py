# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:56:58 2017

@author: Arthur Le Guennec
"""

###########################################################
###        STEP ONE                                     ###
###   Script to rasterize las or laz file and train     ###
###   a random forest to disting water surface to       ###
###   ground to some stuff                              ###
###########################################################

## Goals:   - train a model with raster
##          - find the best scale range to separe water, ground and the rest

## Step:    1) load raster laz files
##          2) load laz files
##          3) compute descriptors on raster with the original file
##          4) keep only ground and water surface points with a good confidence
##             (like >= 0.9)

from laspy.file import File
from math import atan2
from sklearn.neighbors import KDTree
from function_las import *

import copy
import numpy as np
import scipy.linalg as la
import pandas as pd



import os
import sys




def main():
    dir_in = "../../data/rasterized_tiles/"
    
    filenames_train_C2 = ["tile_C2_868000_6524000_raster_2_5_low.laz",
                          "tile_C2_868000_6524500_raster_2_5_low.laz",
                          "tile_C2_868500_6523000_raster_2_5_low.laz",
                          "tile_C2_868500_6524000_raster_2_5_low.laz",
                          "tile_C2_869000_6523000_raster_2_5_low.laz",
                          "tile_C2_869000_6524000_raster_2_5_low.laz",
                          "tile_C2_869000_6525000_raster_2_5_low.laz",
                          "tile_C2_869500_6524500_raster_2_5_low.laz",
                          "tile_C2_869500_6525000_raster_2_5_low.laz"]
    
    filenames_train_C3 = ["tile_C3_868000_6524000_raster_2_5_low.laz",
                          "tile_C3_868000_6524500_raster_2_5_low.laz",
                          "tile_C3_868500_6523000_raster_2_5_low.laz",
                          "tile_C3_868500_6524000_raster_2_5_low.laz",
                          "tile_C3_869000_6523000_raster_2_5_low.laz",
                          "tile_C3_869000_6524000_raster_2_5_low.laz",
                          "tile_C3_869000_6525000_raster_2_5_low.laz",
                          "tile_C3_869500_6524500_raster_2_5_low.laz",
                          "tile_C3_869500_6525000_raster_2_5_low.laz"]
    
    filename_test_C2 = ["tile_C2_869000_6523500_raster_2_5_low.laz"]
    filename_test_C3 = ["tile_C3_869000_6523500_raster_2_5_low.laz"]
    
    for filename in filenames_train_C2:
        inFile_C2 = load_las_file(dir_in + filename)
        if (filename == filenames_train_C2[0]):
            nam_feat = name_features.copy()

            [foo, labels_train, column_names] = extract_features(inFile_C2, nam_feat, scales)
            data_train = foo
        else:
            nam_feat = name_features.copy()
            [foo, lab_tmp, column_names] = extract_features(inFile_C2, nam_feat, scales)
            data_train = np.vstack([data_train, foo])
            #foo = inFile_C2.classification
            labels_train = np.concatenate([labels_train, lab_tmp])
    
    return



if __name__ == "__main__":
    main()