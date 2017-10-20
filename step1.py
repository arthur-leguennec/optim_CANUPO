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
from useful_functions.function_las import load_las_file, change_labelisation
from useful_functions.function_las import save_cloud
from useful_functions.function_canupo import compute_pca_for_corepoint
from useful_functions.function_canupo import compute_pca
from useful_functions.function_canupo import compute_feature_for_corepoint

import numpy as np
import scipy.linalg as la
import pandas as pd
import copy


import os
import sys


def just_change_label(inFile):
    '''
    Function for change label for this script
    '''
    change_labelisation(inFile, 
                        labels_in=[0],
                        label_out=1, 
                        verbose=True)
    change_labelisation(inFile,
                        labels_in=[3, 4, 20],
                        label_out=5, 
                        verbose=True)
    change_labelisation(inFile,
                        labels_in=[7],
                        label_out=[18],
                        verbose=True)
    
def get_points(inFile):
    '''
    Return an numpy array of points (coordinates, intensity, classification, ...)
    '''
    cloud = np.vstack([inFile.x,
                       inFile.y,
                       inFile.z,
                       inFile.classification,
                       inFile.intensity,
                       inFile.num_returns,
                       inFile.return_num]).transpose()
    
    points = np.require(cloud, 
                        dtype=np.float64, 
                        requirements=['C', 'A'])
    
    return points


def main():
    dir_raster_in = "../../data/rasterized_tiles/"
    dir_in = "../../data/labeled_tiles/"
    
    C3 = True

    #scales = [0.25, 0.5, 1, 2, 2.5, 3, 4, 5, 6, 7]
    scales = [2, 3]

    f = open('filename_data.txt', 'r')
    filenames_train_C2 = f.readlines()
    if (C3 == True):
        filenames_train_C3 = copy.copy(filenames_train_C2)
    filenames_train_raster_C2 = copy.copy(filenames_train_C2)
    for i in range(len(filenames_train_C2)):
        filenames_train_C2[i] = str(filenames_train_C2[i][0:-1]) #to delete '\n'
        foo = str(filenames_train_C2[i])
        if (C3 == True):
            filenames_train_C3[i] = foo.replace('C2', 'C3')
        filenames_train_raster_C2[i] = foo.replace('.laz', 
                                                   '_raster_2_5_low.las')
    f.close()
    
    
    f = open('features_C2C3.txt')
    names_features = f.readlines()
    for i in range(len(names_features)):
        names_features[i] = str(names_features[i][0:-1]) #to delete '\n'
    f.close()
    
    
    for i in range(len(filenames_train_C2)):
        # load laz file (point cloud of C2, C3 and C2 rasterized)
        inFile_C2 = load_las_file(dir_in + filenames_train_C2[i])
        if (C3 == True):
            inFile_C3 = load_las_file(dir_in + filenames_train_C3[i])
        inFile_C2_raster = load_las_file(dir_raster_in 
                                         + filenames_train_raster_C2[i],
                                         mode="rw")
        
        # we change some labels
        just_change_label(inFile_C2_raster)
        
        # Retrieve some element (coordinates (x, y, z), intensity, 
        # classification, ...)
        points_C2 = get_points(inFile_C2)
        if (C3 == True):
            points_C3 = get_points(inFile_C3)
        points_C2_raster = get_points(inFile_C2_raster)

        # Only the coordinates (useful for kdtree!)
        coord_C2 = points_C2[:, 0:3]
        if (C3 == True):
            coord_C3 = points_C3[:, 0:3]
        coord_C2_raster = points_C2_raster # No use this for kdtree

        num_returns = coord_C2_raster[:, 5]   # number of echo
        return_num = coord_C2_raster[:, 6]    # position of echo
        
        print("Compute KDTree C2 ... ", end='', flush=True)
        kdtree_C2 = KDTree(coord_C2, leaf_size=40)
        print("Done!")
        if (C3 == True):
            print("Compute KDTree C3 ... ", end='', flush=True)
            kdtree_C3 = KDTree(coord_C3, leaf_size=40)
            print("Done!")
            print("Compute KDTree C2+C3 ... ", end='', flush=True)
            coord_C2C3 = np.vstack([coord_C2, coord_C3])
            kdtree_C2C3 = KDTree(coord_C2C3, leaf_size=40)
            print("Done!")
        
        intensity_C2 = points_C2[:, 4]
        if (C3 == True):
            intensity_C3 = points_C3[:, 4]
        
        ##############################
        ## First feature computed ! ##
        ##############################
        ratio_echo = np.divide(return_num, num_returns)
        ratio_echo = ratio_echo.reshape(-1, 1)
        
        
        column_names = [attr + '_s' + str(r).replace('.', '_') 
                            for r in scales for attr in names_features]
        
        column_names = np.hstack(['ratio_echo', column_names])
        
        
        
        for index_radius, radius in enumerate(scales):
            print("Scale: " + str(radius))
            for index_point, point in enumerate(points_C2_raster):
                point_coord = point[0:3]
                
                ###########################
                ## compute pca and slope ##
                ###########################
                feat_one_point = []
                if (C3 == True):
                    feat_pca = compute_pca_for_corepoint(kdtree=kdtree_C2C3, 
                                                         point=point_coord, 
                                                         coordinates=coord_C2C3, 
                                                         radius=radius, 
                                                         verbose=False)
                else:
                    feat_pca = compute_pca_for_corepoint(kdtree=kdtree_C2, 
                                                         point=point_coord, 
                                                         coordinates=coord_C2, 
                                                         radius=radius, 
                                                         verbose=False)
                feat_one_point = feat_pca
                
                ################################
                ## compute intensity features ##
                ################################
                feat_intensity = compute_feature_for_corepoint(kdtree=kdtree_C2, 
                                                               point=point_coord, 
                                                               feature=intensity_C2, 
                                                               radius=radius, 
                                                               option='std', 
                                                               verbose=False)
                feat_one_point = np.hstack([feat_one_point, feat_intensity])
                
                if (C3 == True):
                    feat_intensity = compute_feature_for_corepoint(kdtree=kdtree_C3, 
                                                                   point=point_coord, 
                                                                   feature=intensity_C3, 
                                                                   radius=radius, 
                                                                   option='std', 
                                                                   verbose=False)
                    feat_one_point = np.hstack([feat_one_point, feat_intensity])
                    
                    feat_intensity = compute_feature_for_corepoint(kdtree=kdtree_C2, 
                                                                   point=point_coord, 
                                                                   feature=intensity_C2, 
                                                                   radius=radius, 
                                                                   option='mean', 
                                                                   verbose=False)
                    feat_tmp = copy.copy(feat_intensity)
                    
                    feat_intensity = compute_feature_for_corepoint(kdtree=kdtree_C3, 
                                                                   point=point_coord, 
                                                                   feature=intensity_C3, 
                                                                   radius=radius, 
                                                                   option='mean', 
                                                                   verbose=False)
                    
                    feat_one_point = np.hstack([feat_one_point, np.divide(feat_tmp, feat_intensity)])
                
                #############################
                ## compute height features ##
                #############################
                feat_height = compute_feature_for_corepoint(kdtree=kdtree_C2, 
                                                            point=point_coord, 
                                                            feature=coord_C2[:, 2], 
                                                            radius=radius, 
                                                            option='std', 
                                                            verbose=False)
                feat_one_point = np.hstack([feat_one_point, feat_height])
                
                if (C3 == True):
                    feat_height = compute_feature_for_corepoint(kdtree=kdtree_C3, 
                                                                point=point_coord, 
                                                                feature=coord_C3[:, 2], 
                                                                radius=radius, 
                                                                option='std', 
                                                                verbose=False)
                    feat_one_point = np.hstack([feat_one_point, feat_intensity])
                    
                    feat_height = compute_feature_for_corepoint(kdtree=kdtree_C2, 
                                                                point=point_coord, 
                                                                feature=coord_C2[:, 2], 
                                                                radius=radius, 
                                                                option='mean', 
                                                                verbose=False)
                    feat_tmp = copy.copy(feat_height)
                    
                    feat_height = compute_feature_for_corepoint(kdtree=kdtree_C3, 
                                                                point=point_coord, 
                                                                feature=coord_C3[:, 2], 
                                                                radius=radius, 
                                                                option='mean', 
                                                                verbose=False)
                    
                    feat_one_point = np.hstack([feat_one_point, np.subtract(feat_tmp, feat_height)])
                
                
                if index_point == 0:
                    feat_one_scale = feat_one_point
                else:
                    feat_one_scale = np.vstack([feat_one_scale, feat_one_point])
                    
            if index_radius == 0:
                feat = feat_one_scale
            else:
                feat = np.hstack([feat, feat_one_scale])
        features = {}
        
        feat = np.hstack([ratio_echo, feat])
        
        for i in range(len(column_names)):
            features[column_names[i]] = feat[:, i]
            
        save_cloud(inFile_C2_raster, 'here.laz', keep_ind=[], features=features)
    
    return
    
    

if __name__ == "__main__":
    main()