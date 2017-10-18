#!/usr/bin/env python



"""
LiDAR processing examples
"""


from __future__ import division
from laspy.file import File

import numpy as np
#import scipy.interpolate
import pandas as pd
import copy
import argparse


def save_cloud(inFile, filename, keep_ind = [], features = {}):
    '''
    Save a point cloud in laz or las file
    :params inFile: Base file object in laspy.
    :params filename: filename of the output.
    :params keep_ind: list of index to keep, if this parameter is empty , all 
                        the index are kept (by default, this parameter is empty)
    :params features: dictionnary of feature to add. The names of features are
                        the keys, and the values are a numpy array
                        
    :Example:
    
    >> from laspy.file import File
    >> import numpy as np
    >> inFile = File("input.laz", mode = "r")
    >> ind = np.argwhere(inFile.classification == 2) # we keep the ground
    >> save_cloud(inFile, "ground.laz", ind)
    '''
    
    # Problem in data, idk why but some labels are equal to 0
    label = inFile.classification
    ind = np.argwhere(label != 0).reshape(-1)
    
    if (keep_ind != []):
        keep_ind = np.array(keep_ind).reshape(-1) #to be sure
        keep_ind = np.hstack([ind, keep_ind])
        keep_ind = np.unique(keep_ind)

    # we kept the same header, rather to create a new one
    new_header = copy.copy(inFile.header)
    #modification of format if to access at RGB dimensions
    new_header.data_format_id = 3
    #Create outfile
    outFile = filename
    
    outCloud = File(outFile, 
                    mode = "w", 
                    header=new_header, 
                    vlrs = inFile.header.vlrs)


    for dimension in inFile.point_format:
        if (dimension == inFile.point_format[13]):
            break
        dat = inFile.reader.get_dimension(dimension.name)[keep_ind].reshape(-1)
        outCloud.writer.set_dimension(dimension.name, dat)

    # if we have some features to add ...
    if (len(features) > 0):
        for name_feat in features.keys():
            outCloud.define_new_dimension(name = name_feat,
                                      data_type = 9,
                                      description = "ntd")
    
        multiscale_features_raw = pd.DataFrame()
        features_raw = []
        name_columns = []
        for name_feat in features.keys():
            if (name_feat == features.keys()[0]):
                features_raw = features[name_feat].reshape(-1, 1)
            else:
                features_raw = np.hstack([features_raw, 
                                          features[name_feat].reshape(-1, 1)])
            
            name_columns.append(name_feat)
        
        features_raw = pd.DataFrame(features_raw)
        features_raw.columns = np.array(name_columns)
        
        multiscale_features_raw = pd.concat([multiscale_features_raw, features_raw], axis=1)
        
        for name_feat in features.keys():
            exec("outCloud." + name_feat + " = multiscale_features_raw.as_matrix([\'" + name_feat + "\']).ravel()")
    
    outCloud.close()
    

def extract_value(coord, option='minimum_height'):
    '''
    Return the indice of value of the list of coordinates according to the option
    
    :params coord: XYZ coordinates in a numpy array.
    :params option: according to the option, the returned value will be the
                    minimum (or maximum) altitude.
                    A random option it's possible too.
                    
    :Example:
    
    >> import numpy as np
    >> coord = np.random.rand(100, 3)
    >> ind_v = extract_value(coord, 'minimum_height') # ind_v will be the 
                                                        indice of the lowest 
                                                        point
    '''
    if (option == 'minimum_height'):
        return np.argmin(coord[:, 2])
    elif (option == 'maximum_height'):
        return np.argmax(coord[:, 2])
    elif (option == 'random_height'):
        return np.random.randint(np.shape(coord)[0] - 1)
    
    
def construct_grid(coord, option='minimum_height', step=2.5):
    '''
    Return an array in 2 dimensions where each element is an indice.
    This array correspond at the raster.
    Warning, this function is operational, but tool lika 'lasthin' of LASTools
    are more efficient
    
    :params coord: XYZ coordinates in a numpy array.
    :params step: size of each cell. By default, 2.5x2.5 is given
    
    :Example:
    
    >> import numpy as np
    >> coord = np.random.rand(10000, 3)
    >> ind_grid = construct_grid(coord, step=1.0)
    >> # ind_grid[0] will return a list of indice, representing the indice of 
    >> # coordinates in the cell #0.
    '''
    
    x_min = np.min(coord[:, 0])
    y_min = np.min(coord[:, 1])
    x_max = np.max(coord[:, 0])
    y_max = np.max(coord[:, 1])
    
    number_of_grid_x = int(np.floor((x_max - x_min) / step))
    number_of_grid_y = int(np.floor((y_max - y_min) / step))
    print("Size of raster: " + str(number_of_grid_x) + "x" + str(number_of_grid_y))
    
    list_of_ind = np.ndarray(shape=(number_of_grid_x, number_of_grid_y),
                             dtype=int)
    
    i = 0
    j = 0
    for i in range(number_of_grid_x):
        y_min = np.min(coord[:, 1])
        print(str(i) + "x" + str(j))
        for j in range(number_of_grid_y):
            ind_in_grid = np.argwhere((coord[:, 0]>=x_min)&(coord[:, 0]<x_min + step)&
                                      (coord[:, 1]>=y_min)&(coord[:, 1]<y_min + step))
            ind_in_grid = ind_in_grid.reshape(-1)
            if (len(ind_in_grid) > 0):
                ind = extract_value(coord[ind_in_grid, :], option).reshape(-1)
                list_of_ind[i, j] = ind_in_grid[ind]
            else:
                list_of_ind[i, j] = -1
            y_min = y_min + step
        x_min = x_min + step
        
    list_of_ind.reshape(-1)
    ind_to_del = np.argwhere(list_of_ind == -1).reshape(1)
    list_of_ind = list_of_ind[ind_to_del].reshape(-1)
    list_of_ind = np.unique(list_of_ind)
    list_of_ind = list_of_ind[~np.isnan(list_of_ind)]
    return list_of_ind


def extract_feature(inFile, name_feature):
    '''
    Extract one feature according the name given in parameters
    
    :params inFile: Base file object in laspy.
    :params name_feature: Name of the feature
    
    :Example:
    
    >> from laspy.file import File
    >> import numpy as np
    >> inFile = File("input.laz", mode = "r")
    >> feat = extract_feature(inFile, 'ratio_echo') # ratio echo being a 
    >>                                              # feature created by us
    '''
    label = inFile.classification

    # Problème dans les données, certaines labels sont à 0
    ind = np.argwhere(label != 0)

    label = label[ind]

    change_labelisation(inFile,
                        labels_in=[3, 4, 20],
                        label_out=5,
                        verbose=True)
    change_labelisation(inFile,
                        labels_in=7,
                        label_out=18,
                        verbose=True)

    #Modification des labels pour simplifier (ex: basse, moyenne et haute vegetation => classe vegetation)
    ind_tmp = np.argwhere(label == 3)
    label[ind_tmp] = 5  # low vegetation to high vegetation
    ind_tmp = np.argwhere(label == 4)
    label[ind_tmp] = 5  # medium to high vegetation
    ind_tmp = np.argwhere(label == 20)
    label[ind_tmp] = 5  # shrunk to high vegetation
    ind_tmp = np.argwhere(label == 7)
    label[ind_tmp] = 18  # low noise to high noise
    del ind_tmp

    feature = inFile.reader.get_dimension(name_feature)[ind].reshape(-1)

    if (name_feature[0:16] == "diff_mean_height"):
        feature = np.nan_to_num(feature)
    else:
        feature = feature.copy()
        ind_nan = np.isnan(feature)
        feature[ind_nan] = -1

    feature = feature.transpose()

    return feature


def change_labelisation(inFile, labels_in=[], label_out=1, verbose=False):
    '''
    change_labelisation is a function that change the classification if needs.
    
    :params inFile: Base file object in laspy.
    :params labels_in: list of labels that have to change.
    :params label_out: desired label.
    :params verbose: Controls the verbosity of the function.
    
    :Example:
    >> from laspy.file import File
    >> import numpy as np
    >> inFile = File("input.laz", mode = "r")
    >> change_labelisation(inFile, [3,4], 5) # According to the ISPRS notation,
    >>                                       # this fucntion will put low and
    >>                                       # medium vegatation in high
    >>                                       # vegetation
    '''
    if (verbose == True or verbose > 0):
        print("We change label " + str(labels_in) + " to " + str(label_out))
        
    label = inFile.classification
    shape_label = np.shape(label)
    
    for label_in in labels_in:
        ind = np.argwhere(label == label_in)
        label[ind] = label_out
        label.reshape(shape_label)
        inFile.classification = label

    
    

def load_las_file(name_dir_file):
    '''
    Load and return the point cloud. 
    The value returned is a base file object in laspy.
    
    :params name_dir_file: filename of the las or laz file
    
    :Example:
    
    >> from laspy.file import File
    >> inFile = load_las_file("input.laz")
    Loading file : input.laz
    >>
    '''
    print('Loading file : ' + name_dir_file)
    inFile = File(name_dir_file, mode = "r")
    return inFile


def rasterize_z(filename, option, step):
    inFile = load_las_file(filename)
    
    cloud = np.vstack([inFile.x,
                       inFile.y,
                       inFile.z]).transpose()
    
    points = np.require(cloud, dtype=np.float64, requirements=['C', 'A'])
    
    coord = points[:, 0:3]
    
    ind = construct_grid(coord, option, step)
    
    save_cloud(inFile, 'output.laz', keed_ind=ind)    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-filename')
    parser.add_argument('-option')
    parser.add_argument('-step')
    
    args = parser.parse_args()
    
        
    rasterize_z(args.filename, args.option, float(args.step))
    
    
    
    