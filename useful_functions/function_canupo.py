# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 2017

@author: Arthur Le Guennec
"""

__all__ = ['compute_pca_for_corepoint', 'compute_pca', 'compute_feature_for_corepoint']


from math import atan2
from sklearn.neighbors import KDTree

import numpy as np
import scipy.linalg as la
import copy


def compute_pca_for_corepoint(index,
                              coordinates,
                              radius,
                              verbose=False):
    """Compute PCA descriptors around one point with several scales.
    
    Args:
        index: Index inside the ball (replace KDTree)
        point: the point where we will compute features. Have to be a 
               coordinate
        coordinates: list of coordinates.
        radius: size of desired neighborhood.
        verbose: Controls the verbosity of the function. False by default.
    
    Example:
        >> from laspy.file import File
        >> from sklearn.neighbors import KDTree
        >> import numpy as np
        >> inFile = File("input.laz", mode = "r")
        >> cloud = np.vstack([inFile.x,
                              inFile.y,
                              inFile.z,
                              inFile.intensity]).transpose()
        >> points = np.require(cloud, 
                               dtype=np.float64, 
                               requirements=['C', 'A'])
        >> intensity = points[:, 3]
        >> coord = points[:, 0:3]
        >> kdtree = KDTree(coord, leaf_size=40)
        >> index = kdtree.query_radius(coord[10, :], 2)[0]
        >> scales_features = compute_pca_for_corepoint(index,
                                                       coord[10, :],
                                                       coord)
    """
    lambda1s = []
    lambda2s = []
    lambda3s = []
    slopes = []

    [lambda1, 
     lambda2, 
     lambda3, 
     slope] = compute_pca(coordinates[index])
    lambda1s.append(lambda1)
    lambda2s.append(lambda2)
    lambda3s.append(lambda3)
    slopes.append(slope)
        
    desc_one_scale = np.vstack([lambda1s,
                                lambda2s,
                                lambda3s,
                                slope]).transpose()

    return desc_one_scale



def compute_pca(coord):
    """Function that compute PCA for a point cloud
    
    Args:
        coord: coordinates of each points
    
    Example:
        >> import numpy as np
        >> coor = np.random.rand(10000, 3) # point cloud of 10000 points
        >> [l1, l2, l3, slope] = compute_pca(coord)
    """

    nb_pts_radius = np.size(coord, axis=0)
    
    if (nb_pts_radius >= 3): # If we have less than 3 points, we can't
                             # compute PCA
        eigvals, eigvects = la.eigh(np.cov(coord.transpose()))
        eigvals = eigvals / la.norm(eigvals)
        
        eigvals = np.abs(eigvals)
        index = eigvals.argsort()[::-1]
        eigvects = eigvects[:, index]
        normal = eigvects[2]
        normal_x, normal_y, normal_z = normal
        
        [lambda3, lambda2, lambda1] = eigvals

        #Compute slope
        slope = atan2(normal_z, np.sqrt(normal_x*normal_x + normal_y*normal_y))
        slope = 90 - np.absolute(np.degrees(slope), dtype='f')
    else:
        [lambda3, lambda2, lambda1] = [np.NAN, np.NAN, np.NAN]
        slope = np.NAN
    
    return [lambda1, lambda2, lambda3, slope]


def compute_feature_for_corepoint(feature,
                                  option='mean',
                                  verbose=False):
    """Compute descriptors around one point with several scales.
    
    Args:
        points: numpy array
        feature: numpy array of the concerned features
        option: what kind of feature we want, by default it's the mean
                (possibility: mean or std)
        verbose: Controls the verbosity of the function. False by default.
    
    Example:
        >> from laspy.file import File
        >> from sklearn.neighbors import KDTree
        >> import numpy as np
        >> inFile = File("input.laz", mode = "r")
        >> cloud = np.vstack([inFile.x,
                             inFile.y,
                             inFile.z]).transpose()
        >> points = np.require(cloud, 
                               dtype=np.float64, 
                               requirements=['C', 'A'])
        >> intensity = points_C3[:, 3]
        >> coord = points[:, 0:3]
        >> kdtree = KDTree(coord, leaf_size=40)
        >> index = kdtree.query_radius(coord[10, :], 2)[0]
        >> scales_features = compute_feature_for_corepoint(intensity[index],
                                                           radius=2,
                                                           option='std')
    """
    point_tmp = np.array([point])
    point_tmp = point_tmp.reshape(1, -1)
    ind, dist = kdtree.query_radius(point_tmp,
                                    radius,
                                    return_distance=True,
                                    sort_results=True)
    
    ind = ind[0]
    dist = dist[0]
    
    feature_s = []
    feat = []
    
    # tmp is a list of indices of indices ...
    tmp = np.argwhere(dist < radius).reshape(-1)
    ind_in_radius = ind[tmp]
    
    # How many points we have in the sphere ?
    nb_pts_radius = len(ind_in_radius)
    
#    loc_feature = np.array(feature)
#    loc_feature = loc_feature[ind_in_radius]
    
    if (option == 'mean'):
        if (nb_pts_radius >= 3):
            feat = np.mean(feature[ind_in_radius], dtype='f')
        else:
            feat = np.NAN
    elif (option == 'std'):
        if (nb_pts_radius >= 3):
            feat = np.std(feature[ind_in_radius], dtype='f')
        else:
            feat = np.NAN
    
    feature_s.append(feat)
    desc_one_scale = np.array([feature_s])
            
    return desc_one_scale.reshape(1, -1)
