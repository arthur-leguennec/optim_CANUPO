# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 2017

@author: Arthur Le Guennec
"""


from math import atan2
from sklearn.neighbors import KDTree

import numpy as np
import scipy.linalg as la


def compute_pca_for_corepoint(kdtree, 
                              point,
                              coordinates,
                              radius,
                              verbose=False):
    '''
    Compute PCA descriptors around one point with several scales. According to
    parameters, it's possible to used several point cloud (C2 and C3 here)
    
    :params kdtree: KDTree computed with sklearn.neighbors. Can be a list of 
                    KDTree (in case where we have several PC) or only one
    :params point: the point where we will compute features. Have to be a 
                   coordinate
    :params coordinates: list of coordinates. If there is more than one point 
                         cloud, just need of coordinates=[coord1, coord2]
    :params radius: size of desired neighborhood
    :params verbose: Controls the verbosity of the function. False by default.
    
    :Example:
    
    >> from laspy.file import File
    >> from sklearn.neighbors import KDTree
    >> import numpy as np
    >> inFile_C2 = File("input_C2.laz", mode = "r")
    >> inFile_C3 = File("input_C3.laz", mode = "r")
    >> cloud_C2 = np.vstack([inFile_C2.x,
                             inFile_C2.y,
                             inFile_C2.z,
                             inFile_C2.intensity]).transpose()
    >> cloud_C3 = np.vstack([inFile_C3.x,
                             inFile_C3.y,
                             inFile_C3.z,
                             inFile_C3.intensity]).transpose()
    >> points_C2 = np.require(cloud_C2, 
                              dtype=np.float64, 
                              requirements=['C', 'A'])
    >> points_C3 = np.require(cloud_C3, 
                              dtype=np.float64, 
                              requirements=['C', 'A'])
    >> intensity_C2 = points_C2[:, 3]
    >> intensity_C3 = points_C3[:, 3]
    >> coord_C2 = points_C2[:, 0:3]
    >> coord_C3 = points_C3[:, 0:3]
    >> kdtree_C2 = KDTree(coord_C2, leaf_size=40)
    >> kdtree_C3 = KDTree(coord_C3, leaf_size=40)
    >> kdtree
    >> scales_features = compute_pca_for_corepoint([kdtree_C2, kdtree_C3],
                                                   coord_C2[10, :],
                                                   [1, 2, 3])
    '''
    ind = []
    dist = []
    for i in range(len(kdtree)):
        ind[i], dist[i] = kdtree[i].query_radius(point,
                                                 radius,
                                                 return_distance=True,
                                                 sort_results=True)
        ind[i] = ind[i][0]
        dist[i] = dist[i][0]
 
    lambda1s = []
    lambda2s = []
    lambda3s = []
    slopes = []
    ind_in_radius = []
    for i in range(len(kdtree)):
        # tmp is a list of indices of indices ...
        tmp = np.argwhere(dist[i] < radius).reshape(-1)
        ind_in_radius.append(ind[i][tmp])
    
    # How many points we have in the sphere ?
    nb_pts_radius = 0
    for ind in ind_in_radius:
        nb_pts_radius = nb_pts_radius + len(ind)
        
    for coord in coordinates:
        if (coord == coordinates[0]):
            coord_all = coordinates[0]
        else:
            coord_all = np.vstack([coord_all, coord])
    
    [lambda1, lambda2, lambda3, slope] = compute_pca(coord_all)
    
    lambda1s.append(lambda1)
    lambda2s.append(lambda2)
    lambda3s.append(lambda3)
    slopes.append(slope)
        
    desc_one_scale = np.vstack([lambda1s,
                                lambda2s,
                                lambda3s]).transpose()
            
    return desc_one_scale



def compute_pca(coord):
    '''
    Function that compute PCA for a point cloud
    
    :params coord: coordinates of each points
    
    :Example:
    
    >> import numpy as np
    >> coor = np.random.rand(10000, 3) # point cloud of 10000 points
    >> [l1, l2, l3, slope] = compute_pca(coord)
    '''

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
        slope = 90 - np.absolute(np.degrees(atan2(slope)), dtype='f')
    else:
        [lambda3, lambda2, lambda1] = [np.NAN, np.NAN, np.NAN]
        slope = np.NAN
    
    return [lambda1, lambda2, lambda3, slope]


def compute_feature_for_corepoint(kdtree, 
                                  point, 
                                  feature, 
                                  radius, 
                                  option='mean',
                                  verbose=False):
    '''
    Compute descriptors around one point with several scales. According to
    parameters, it's possible to used several point cloud (C2 and C3 here)
    
    :params kdtree: KDTree computed with sklearn.neighbors. Can be a list of 
                    KDTree (in case where we have several PC) or only one
    :params point: the point where we will compute features. Have to be a 
                   coordinate
    :params feature: numpy array of the concerned features
    :params radius: size of desired neighborhood
    :params option: what kind of feature we want, by default it's the mean
                    (possibility: mean or std)
    :params verbose: Controls the verbosity of the function. False by default.
    
    :Example:
    
    >> from laspy.file import File
    >> from sklearn.neighbors import KDTree
    >> import numpy as np
    >> inFile_C2 = File("input_C2.laz", mode = "r")
    >> inFile_C3 = File("input_C3.laz", mode = "r")
    >> cloud_C2 = np.vstack([inFile_C2.x,
                             inFile_C2.y,
                             inFile_C2.z,
                             inFile_C2.intensity]).transpose()
    >> cloud_C3 = np.vstack([inFile_C3.x,
                             inFile_C3.y,
                             inFile_C3.z,
                             inFile_C3.intensity]).transpose()
    >> points_C2 = np.require(cloud_C2, 
                              dtype=np.float64, 
                              requirements=['C', 'A'])
    >> points_C3 = np.require(cloud_C3, 
                              dtype=np.float64, 
                              requirements=['C', 'A'])
    >> in_C2 = points_C2[:, 3]
    >> in_C3 = points_C3[:, 3]
    >> coord_C2 = points_C2[:, 0:3]
    >> coord_C3 = points_C3[:, 0:3]
    >> kdtree_C2 = KDTree(coord_C2, leaf_size=40)
    >> kdtree_C3 = KDTree(coord_C3, leaf_size=40)
    >> kdtree
    >> scales_features = compute_feature_for_corepoint([kdtree_C2, kdtree_C3],
                                                       coord_C2[10, :],
                                                       [in_C2, in_C3],
                                                       radius=2)
    '''
    ind = []
    dist = []
    for i in range(len(kdtree)):
        ind[i], dist[i] = kdtree[i].query_radius(point,
                                                 radius,
                                                 return_distance=True,
                                                 sort_results=True)
        
        ind[i] = ind[i][0]
        dist[i] = dist[i][0]
    
    feature_s = []
    feat = []
    ind_in_radius = []
    for i in range(len(kdtree)):
        feature_s.append([])
        # tmp is a list of indices of indices ...
        tmp = np.argwhere(dist[i] < radius).reshape(-1)
        ind_in_radius.append(ind[i][tmp])
    
    # How many points we have in the sphere ?
    nb_pts_radius = 0
    for ind in ind_in_radius:
        nb_pts_radius = nb_pts_radius + len(ind)
    
    if (option == 'mean'):
        for i in range(len(feature_s)):
            if (nb_pts_radius <= 3):
                feat.append(np.mean(feature, dtype='f'))
            else:
                feat.append(np.NAN)
    elif (option == 'std'):
        for i in range(len(feature_s)):
            if (nb_pts_radius <= 3):
                feat.append(np.std(feature, dtype='f'))
            else:
                feat.append(np.NAN)
    
    for i in range(len(feature_s)):
        feature_s[i].append(feat)
        if (i == 0):
            desc_one_scale = feature_s[i]
        else:
            desc_one_scale = np.vstack([desc_one_scale,
                                        feature_s[i]]).transpose()

    desc_all_scale = desc_one_scale
            
    return desc_all_scale
