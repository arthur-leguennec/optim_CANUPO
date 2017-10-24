# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:03:38 2017

@author: Arthur Le Guennec
"""
__all__ = ['plot_confusion_matrix', 'plot_feature_importance']


import itertools
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Args: 
        cm: confusion matrix
        classes: name of classes used
        mormalize: if True, result will be normalized (useful if classes 
                       are very unbalanced). By default, normalize=False
        title: Title of plot
        cmap: Color range (by default, blue)
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    cm = np.nan_to_num(cm)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




def plot_feature_importance(feat_imp, column_names, indices=[],
                            title='Features importances'):
    """This function prints and plots the features importances.
    
    Args:
        feat_imp: List of feature importances
        column_names: Name of features
        indices: Desired order of features importance. 
                     By default indices=[]
        title: Title of plot
    """

    if (indices == []):
        indices = range(len(feat_imp))

    plt.title(title)
    plt.bar(range(len(feat_imp)),
            feat_imp[indices],
            align="center",
            color="r")

    foo = column_names.copy()
    for i in range(len(indices)):
        column_names[i] = foo[indices[i]]
    column_names = tuple(column_names)

    plt.xticks(range(len(feat_imp)), column_names, rotation=90)
    plt.xlim([-1,len(feat_imp)])
    
    
    
def plot_simple_plot(data_Y, 
                     labels_X = [], 
                     title='Title by default'):
    """Function to plot a simple plot
    
    Args:
        data_Y: data
        labels_X: name of the label, [] by default
        title: NTD
    """    
    plt.figure()
    plt.title(title)
    
    line = plt.plot(range(len(data_Y)), data_Y)
    plt.setp(line, xticklabels=[labels_X])
    plt.show()
    
    
    
    
    
    
    