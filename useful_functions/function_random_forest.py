# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:02:19 2017

@author: Arthur Le Guennec
"""

from sklearn.ensemble import RandomForestClassifier

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'




def train_classifier(data, labels, verbose=True):
    """
    Train the classifier
    
    :params data: data X
    :params labels: data Y
    :params verbose: Controls the verbosity of the function.
    """
    
    if (verbose == True):
        verbose = 1
    else:
        verbose = 0
    
    labels = labels.reshape(-1, 1)
    random_forest = RandomForestClassifier(n_estimators = 100,
                                          criterion="gini",
                                          max_features="auto",
                                          oob_score=True,
                                          n_jobs=-1,
                                          verbose=1)
    

    random_forest.fit(data, labels)
    return random_forest


def test_classifier(model, data, labels=[]):
    """
    Test the classifier. If the parameter labels=[], then value of error rate 
    or mean confidence will not return.
    
    :params model: trained random forest
    :params data: data X
    :params labels: true label if it's possible.
    """
    
    labels_predict = model.predict(data)
    confid_predict = model.predict_proba(data)
    confid_predict = np.max(confid_predict, axis=1)


    if (len(labels) == np.shape(data)[0]):
        labels = labels.reshape(-1)
        foo = np.equal(labels, labels_predict)
        error_rate = np.count_nonzero(foo) / len(labels)
        confid_good = confid_predict[foo]
        mean_confid = np.mean(confid_good)
        return labels_predict, confid_predict, error_rate, mean_confid
    else:
        return labels_predict, confid_predict