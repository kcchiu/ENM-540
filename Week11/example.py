#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 22:45:30 2018

@author: Paris
"""

import numpy as np
import matplotlib.pyplot as plt

import sklearn.datasets

from models import PCA

if __name__ == "__main__":    

    X = sklearn.datasets.load_iris().data
    y = sklearn.datasets.load_iris().target

    Z_dim = X.shape[1]
    model = PCA(X, Z_dim)
    
    model.fit()
    
    Z = model.encode(X)
    X_star = model.decode(Z)
    
    error = np.linalg.norm(X-X_star,2)/np.linalg.norm(X,2)
    
    # Plot the projection onto the first two principal components
    plt.figure(1)
    plt.scatter(Z[:,0],Z[:,1], c = y)
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
