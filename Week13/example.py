#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 18:02:26 2018

@author: Paris
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

from models_tf import GAN

np.random.seed(1234)
    
if __name__ == "__main__":
    
    N = 2000
    X_dim = 2
    Z_dim = 2
    
    # generate synthetic data    
    def circle(N):    
        x = np.random.randn(N,2)
        return x/10 + x/np.linalg.norm(x,2,axis = 1, keepdims = True)
    
    def banana(N): 
        alpha = 1.15
        beta = 0.5
        mean = np.zeros(2)
        cov = np.array([[1.0, 0.9], [0.9, 1.0]])
        z = np.random.multivariate_normal(mean, cov, N)
        x1 = alpha*z[:,0:1]
        x2 = (z[:,1:2]/alpha) + beta*(x1**2 + alpha**2)
        return np.concatenate([x1,x2], axis = 1)
    
    X = circle(N)
       
    # Model creation
    layers_G = np.array([Z_dim,50,50,50,X_dim])       
    layers_D = np.array([X_dim,50,50,50,1])
    model = GAN(X, layers_G, layers_D)
        
    model.train(nIter = 50000, batch_size = 100)
        
    X_star = model.generate_samples(2000)
    
    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 14})   
    plt.scatter(X[:,0], X[:,1], color='blue', alpha = 0.2)
    plt.scatter(X_star[:,0], X_star[:,1], color='red', alpha = 0.2)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    
    