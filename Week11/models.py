#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 20:56:05 2017

@author: Paris
"""

import numpy as np

class PCA:
    # Initialize the class
    def __init__(self, X, Z_dim):    
        # Normalize
        self.Xmean, self.Xstd = X.mean(0), X.std(0)
        X = (X - self.Xmean)/self.Xstd  
        
        self.X = X
        self.N = X.shape[0]
        self.Z_dim = Z_dim
        
        # Sample covariance matrix
        self.S = np.matmul(X.T,X)/(self.N-1)
        
    # Computes the eigendecomposition of the covariance
    def fit(self):
        values, vectors = np.linalg.eig(self.S)
        idx = np.argsort(values)[::-1]
        self.values = values[idx[:self.Z_dim]]
        self.vectors = vectors[:,idx[:self.Z_dim]]
        
    # Encodes data into latent space
    def encode(self, X_star):
        X_star = (X_star - self.Xmean)/self.Xstd 
        Z = np.matmul(self.X, self.vectors)
        return Z
    
    # Decodes latent values back to physical space
    def decode(self, Z):
        X = np.matmul(Z, self.vectors.T)
        # De-normalize
        X = X*self.Xstd + self.Xmean
        return X