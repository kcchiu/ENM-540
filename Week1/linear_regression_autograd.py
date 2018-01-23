#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:19:55 2018

@author: Paris
"""


import autograd.numpy as np
from autograd import grad

import matplotlib.pyplot as plt
from pyDOE import lhs

if __name__ == "__main__": 
    
    # N is the number of training points.
    # D_in is input dimension
    # D_out is output dimension.
    N, D_in, D_out = 64, 1, 1
    
    # Create random input and output data
    X = lhs(D_in, N)
    y = 5*X + np.random.randn(N,D_out)
    
    # Define loss
    def loss(w):
        y_pred = np.matmul(X,w)
        return np.sum((y_pred - y)**2)
    
    # Compute gradients using autograd
    grad_loss = grad(loss)
    
    # Randomly initialize weights
    w = np.random.randn(D_in, D_out)
    
    learning_rate = 1e-3
    for it in range(1000):
      print("Iteration: %d, loss: %f" % (it, loss(w)))
      w = w - learning_rate * grad_loss(w)
      
    y_pred = np.matmul(X,w)
    
    plt.figure(1)
    plt.plot(X,y,'o')
    plt.plot(X, y_pred)
    plt.show()
