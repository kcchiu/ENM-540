#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:19:55 2018

@author: Paris
"""


import numpy as np
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
    
    # Randomly initialize weights
    w = np.random.randn(D_in, D_out)
    
    learning_rate = 1e-3
    for it in range(10000):
      # Forward pass: compute predicted y
      y_pred = np.matmul(X,w)
      
      # Compute and print loss
      loss = np.sum((y_pred - y)**2)
      print("Iteration: %d, loss: %f" % (it, loss))
      
      # Backprop to compute gradients of W with respect to loss
      grad_w = np.matmul(np.matmul(X.T,X),w) - np.matmul(X.T,y)
      
      # Update weights via gradient descent
      w = w - learning_rate * grad_w
      
    plt.figure(1)
    plt.plot(X,y,'o')
    plt.plot(X, y_pred)
    plt.show()
