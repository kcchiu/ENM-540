#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 13:19:56 2018

@author: Paris
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from models import LinearRegression
from optimizers import SGD, Adam, RMSprop

if __name__ == "__main__": 
    
    # N is the number of training points.
    # D_in is input dimension
    # D_out is output dimension.
    N, D_in, D_out = 64, 1, 1
    
    # Add some noise to the observations
    noise_var = 0.5
    
    # Create random input and output data
    X = lhs(D_in, N)
    y = 5*X + noise_var*np.random.randn(N,D_out)
    
    # Define the model
    model = LinearRegression(X,y)
       
    # Define an optimizer
    optimizer = SGD(model.num_params, lr = 1e-3, momentum = 0.9)
#    optimizer = Adam(model.num_params, lr = 1e-3)
#    optimizer = RMSprop(model.num_params, lr = 1e-3)
    
    # Train the model
    model.train(10000, optimizer)
       
    # Print the learned parameters
    print('w = %e, sigma_sq = %e' % (model.theta[:-1], np.exp(model.theta[-1])))
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Plot
    plt.figure(1)
    plt.plot(X,y,'o')
    plt.plot(X, y_pred)
    plt.show()