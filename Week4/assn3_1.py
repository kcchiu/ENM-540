#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:20:02 2018

@author: Paris
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.interpolate import griddata

from models_tf import NeuralNetwork
#from models_pytorch import NeuralNetwork


def main(N):

    N = 1000
    layers = [2,50,50,1]
    noise = 0.0
    
    # Generate Training Data   
    def f(x,y):
        return np.cos(np.pi*x)*np.cos(np.pi*y)
    
    # Specify input domain bounds
    lb = 50.0*np.ones((1,2))
    ub = 54.0*np.ones((1,2)) 
    
    # Generate data
    X = lb + (ub-lb)*lhs(2, N)
    Y = f(X[:,0], X[:,1])[:,None] + noise*np.random.randn(N,1)
    
    # Generate Test Data
    N_star = 1000
    X_star = lb + (ub-lb)*lhs(2, N_star)
    Y_star = f(X_star[:,0], X_star[:,1])[:,None]
            
    # Create model
    model = NeuralNetwork(X, Y, layers)
        
    # Training
    model.train(nIter = 80000, batch_size = 100)
    
    # Prediction
    Y_pred = model.predict(X_star)
    
    # Relative L2 error
    error = np.linalg.norm(Y_star - Y_pred, 2)/np.linalg.norm(Y_star, 2)

    # Plotting
#    lb = X_star.min(0)
#    ub = X_star.max(0)
#    nn = 200
#    x = np.linspace(lb[0], ub[0], nn)
#    y = np.linspace(lb[1], ub[1], nn)
#    X, Y = np.meshgrid(x,y)
#    
#    Y_star_plot = griddata(X_star, Y_star.flatten(), (X, Y), method='cubic')
#    Y_pred_plot = griddata(X_star, Y_star.flatten(), (X, Y), method='cubic')
#    
#    plt.figure(1)
#    plt.subplot(1,3,1)
#    plt.pcolor(X,Y,Y_star_plot, cmap = 'jet')
#    plt.title('Exact')
#    plt.colorbar()
#    plt.subplot(1,3,2)
#    plt.pcolor(X,Y,Y_pred_plot, cmap = 'jet')
#    plt.title('Prediction')
#    plt.colorbar()
#    plt.subplot(1,3,3)
#    plt.pcolor(X,Y,np.abs(Y_star_plot-Y_pred_plot), cmap = 'jet')
#    plt.title('Absolute error')
#    plt.colorbar()
    
    return error
    
   

if __name__ == "__main__": 
    
    N = [100, 250, 500, 1000, 2500, 5000, 5000, 6000, 7000, 8000, 9000, 10000]
    
    error = np.zeros(len(N))
    
    for i in range(0,len(N)):
        error[i] = main(N[i])
    
    plt.figure(1)
    plt.plot(N, np.log(error), '-b*')
    plt.xlabel('$N$')
    plt.ylabel('log(error)')
    plt.legend()
    
