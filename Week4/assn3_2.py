#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:20:02 2018

@author: Paris
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

from models_tf import PDEnet
#from models_pytorch import PDEnet


def NNsolver(N_f):

    layers = [1,50,50,1]
    
    # Generate Training Data   
    def f(x):
        return -(np.pi**2 + 1.0)*np.sin(np.pi*x)
    
    # Specify input domain bounds
    lb = -1.0*np.ones((1,1))
    ub = 1.0*np.ones((1,1)) 
    
    # Given data on u
    X_u = np.array([-1.0, 1.0])[:,None]
    Y_u = np.array([0.0, 0.0])[:,None]
    
    # Generate data for f
    X_f = lb + (ub-lb)*lhs(1, N_f)
    Y_f = f(X_f)
    
    # Generate Test Data
    N_star = 1000
    X_star = lb + (ub-lb)*np.linspace(0,1,N_star)[:,None]
    u_star = np.sin(np.pi*X_star)
    f_star = f(X_star)

    # Create model
    model = PDEnet(X_u, Y_u, X_f, Y_f, layers)
    
    # Train
    model.train()
    
    # Predict at test points
    u_pred = model.predict_u(X_star)
    f_pred = model.predict_f(X_star)
    
    # Relative L2 error
    error_u = np.linalg.norm(u_star - u_pred, 2)/np.linalg.norm(u_star, 2)
    error_f = np.linalg.norm(f_star - f_pred, 2)/np.linalg.norm(f_star, 2)
    
    # Plot
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(X_star, u_star, '-k', linewidth = 2)
    plt.plot(X_star, u_pred, '--r', linewidth = 2)
    plt.plot(X_u, Y_u, 'o')
    plt.xlabel('$x$')
    plt.xlabel('$u(x)$')
    plt.subplot(1,2,2)
    plt.plot(X_star, f_star, '-k', linewidth = 2)
    plt.plot(X_star, f_pred, '--r', linewidth = 2)
    plt.plot(X_f, Y_f, 'o')
    plt.xlabel('$x$')
    plt.xlabel('$f(x)$')
    
    return error_u, error_f
    

if __name__ == "__main__": 
    
#    N_f = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    N_f = [1000]
    
    error_u = np.zeros(len(N_f))
    error_f = np.zeros(len(N_f))
    
    for i in range(0,len(N_f)):
        error_u[i], error_f[i] = NNsolver(N_f[i])
    
#    plt.figure(1)
#    plt.plot(N_f, np.log(error_u), '-b*', label = 'u(x)')
#    plt.plot(N_f, np.log(error_f), '-ro', label = 'f(x)')
#    plt.xlabel('$N_f$')
#    plt.ylabel('log(error)')
#    plt.legend()
    
