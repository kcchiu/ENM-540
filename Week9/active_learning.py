#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:02:36 2017

@author: Paris
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from models import GPRegression

np.random.seed(1234)


if __name__ == "__main__":    
    
    N = 10
    D = 1
    lb = -0.5*np.ones(D)
    ub = 1.0*np.ones(D)
    noise = 0.00
    tol = 1e-6
    nsteps = 20
    
    def f(x):
        return x * np.sin(4.0*np.pi*x)
  
    # Training data    
    X = lb + (ub-lb)*lhs(D, N)
    y = f(X) + noise*np.random.randn(N,D)
    
    # Test data
    nn = 200
    X_star = np.linspace(lb, ub, nn)[:,None]
    y_star = f(X_star)
    
    # Define model
    model = GPRegression(X, y)
    
    plt.figure(1, facecolor = 'w')
    
    for i in range(0,nsteps):
        # Train 
        model.train()
        
        # Predict
        y_pred, y_var = model.predict(X_star)
        y_var = np.abs(np.diag(y_var))[:,None]
        
        # Sample where posterior variance is maximized
        new_X = X_star[np.argmax(y_var),:][:,None]
        new_y = f(new_X) + noise*np.random.randn(1,D)

        # Check for convergence
        if np.max(y_var) < tol:
            print("Converged!")
            break
        
           
        # Plot
        plt.cla()
        plt.plot(X_star, y_star, 'b-', label = "Exact", linewidth=2)
        plt.plot(X_star, y_pred, 'r--', label = "Prediction", linewidth=2)
        lower = y_pred - 2.0*np.sqrt(y_var)
        upper = y_pred + 2.0*np.sqrt(y_var)
        plt.fill_between(X_star.flatten(), lower.flatten(), upper.flatten(), 
                         facecolor='orange', alpha=0.5, label="Two std band")
        plt.plot(X,y,'bo', label = "Data")
        plt.plot(new_X, 0.0,'m*')
        ax = plt.gca()
        ax.set_xlim([lb[0], ub[0]])
        ax.set_ylim([-1,1])
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title("Iteration #%d" % (i+1))
        plt.pause(0.5)
        plt.savefig("./figures/AL_it_%d.png" % (i+1), format='png', dpi=300)
        
        # Add new point to the training set
        X = np.concatenate([X, new_X], axis = 0)
        y = np.concatenate([y, new_y], axis = 0)
        
        model = GPRegression(X, y)



   
   