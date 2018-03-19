#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 19:28:27 2018

@author: Paris
"""

"""
Solves the equation:
    u_xxxx = f    
"""
    
import numpy as np
from kernels import Beam_kernels
from pyDOE import lhs
import matplotlib.pyplot as plt

from models import BeamGP

np.random.seed(1234)

if __name__ == "__main__": 
    
    N_f = 50
    D = 1
    lb = 0.0*np.ones(D)
    ub = 1.0*np.ones(D)
    noise_f = 0.00
    
    # Forcing term
    def f(x):
        f = 10.0*(1.0-x)*np.sin(3.0*np.pi*(x-0.01))
        return f
    
    # Exact solution
    def u(x):
        u = (0.65479E-3+(-0.115342E-1)*x+0.659469E-2*x**2+0.176577E-2*x**3+ \
            (-0.65479E-3)*np.cos(0.942478E1*x)+0.119273E-3*x*np.cos(0.942478E1*x)+ \
            0.121116E-2*np.sin(0.942478E1*x)+(-0.126178E-2)*x*np.sin(0.942478E1*x))
        return u
    
    # Boundary condtions data
    X_u = np.array([0.0, 0.0, 1.0, 1.0])[:,None]
    y_u = np.array([0.0, 0.0, 0.0, 0.0])[:,None]
    
    # Forcing training data
    X_f = lb + (ub-lb)*lhs(D, N_f)
    y_f = f(X_f) + noise_f*np.random.randn(N_f,D)
    
    # Test data
    nn = 500
    X_star = np.linspace(lb, ub, nn)[:,None]
    u_star = u(X_star)
    f_star = f(X_star)
 
    # Compute required kernels
    k_uu, k_uu1, k_uu2, k_uu3, k_uf, \
    k_u1u1, k_u1u2, k_u1u3, k_u1f, \
    k_u2u2, k_u2u3, k_u2f, \
    k_u3u3, k_u3f, \
    k_ff = Beam_kernels()
    
    # Define model
    model = BeamGP(X_u, y_u, X_f, y_f, 
                   k_uu, k_uu1, k_uu2, k_uu3, k_uf,
                   k_u1u1, k_u1u2, k_u1u3, k_u1f,
                   k_u2u2, k_u2u3, k_u2f,
                   k_u3u3, k_u3f,
                   k_ff)
    
    # Training
    model.train()
    
    # New acquisition point
    UCB_min = model.DE_UCB_min(X_star)
    UCB_max = model.DE_UCB_max(X_star)
    v_min, v_max = UCB_min.min(), UCB_max.max()
    min_id, max_id = np.argmin(UCB_min), np.argmax(UCB_min)    
    if (np.abs(v_min) > np.abs(v_max)):
        new_X = X_star[min_id,:]
    else:
        new_X = X_star[max_id,:]
    
    # Prediction
    u_pred, u_var = model.predict_u(X_star)
    f_pred, f_var = model.predict_f(X_star)    
    u_var = np.abs(np.diag(u_var))[:,None]
    f_var = np.abs(np.diag(f_var))[:,None]
    
    # Plotting
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(X_star, u_star, 'b-', label = "Exact", linewidth=2)
    plt.plot(X_star, u_pred, 'r--', label = "Prediction", linewidth=2)
    lower = u_pred - 2.0*np.sqrt(u_var)
    upper = u_pred + 2.0*np.sqrt(u_var)
    plt.fill_between(X_star.flatten(), lower.flatten(), upper.flatten(), 
                     facecolor='orange', alpha=0.5, label="Two std band")
    ax = plt.gca()
    ax.set_xlim([lb[0], ub[0]])
    plt.xlabel('$x$')
    plt.ylabel('$u(x)$')
    
    plt.subplot(1,2,2)
    plt.plot(X_star, f_star, 'b-', label = "Exact", linewidth=2)
    plt.plot(X_star, f_pred, 'r--', label = "Prediction", linewidth=2)
    lower = f_pred - 2.0*np.sqrt(f_var)
    upper = f_pred + 2.0*np.sqrt(f_var)
    plt.fill_between(X_star.flatten(), lower.flatten(), upper.flatten(), 
                     facecolor='orange', alpha=0.5, label="Two std band")
    plt.plot(X_f,y_f,'bo', markersize = 8, alpha = 0.5, label = "Data")
    ax = plt.gca()
    ax.set_xlim([lb[0], ub[0]])
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    
    
   
    
