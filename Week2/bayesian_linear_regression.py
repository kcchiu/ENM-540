#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 22:44:02 2018

@author: Paris
"""

"""
Bayesian Linear Regression
-----------------
Bayesian Linear Regression with toy data in 1D.
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


# Generate data
N = 20

X = np.linspace(0, 2, num=N)[:,None]
# X = np.array([5, 14, 19], dtype=np.float)
t_real = np.sin(X)
t = t_real + np.random.randn(N,1) * 0.25

plt.figure(1)
plt.scatter(X, t, label='Data points')

# Generative process: p(t|W,X,beta) = N(t|XW+b,beta)
beta = 1

# Prior: N(w|0,1/alpha*I)
alpha = 1

# Posterior: N(w|m,s):
s = 1/(alpha + beta * np.matmul(X.T,X))
m = beta * s * np.matmul(X.T,X)

# Infer p(t|t,alpha,beta) the predictive distribution
X_pred = np.linspace(0, 2, num=100)[:,None]

m_pred = m * X_pred
s_pred = 1/beta + np.matmul(X_pred.T,X_pred) * s
std_pred = np.sqrt(s_pred)

plt.plot(X_pred, m_pred, color='red', alpha=0.75, label='Regression line')
upper = m_pred+std_pred
lower = m_pred-std_pred
plt.fill_between(
    X_pred.flatten(), lower.flatten(), upper.flatten(),
    interpolate=True, color='green', alpha=0.1, label='+- 1 stddev'
)

# Sample from predictive dist.
#ys = np.random.normal(m_pred, std_pred)
#
#plt.plot(X_pred, ys, alpha=0.15, label='Posterior samples')
#plt.legend(loc='best')
#plt.show()