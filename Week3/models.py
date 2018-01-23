#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 17:33:54 2018

@author: Paris
"""

import autograd.numpy as np
from autograd import grad


class LinearRegression:
  """
    Linear regression model: y = (w.T)*x + \epsilon
    p(y|x,theta) ~ N(y|(w.T)*x, sigma^2), theta = (w, sigma^2)
  """
  # Initialize model class
  def __init__(self, X, y):
      
      self.X = X
      self.y = y
      self.n = X.shape[0]
      
      # Randomly initialize weights and noise variance
      w = np.random.randn(X.shape[1], y.shape[1])
      sigma_sq = np.array([np.log([1e-3])])
      
      # Concatenate all parameters in a single vector
      self.theta = np.concatenate([w.flatten(), sigma_sq.flatten()])
      
      # Count total number of parameters
      self.num_params = self.theta.shape[0]
      
      # Define loss gradient function using autograd
      self.grad_loss = grad(self.loss)
    
  # Evaluates the forward prediction of the model
  def forward_pass(self, w):
      y = np.matmul(self.X, w)
      return y

  # Evaluates the negative log-marginal likelihood loss, i.e. -log p(y|x,theta)
  def loss(self, theta): 
      # Fetch individual parameters from the theta vector and reshape if needed
      w = np.reshape(theta[:-1],(self.X.shape[1], self.y.shape[1]))
      sigma_sq = np.exp(theta[-1])
      # Evaluate the model's prediction
      y_pred = self.forward_pass(w)
      # Compute the loss
      NLML = 0.5 * self.n * np.log(2.0*np.pi*sigma_sq) + \
             0.5 * (np.sum(self.y - y_pred)**2) / sigma_sq
      return NLML
        
  # Given an optimizer, trains the model for a number of steps      
  def train(self, num_steps, optimizer):
      for it in range(0, num_steps):
          # Evaluate loss using current parameters
          theta = self.theta
          loss = self.loss(theta)
          print("Iteration: %d, loss: %.3e" % (it, loss))
          
          # Update parameters
          grad_theta = self.grad_loss(theta)
          self.theta = optimizer.step(theta, grad_theta)
          
  def predict(self, X_star):
      w = np.reshape(self.theta[:-1],(self.X.shape[1], self.y.shape[1]))
      y_pred = np.matmul(X_star, w)
      return y_pred